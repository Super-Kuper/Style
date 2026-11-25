"""HTTP-клиент ComfyUI (`contract/worker-contract.md` §1).

Штатный сценарий handler'а: ``wait_ready`` → ``submit`` → ``wait`` →
``fetch_output``; отмена — ``interrupt``. Транспорт инъектируется
(``httpx.AsyncClient``), поэтому тесты идут на ``httpx.MockTransport`` без
сети и GPU.

Воркер не строит и не редактирует граф: ``submit`` отправляет ``prompt``
байт-в-байт тем, что пришёл в ``input.prompt``.

ComfyUI доступен только на `127.0.0.1` внутри воркера, наружу не
публикуется; здесь нет ни авторизации, ни секретов.
"""

from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import time
import uuid
from dataclasses import dataclass
from typing import Any

import httpx

from .config import Config
from .errors import WorkerError

_LOG = logging.getLogger(__name__)

# Ключи outputs узлов, в которых ComfyUI кладёт картинку. SaveImage пишет в
# "images"; "gifs" оставлен для совместимых save-нод с тем же форматом записи.
_OUTPUT_KEYS = ("images", "gifs")

#: Как часто ожидание спрашивает ``GET /queue``, чтобы записать в лог переход
#: «в очереди → выполняется». Сам опрос истории идёт чаще; очередь нужна только
#: для наблюдаемости холодного первого кадра, поэтому 5 секунд достаточно.
QUEUE_POLL_SECONDS = 5.0


@dataclass(frozen=True)
class FileOutput:
    """Скачанный из ComfyUI файл: байты + имя + content-type."""

    data: bytes
    filename: str
    content_type: str = "application/octet-stream"


def _describe_error_body(response: httpx.Response) -> str:
    """Короткий текст ошибки ComfyUI для сообщения (JSON error/node_errors)."""

    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:500]
    if not isinstance(body, dict):
        return str(body)[:500]
    parts: list[str] = []
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("type") or error.get("details")
        if message:
            parts.append(str(message))
    elif error:
        parts.append(str(error))
    node_errors = body.get("node_errors")
    if isinstance(node_errors, dict) and node_errors:
        for node_id, node_error in node_errors.items():
            parts.append(f"node {node_id}: {node_error}")
    return "; ".join(parts)[:1000] or response.text.strip()[:500]


def _describe_status_error(status: dict[str, Any]) -> str:
    """Текст ошибки из ``history[prompt_id].status`` (ComfyUI execution_error)."""

    messages = status.get("messages")
    parts: list[str] = []
    if isinstance(messages, list):
        for item in messages:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                kind, data = item
                if isinstance(data, dict):
                    text = (
                        data.get("exception_message")
                        or data.get("exception_type")
                        or data.get("message")
                        or json.dumps(data, ensure_ascii=False)
                    )
                    parts.append(f"{kind}: {text}")
                else:
                    parts.append(f"{kind}: {data}")
            else:
                parts.append(str(item))
    if not parts:
        parts.append(json.dumps(status, ensure_ascii=False))
    return "; ".join(parts)[:2000]


def _first_output_file(history_entry: Any) -> dict[str, Any] | None:
    """Первый файл из ``outputs[*]["images"|"gifs"]``."""

    if not isinstance(history_entry, dict):
        return None
    outputs = history_entry.get("outputs")
    if not isinstance(outputs, dict):
        return None
    for node_output in outputs.values():
        if not isinstance(node_output, dict):
            continue
        for key in _OUTPUT_KEYS:
            files = node_output.get(key)
            if not isinstance(files, list):
                continue
            for file_info in files:
                if isinstance(file_info, dict) and file_info.get("filename"):
                    return file_info
    return None


class ComfyClient:
    """Минимальный клиент ComfyUI API (httpx), с инъекцией транспорта."""

    def __init__(
        self,
        config: Config,
        http: httpx.AsyncClient | None = None,
        client_id: str | None = None,
    ) -> None:
        self.config = config
        self.client_id = client_id or uuid.uuid4().hex
        self._owns_http = http is None
        self._http = (
            http
            if http is not None
            else httpx.AsyncClient(
                base_url=config.comfy_url,
                timeout=config.comfy_request_timeout_seconds,
            )
        )
        # Причина последнего отказа system_stats — для health (booting/unavailable).
        self.last_error: str | None = None

    # -- служебное ---------------------------------------------------------

    def _url(self, path: str) -> str:
        return f"{self.config.comfy_url.rstrip('/')}{path}"

    def _request_timeout(self, deadline: float) -> float:
        """Таймаут одного запроса, не выходящий за общий дедлайн."""

        remaining = deadline - time.monotonic()
        limit = max(0.001, min(self.config.comfy_request_timeout_seconds, remaining))
        return limit

    async def _sleep(self, interval: float) -> None:
        await asyncio.sleep(interval)

    @staticmethod
    def _next_interval(interval: float, maximum: float) -> float:
        return min(interval * 1.5, maximum)

    # -- lifecycle ---------------------------------------------------------

    async def close(self) -> None:
        """Закрыть собственный httpx-клиент (инъектированный — не наш)."""

        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> ComfyClient:
        return self

    async def __aexit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        await self.close()
        return False

    # -- API ComfyUI -------------------------------------------------------

    async def wait_ready(self, timeout: float | None = None) -> dict[str, Any]:
        """Дождаться ``GET /system_stats`` (старт ComfyUI) или ошибка boot_timeout."""

        limit = self.config.comfy_boot_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + limit
        interval = self.config.comfy_poll_initial_seconds
        last_error = "нет ответа"
        while True:
            try:
                response = await self._http.get(
                    self._url("/system_stats"),
                    timeout=self._request_timeout(deadline),
                )
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}"
                else:
                    try:
                        payload = response.json()
                    except ValueError:
                        last_error = "ответ /system_stats — не JSON"
                    else:
                        if isinstance(payload, dict):
                            self.last_error = None
                            return payload
                        last_error = "ответ /system_stats — не объект"
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            now = time.monotonic()
            if now >= deadline:
                raise WorkerError(
                    "comfy_boot_timeout",
                    f"ComfyUI не готов за {limit:g} с: {last_error}",
                )
            await self._sleep(min(interval, max(0.001, deadline - now)))
            interval = self._next_interval(interval, self.config.comfy_poll_max_seconds)

    async def submit(self, prompt: dict[str, Any]) -> str:
        """Отправить готовый API-граф в ``POST /prompt`` → ``prompt_id``."""

        payload = {"prompt": prompt, "client_id": self.client_id}
        try:
            response = await self._http.post(
                self._url("/prompt"),
                json=payload,
                timeout=self.config.comfy_request_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise WorkerError(
                "comfy_submit_failed",
                f"POST /prompt: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise WorkerError(
                "comfy_submit_failed",
                f"POST /prompt: HTTP {response.status_code}: {_describe_error_body(response)}",
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise WorkerError("comfy_submit_failed", "POST /prompt: ответ не JSON") from exc
        if not isinstance(body, dict):
            raise WorkerError("comfy_submit_failed", f"POST /prompt: неожиданный ответ {body!r}")
        if body.get("error") or body.get("node_errors"):
            raise WorkerError(
                "comfy_submit_failed",
                f"ComfyUI отклонил граф: {_describe_error_body(response)}",
            )
        prompt_id = body.get("prompt_id")
        if not isinstance(prompt_id, str) or not prompt_id:
            raise WorkerError("comfy_submit_failed", "в ответе POST /prompt нет prompt_id")
        return prompt_id

    async def wait(self, prompt_id: str, timeout: float | None = None) -> dict[str, Any]:
        """Дождаться записи ``GET /history/{prompt_id}`` или ошибка/таймаут."""

        limit = self.config.job_timeout_seconds if timeout is None else timeout
        deadline = time.monotonic() + limit
        interval = self.config.comfy_poll_initial_seconds
        path = f"/history/{prompt_id}"
        last_error = "запись ещё не готова"
        queue_checked_at = 0.0
        queue_state: str | None = None
        while True:
            now = time.monotonic()
            # Раз в QUEUE_POLL_SECONDS фиксируем состояние очереди: по логу видно,
            # что граф действительно попал в исполнение, а не ждёт в ComfyUI.
            if now - queue_checked_at >= QUEUE_POLL_SECONDS:
                queue_checked_at = now
                state = await self._queue_state(prompt_id)
                if state is not None and state != queue_state:
                    queue_state = state
                    _LOG.info("ComfyUI %s: %s", prompt_id, state)
            try:
                response = await self._http.get(
                    self._url(path),
                    timeout=self._request_timeout(deadline),
                )
            except httpx.HTTPError as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            else:
                if response.status_code >= 400:
                    last_error = f"HTTP {response.status_code}"
                else:
                    try:
                        body = response.json()
                    except ValueError:
                        last_error = "ответ /history — не JSON"
                    else:
                        entry = body.get(prompt_id) if isinstance(body, dict) else None
                        if isinstance(entry, dict):
                            status = entry.get("status")
                            if isinstance(status, dict) and status.get("status_str") == "error":
                                raise WorkerError(
                                    "comfy_execution_failed",
                                    f"ComfyUI: {_describe_status_error(status)}",
                                )
                            return entry
                        last_error = "запись ещё не готова"
            now = time.monotonic()
            if now >= deadline:
                raise WorkerError(
                    "comfy_timeout",
                    f"нет результата {path} за {limit:g} с: {last_error}",
                )
            await self._sleep(min(interval, max(0.001, deadline - now)))
            interval = self._next_interval(interval, self.config.comfy_poll_max_seconds)

    async def fetch_output(self, history_entry: dict[str, Any]) -> FileOutput:
        """Скачать первый файл картинки из history-entry через ``GET /view``."""

        file_info = _first_output_file(history_entry)
        if file_info is None:
            raise WorkerError(
                "comfy_output_missing",
                "в history.outputs нет файлов images/gifs (SaveImage не сработал?)",
            )
        filename = str(file_info.get("filename") or "")
        if not filename:
            raise WorkerError("comfy_output_missing", "в history.outputs файл без filename")
        subfolder = str(file_info.get("subfolder") or "")
        file_type = str(file_info.get("type") or "output")
        params = {"filename": filename, "subfolder": subfolder, "type": file_type}
        try:
            response = await self._http.get(
                self._url("/view"),
                params=params,
                timeout=self.config.comfy_request_timeout_seconds,
            )
        except httpx.HTTPError as exc:
            raise WorkerError(
                "comfy_output_missing",
                f"GET /view {filename!r}: {type(exc).__name__}: {exc}",
            ) from exc
        if response.status_code >= 400:
            raise WorkerError(
                "comfy_output_missing",
                f"GET /view {filename!r}: HTTP {response.status_code}: {response.text[:300]}",
            )
        data = response.content
        if not data:
            raise WorkerError("comfy_output_missing", f"ComfyUI вернул пустой файл {filename!r}")
        content_type = response.headers.get("content-type") or mimetypes.guess_type(filename)[0]
        return FileOutput(
            data=data,
            filename=filename,
            content_type=content_type or "application/octet-stream",
        )

    async def _queue_state(self, prompt_id: str) -> str | None:
        """Состояние графа в очереди ComfyUI: «выполняется» / «в очереди» / неизвестно.

        ``GET /queue`` отдаёт ``queue_running``/``queue_pending`` — списки записей
        вида ``[number, prompt_id, prompt, extra, outputs]``. Ошибки чтения не
        валят ожидание: наблюдаемость не должна менять результат job.
        """

        try:
            response = await self._http.get(
                self._url("/queue"),
                timeout=self.config.comfy_request_timeout_seconds,
            )
        except httpx.HTTPError:
            return None
        if response.status_code >= 400:
            return None
        try:
            body = response.json()
        except ValueError:
            return None
        if not isinstance(body, dict):
            return None

        def _has(items: Any) -> bool:
            return isinstance(items, list) and any(
                isinstance(item, list) and len(item) > 1 and item[1] == prompt_id for item in items
            )

        if _has(body.get("queue_running")):
            return "выполняется (граф снят с очереди и считается)"
        pending = body.get("queue_pending")
        if isinstance(pending, list):
            for position, item in enumerate(pending, start=1):
                if isinstance(item, list) and len(item) > 1 and item[1] == prompt_id:
                    return f"в очереди, позиция {position}"
        return None

    async def delete_from_queue(self, prompt_id: str) -> None:
        """Best-effort ``POST /queue {"delete": [id]}``: снять граф из очереди."""

        try:
            response = await self._http.post(
                self._url("/queue"),
                json={"delete": [prompt_id]},
                timeout=self.config.comfy_request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - best effort, но не молча
            _LOG.warning("ComfyUI /queue delete не выполнен: %s: %s", type(exc).__name__, exc)
            return
        if response.status_code >= 400:
            _LOG.warning(
                "ComfyUI /queue delete ответил HTTP %s: %s",
                response.status_code,
                response.text[:200],
            )

    async def interrupt(self) -> None:
        """Best-effort ``POST /interrupt``: отмена job гасит граф ComfyUI."""

        try:
            response = await self._http.post(
                self._url("/interrupt"),
                timeout=self.config.comfy_request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001 - best effort, но не молча
            _LOG.warning("ComfyUI /interrupt не выполнен: %s: %s", type(exc).__name__, exc)
            return
        if response.status_code >= 400:
            _LOG.warning(
                "ComfyUI /interrupt ответил HTTP %s: %s",
                response.status_code,
                response.text[:200],
            )

    async def system_stats(self, timeout: float | None = None) -> dict[str, Any] | None:
        """``GET /system_stats`` для health: короткий таймаут, без исключений."""

        limit = self.config.health_timeout_seconds if timeout is None else timeout
        try:
            response = await self._http.get(self._url("/system_stats"), timeout=limit)
        except httpx.ConnectTimeout as exc:
            self.last_error = "timeout"
            _LOG.debug("health: ComfyUI connect timeout: %s", exc)
            return None
        except (httpx.ConnectError, httpx.ReadTimeout, httpx.PoolTimeout) as exc:
            self.last_error = "connect" if isinstance(exc, httpx.ConnectError) else "timeout"
            _LOG.debug("health: ComfyUI недоступен: %s", exc)
            return None
        except httpx.HTTPError as exc:
            self.last_error = "error"
            _LOG.debug("health: ComfyUI ошибка: %s", exc)
            return None
        if response.status_code >= 400:
            self.last_error = f"http_{response.status_code}"
            return None
        try:
            payload = response.json()
        except ValueError:
            self.last_error = "invalid_json"
            return None
        if not isinstance(payload, dict):
            self.last_error = "invalid_payload"
            return None
        self.last_error = None
        return payload


__all__ = ["ComfyClient", "FileOutput"]
