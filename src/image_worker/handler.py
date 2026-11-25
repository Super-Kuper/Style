"""RunPod handler контракта `image-serverless/v1`.

Поток: валидация job/input (включая ``input.assets``) → ``action:"health"`` или
``wait_ready`` → доставка assets → ``submit`` → ``wait`` → ``fetch_output`` →
размеры картинки из байтов → телеметрия из фактов исполнения → доставка.
Отмена (``asyncio.CancelledError``) → ``POST /interrupt`` в ComfyUI → исключение
наружу (job FAILED). Внешние зависимости инъектируются
(``build_handler``/``Dependencies``): тесты идут без сети, GPU и runpod.

Ссылки: `contract/worker-contract.md` §1/§2/§4.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .assets import Asset, download_assets, validate_assets
from .comfy_client import ComfyClient
from .config import Config
from .errors import WorkerError
from .health import health_payload
from .imaging import ImageInfo, read_image_info
from .storage import OutputStorage, deliver_image
from .telemetry import build_telemetry, detect_gpu

_LOG = logging.getLogger(__name__)

# Инъекции для тестов: реальные функции — дефолты Dependencies/build_handler.
AssetsDownloader = Callable[[list[Asset], Config], Awaitable[list[Path]]]
ImageReader = Callable[[bytes], ImageInfo]
GpuProvider = Callable[[], tuple[float, str]]


@dataclass
class Dependencies:
    """Всё внешнее, что нужно handler'у (инъекция для тестов)."""

    config: Config
    comfy: ComfyClient
    storage: OutputStorage | None = None
    assets_downloader: AssetsDownloader = download_assets
    image_reader: ImageReader = read_image_info
    gpu_provider: GpuProvider = detect_gpu


def _job_id(job: Any) -> str:
    if isinstance(job, Mapping) and job.get("id") is not None:
        return str(job.get("id"))
    return "<без id>"


def _validate_job(job: Any) -> Mapping[str, Any]:
    """``job.input`` обязателен и должен быть объектом (`contract/input.schema.json`)."""

    if not isinstance(job, Mapping):
        raise WorkerError("input_invalid", "job должен быть объектом RunPod")
    payload = job.get("input")
    if not isinstance(payload, Mapping):
        raise WorkerError("input_invalid", "job.input отсутствует или не является объектом")
    return payload


def _optional_mapping(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = payload.get(name)
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise WorkerError("input_invalid", f"input.{name}: ожидался объект")
    return value


#: Классы сэмплера, у которых число шагов лежит в ``inputs.steps`` (KF-графы).
_SAMPLER_PREFIXES = ("KSampler", "SamplerCustom")
#: Имена моделей в загрузчиках: по ним в логе видно, что именно грузится.
_MODEL_KEYS = ("unet_name", "ckpt_name", "clip_name", "vae_name")


def _graph_summary(prompt: Mapping[str, Any]) -> str:
    """Компактная строка о графе для лога: ноды, шаги, размер, файлы моделей.

    Нужна, чтобы отличать «первый кадр медленный из-за холодной загрузки» от
    «граф ушёл на другие шаги/модель»: по ``prompt`` в логе видно, сколько
    шагов реально поедет в GPU и какие файлы будут читаться с тома.
    """

    nodes = [node for node in prompt.values() if isinstance(node, Mapping)]
    steps: set[int] = set()
    models: list[str] = []
    size = ""
    for node in nodes:
        inputs = node.get("inputs")
        if not isinstance(inputs, Mapping):
            continue
        if str(node.get("class_type", "")).startswith(_SAMPLER_PREFIXES):
            value = inputs.get("steps")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                steps.add(int(value))
        for key in _MODEL_KEYS:
            value = inputs.get(key)
            if isinstance(value, str) and value and value not in models:
                models.append(value)
        if not size:
            width, height = inputs.get("width"), inputs.get("height")
            if isinstance(width, int) and isinstance(height, int) and not isinstance(width, bool):
                size = f"{width}x{height}"
    return (
        f"нод {len(nodes)}, шагов {sorted(steps) or '?'}, размер {size or '?'}, "
        f"модели {models or ['?']}"
    )


async def execute(job: Any, deps: Dependencies) -> dict[str, Any]:
    """Выполнить один job; ошибки — ``WorkerError`` наружу (job FAILED)."""

    payload = _validate_job(job)
    action = payload.get("action")
    if action == "health":
        # Health не трогает assets и не грузит модель (`contract §3`).
        _LOG.info("job %s: health, ComfyUI/граф/assets не трогаем", _job_id(job))
        return await health_payload(deps.config, deps.comfy)
    if action is not None:
        raise WorkerError("input_invalid", f"input.action: неизвестное значение {action!r}")

    prompt = payload.get("prompt")
    if not isinstance(prompt, Mapping) or not prompt:
        raise WorkerError(
            "input_prompt_missing",
            "input.prompt отсутствует или пуст: без action='health' нужен готовый API-граф",
        )
    parameter = _optional_mapping(payload, "parameter")
    acceleration = _optional_mapping(payload, "acceleration")

    comfy = deps.comfy
    config = deps.config
    # Структура/заявленные размеры assets проверяются до ожидания ComfyUI:
    # ошибка контракта видна сразу, а не после холодного старта (`contract §3`).
    assets = validate_assets(payload.get("assets", []), config)

    # ComfyUI может стартовать дольше endpoint'а — ждём до COMFY_BOOT_TIMEOUT_SECONDS.
    await comfy.wait_ready(config.comfy_boot_timeout_seconds)
    # Assets — до submit: граф ссылается на имена файлов в ComfyUI/input.
    asset_paths = await deps.assets_downloader(assets, config)
    if asset_paths:
        _LOG.info("job %s: доставлено assets=%d", _job_id(job), len(asset_paths))

    started = time.monotonic()
    # Снимок того, что уйдёт в GPU: число шагов и файлы моделей видны до submit,
    # поэтому по логу понятно, почему первый кадр на холодном томе не успевает.
    _LOG.info(
        "job %s: граф %s, ожидание до %gs",
        _job_id(job),
        _graph_summary(prompt),
        config.job_timeout_seconds,
    )
    prompt_id = await comfy.submit(dict(prompt))
    _LOG.info("job %s: граф отправлен, prompt_id=%s", _job_id(job), prompt_id)

    try:
        history_entry = await comfy.wait(prompt_id, config.job_timeout_seconds)
    except asyncio.CancelledError:
        # Отмена job должна гасить текущий граф в ComfyUI, затем проброс.
        _LOG.warning("job %s: отмена, прерываю ComfyUI (prompt_id=%s)", _job_id(job), prompt_id)
        await comfy.interrupt()
        raise
    except WorkerError as exc:
        # Таймаут ожидания — тоже повод освободить GPU: иначе граф продолжает
        # считаться, а следующий job стоит за ним в очереди ComfyUI.
        if exc.code == "comfy_timeout":
            _LOG.warning(
                "job %s: таймаут ожидания, прерываю граф ComfyUI (prompt_id=%s)",
                _job_id(job),
                prompt_id,
            )
            await comfy.interrupt()
            await comfy.delete_from_queue(prompt_id)
        raise
    # Фактические секунды GPU: submit → history completed.
    elapsed_seconds = time.monotonic() - started

    output = await comfy.fetch_output(history_entry)
    # Размеры — факт готового файла, а не снимок parameter.width/height.
    image = deps.image_reader(output.data)
    gpu = deps.gpu_provider()
    telemetry = build_telemetry(
        prompt,
        parameter,
        acceleration,
        image=image,
        gpu=gpu,
        elapsed_seconds=elapsed_seconds,
    )
    delivery = await deliver_image(output.data, output.filename, config, deps.storage)

    result: dict[str, Any] = {**delivery, "telemetry": telemetry}
    _LOG.info(
        "job %s: готово за %gs, %s байт, %sx%s, доставка=%s",
        _job_id(job),
        elapsed_seconds,
        len(output.data),
        image.width,
        image.height,
        "image_url" if "image_url" in delivery else "inline base64",
    )
    return result


def build_handler(
    config: Config,
    comfy: ComfyClient,
    storage: OutputStorage | None = None,
    *,
    assets_downloader: AssetsDownloader = download_assets,
    image_reader: ImageReader = read_image_info,
    gpu_provider: GpuProvider = detect_gpu,
) -> "Any":
    """Собрать handler с зафиксированными зависимостями (для тестов и main)."""

    deps = Dependencies(
        config=config,
        comfy=comfy,
        storage=storage,
        assets_downloader=assets_downloader,
        image_reader=image_reader,
        gpu_provider=gpu_provider,
    )

    async def handler(job: Any) -> dict[str, Any]:
        return await execute(job, deps)

    return handler


# Ленивый дефолт для RunPod: config/comfy создаются при первом job, чтобы
# импорт модуля (и тесты) не требовали живого ComfyUI.
_default_deps: Dependencies | None = None


def default_dependencies() -> Dependencies:
    """Зависимости по умолчанию из env (`Config.from_env`, локальный ComfyUI)."""

    global _default_deps
    if _default_deps is None:
        config = Config.from_env()
        _default_deps = Dependencies(config=config, comfy=ComfyClient(config))
    return _default_deps


async def handler(job: Any) -> dict[str, Any]:
    """Точка входа RunPod (`main.py`); тесты подменяют ``_default_deps``."""

    return await execute(job, default_dependencies())


__all__ = [
    "Dependencies",
    "build_handler",
    "default_dependencies",
    "execute",
    "handler",
]
