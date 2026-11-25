"""Доставка результата: base64 для малых файлов, presigned URL для больших.

Картинка обычно < 10 МБ, поэтому штатно отдаётся base64 (`image`). Больший файл
(4K PNG с шумом, маска и т.п.) уходит в S3/MinIO с presigned URL (`image_url`,
`OUTPUT_URL_TTL_SECONDS`). Нет S3 при превышении лимита — видимая ошибка
``output_storage_missing``, а не «успех без файла».

boto3 импортируется лениво (extra `[s3]`), блокирующие вызовы — в потоке.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from .config import Config
from .errors import WorkerError

_LOG = logging.getLogger(__name__)


class OutputStorage(Protocol):
    """Интерфейс выгрузки больших файлов (S3, MinIO или тестовая заглушка)."""

    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """Загрузить байты по ключу; вернуть фактический ключ."""

    async def presigned_url(self, key: str, ttl_seconds: int) -> str:
        """Вернуть presigned HTTP(S)-URL для скачивания."""


class S3Storage:
    """S3/MinIO через boto3: put_object + generate_presigned_url.

    Клиент создаётся лениво в рабочем потоке: import boto3 и создание клиента
    блокирующие, а event loop handler'а занят ожиданием ComfyUI.
    """

    def __init__(self, config: Config) -> None:
        if not config.s3_configured:
            raise WorkerError(
                "output_storage_missing",
                "S3 не настроен: нужны S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET",
            )
        self._config = config
        self._client: Any = None

    def _make_client(self) -> Any:
        try:
            import boto3  # noqa: PLC0415 - ленивый импорт extra [s3]
        except ImportError as exc:
            raise WorkerError(
                "output_storage_failed",
                "boto3 не установлен: нужен extra [s3]",
            ) from exc
        return boto3.client(
            "s3",
            endpoint_url=self._config.s3_endpoint,
            aws_access_key_id=self._config.s3_access_key,
            aws_secret_access_key=self._config.s3_secret_key,
            region_name=self._config.s3_region,
            use_ssl=self._config.s3_secure,
        )

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = self._make_client()
        return self._client

    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        """``put_object`` в потоке; ошибки → ``output_storage_failed``."""

        def _put() -> str:
            client = self._get_client()
            client.put_object(
                Bucket=self._config.s3_bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            )
            return key

        try:
            return await asyncio.to_thread(_put)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001 - любые ошибки boto3 важны
            raise WorkerError(
                "output_storage_failed",
                f"S3 put_object {key!r}: {type(exc).__name__}: {exc}",
            ) from exc

    async def presigned_url(self, key: str, ttl_seconds: int) -> str:
        """``generate_presigned_url`` в потоке; ошибки → ``output_storage_failed``."""

        def _sign() -> str:
            client = self._get_client()
            return str(
                client.generate_presigned_url(
                    "get_object",
                    Params={"Bucket": self._config.s3_bucket, "Key": key},
                    ExpiresIn=ttl_seconds,
                )
            )

        try:
            url = await asyncio.to_thread(_sign)
        except WorkerError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise WorkerError(
                "output_storage_failed",
                f"S3 presigned URL {key!r}: {type(exc).__name__}: {exc}",
            ) from exc
        if not url:
            raise WorkerError("output_storage_failed", f"S3 вернул пустой presigned URL для {key!r}")
        return url


def _object_key(filename: str) -> str:
    """Ключ объекта: ``image-serverless/outputs/<uuid>/<basename>`` (без путей)."""

    name = Path(filename or "image.png").name or "image.png"
    return f"image-serverless/outputs/{uuid4().hex}/{name}"


def _content_type(filename: str) -> str:
    guessed = mimetypes.guess_type(filename or "")[0] if filename else None
    return guessed or "application/octet-stream"


async def deliver_image(
    data: bytes,
    filename: str,
    config: Config,
    storage: OutputStorage | None = None,
) -> dict[str, str]:
    """Вернуть ``{"image": base64}`` или ``{"image_url": presigned}``.

    Ровно одно поле — как требует `contract/output.schema.json`. Пустой файл
    успехом не считается: ``output_empty``.
    """

    if not data:
        raise WorkerError("output_empty", "ComfyUI вернул пустой файл: картинка не сформирована")

    size = len(data)
    base64_size = 4 * ((size + 2) // 3)
    if size <= config.inline_max_bytes and base64_size <= config.inline_budget_bytes:
        _LOG.info("deliver: base64 inline, %s байт (base64 ~%s)", size, base64_size)
        return {"image": base64.b64encode(data).decode("ascii")}

    if storage is None:
        if not config.s3_configured:
            raise WorkerError(
                "output_storage_missing",
                f"файл {filename!r} ({size} байт) превышает inline-лимит "
                f"({config.inline_max_bytes} байт / бюджет {config.inline_budget_bytes} байт), "
                "а S3 не настроен (S3_ENDPOINT/S3_ACCESS_KEY/S3_SECRET_KEY/S3_BUCKET)",
            )
        storage = S3Storage(config)

    key = _object_key(filename)
    content_type = _content_type(filename)
    try:
        stored_key = await storage.upload(data, key, content_type) or key
        url = await storage.presigned_url(stored_key, config.output_url_ttl_seconds)
    except WorkerError:
        raise
    except Exception as exc:  # noqa: BLE001 - внешнее хранилище
        raise WorkerError(
            "output_storage_failed",
            f"не удалось выгрузить {filename!r} в S3: {type(exc).__name__}: {exc}",
        ) from exc
    if not isinstance(url, str) or not url:
        raise WorkerError("output_storage_failed", "хранилище вернул пустой presigned URL")
    _LOG.info("deliver: S3 presigned URL, %s байт, ttl=%s с", size, config.output_url_ttl_seconds)
    return {"image_url": url}


__all__ = ["OutputStorage", "S3Storage", "deliver_image"]
