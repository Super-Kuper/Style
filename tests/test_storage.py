"""Доставка результата: base64 inline, presigned image_url, видимые ошибки."""

from __future__ import annotations

import asyncio
import base64

import pytest

from image_worker.config import Config
from image_worker.errors import WorkerError
from image_worker.storage import deliver_image

DATA = b"\x89PNG\r\n\x1a\n" + b"x" * 100


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, str, str]] = []
        self.signs: list[tuple[str, int]] = []

    async def upload(self, data: bytes, key: str, content_type: str) -> str:
        self.uploads.append((data, key, content_type))
        return key

    async def presigned_url(self, key: str, ttl_seconds: int) -> str:
        self.signs.append((key, ttl_seconds))
        return f"https://s3.example/{key}?sig=example"


def _run(coro):
    return asyncio.run(coro)


def test_inline_base64(config) -> None:
    result = _run(deliver_image(DATA, "keyframe_00001_.png", config))
    assert set(result) == {"image"}
    assert base64.b64decode(result["image"]) == DATA


def test_empty_is_output_empty(config) -> None:
    with pytest.raises(WorkerError) as error:
        _run(deliver_image(b"", "keyframe_00001_.png", config))
    assert error.value.code == "output_empty"


def test_large_without_s3_is_output_storage_missing(tmp_path) -> None:
    config = Config.from_env(
        {
            "OUTPUT_INLINE_MAX_MB": "0.0001",
            "OUTPUT_INLINE_BUDGET_MB": "0.0001",
            "VOLUME_ROOT": str(tmp_path / "volume"),
        }
    )
    with pytest.raises(WorkerError) as error:
        _run(deliver_image(DATA, "big.png", config))
    assert error.value.code == "output_storage_missing"


def test_large_with_storage_returns_url(tmp_path) -> None:
    config = Config.from_env(
        {
            "OUTPUT_INLINE_MAX_MB": "0.0001",
            "OUTPUT_INLINE_BUDGET_MB": "0.0001",
            "VOLUME_ROOT": str(tmp_path / "volume"),
        }
    )
    storage = FakeStorage()
    result = _run(deliver_image(DATA, "big.png", config, storage))
    assert set(result) == {"image_url"}
    assert result["image_url"].startswith("https://s3.example/image-serverless/outputs/")
    assert result["image_url"].endswith("/big.png?sig=example")
    data, key, content_type = storage.uploads[0]
    assert data == DATA
    assert key.endswith("/big.png")
    assert content_type == "image/png"


def test_storage_failure_is_visible(tmp_path) -> None:
    class BrokenStorage(FakeStorage):
        async def upload(self, data: bytes, key: str, content_type: str) -> str:
            raise RuntimeError("s3 down")

    config = Config.from_env(
        {
            "OUTPUT_INLINE_MAX_MB": "0.0001",
            "OUTPUT_INLINE_BUDGET_MB": "0.0001",
            "VOLUME_ROOT": str(tmp_path / "volume"),
        }
    )
    with pytest.raises(WorkerError) as error:
        _run(deliver_image(DATA, "big.png", config, BrokenStorage()))
    assert error.value.code == "output_storage_failed"


def test_object_path_cannot_escape() -> None:
    from image_worker.storage import _object_key

    assert _object_key("../../etc/passwd").endswith("/passwd")
    assert "image-serverless/outputs/" in _object_key("x.png")
