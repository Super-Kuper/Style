"""input.assets: валидация, скачивание, sha256/размер, атомарная замена."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

import httpx
import pytest

from image_worker.assets import Asset, download_assets, validate_assets
from image_worker.errors import WorkerError

PAYLOAD = b"base-image-bytes"
SHA = hashlib.sha256(PAYLOAD).hexdigest()


def _asset(**overrides: object) -> dict:
    entry = {
        "name": "base.png",
        "url": "https://files.example/base.png",
        "sha256": SHA,
        "kind": "image",
        "size": len(PAYLOAD),
    }
    entry.update(overrides)
    return entry


def _run(coro):
    return asyncio.run(coro)


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler), follow_redirects=True)


# --- Валидация ---------------------------------------------------------------


def test_validate_ok(config) -> None:
    assets = validate_assets([_asset(), _asset(name="mask.png", kind="mask")], config)
    assert [asset.name for asset in assets] == ["base.png", "mask.png"]
    assert assets[0].sha256 == SHA
    assert assets[1].kind == "mask"


def test_validate_accepts_parsed_assets(config) -> None:
    parsed = [Asset(name="base.png", url="https://files.example/base.png")]
    assert validate_assets(parsed, config) == parsed


def test_service_asset_names_are_accepted(config) -> None:
    # Имена выбирает сервис: pipeline шлёт basename asset_<sha16>.png.
    entries = [
        {"name": "asset_0123456789abcdef.png", "url": "https://files.example/a.png", "kind": "image"},
        {"name": "asset_fedcba9876543210.png", "url": "https://files.example/b.png", "kind": "mask"},
    ]
    assert [asset.name for asset in validate_assets(entries, config)] == [
        "asset_0123456789abcdef.png",
        "asset_fedcba9876543210.png",
    ]


def test_empty_assets(config) -> None:
    assert validate_assets([], config) == []


@pytest.mark.parametrize("bad", [
    "not-a-list",
    {"name": "base.png", "url": "https://files.example/base.png"},
    [{"name": "a/b.png", "url": "https://files.example/a.png"}],
    [{"name": ".hidden.png", "url": "https://files.example/a.png"}],
    [{"name": "base.png", "url": "file:///etc/passwd"}],
    [{"name": "base.png", "url": "https://files.example/base.png", "sha256": "zz"}],
    [{"name": "base.png", "url": "https://files.example/base.png", "kind": "video"}],
    [{"name": "base.png", "url": "https://files.example/base.png", "size": 0}],
    [{"name": "base.png", "url": "https://files.example/base.png", "unknown": 1}],
])
def test_validate_invalid(config, bad) -> None:
    with pytest.raises(WorkerError) as error:
        validate_assets(bad, config)
    assert error.value.code == "asset_invalid"


def test_duplicate_names_rejected(config) -> None:
    with pytest.raises(WorkerError) as error:
        validate_assets([_asset(), _asset(url="https://files.example/other.png")], config)
    assert error.value.code == "asset_invalid"
    assert "встречается" in error.value.message


def test_too_many_assets(config) -> None:
    many = [_asset(name=f"f{i}.png") for i in range(9)]
    with pytest.raises(WorkerError) as error:
        validate_assets(many, config)
    assert error.value.code == "asset_invalid"


def test_declared_size_over_limit(config) -> None:
    with pytest.raises(WorkerError) as error:
        validate_assets([_asset(size=2 * 1024 * 1024)], config)
    assert error.value.code == "asset_too_large"


def test_declared_total_over_limit(config) -> None:
    entries = [_asset(name=f"f{i}.png", size=900_000) for i in range(3)]
    with pytest.raises(WorkerError) as error:
        validate_assets(entries, config)
    assert error.value.code == "asset_too_large"


# --- Скачивание --------------------------------------------------------------


def _download_handler(payload: bytes = PAYLOAD, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, content=payload)

    return handler


def test_download_writes_file_and_verifies(config) -> None:
    client = _client(_download_handler())
    try:
        paths = _run(download_assets([_asset()], config, http=client))
    finally:
        _run(client.aclose())

    target = Path(config.comfyui_input_dir) / "base.png"
    assert paths == [target]
    assert target.read_bytes() == PAYLOAD
    assert not list(target.parent.glob(".*.part"))


def test_download_overwrites_existing_atomically(config) -> None:
    input_dir = Path(config.comfyui_input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)
    target = input_dir / "base.png"
    target.write_bytes(b"old content")

    client = _client(_download_handler())
    try:
        _run(download_assets([_asset()], config, http=client))
    finally:
        _run(client.aclose())
    assert target.read_bytes() == PAYLOAD
    assert not list(input_dir.glob(".*.part"))


def test_hash_mismatch_is_visible(config) -> None:
    client = _client(_download_handler(payload=b"X" * len(PAYLOAD)))
    try:
        with pytest.raises(WorkerError) as error:
            _run(download_assets([_asset()], config, http=client))
    finally:
        _run(client.aclose())
    assert error.value.code == "asset_hash_mismatch"
    assert not (Path(config.comfyui_input_dir) / "base.png").exists()


def test_declared_size_mismatch_is_visible(config) -> None:
    client = _client(_download_handler())
    try:
        with pytest.raises(WorkerError) as error:
            _run(download_assets([_asset(size=len(PAYLOAD) + 1)], config, http=client))
    finally:
        _run(client.aclose())
    assert error.value.code == "asset_download_failed"
    assert "не совпал" in error.value.message


def test_http_error_is_visible(config) -> None:
    client = _client(_download_handler(status=500))
    try:
        with pytest.raises(WorkerError) as error:
            _run(download_assets([_asset()], config, http=client))
    finally:
        _run(client.aclose())
    assert error.value.code == "asset_download_failed"
    assert "HTTP 500" in error.value.message


def test_empty_download_is_visible(config) -> None:
    client = _client(_download_handler(payload=b""))
    try:
        with pytest.raises(WorkerError) as error:
            _run(download_assets([_asset(size=None)], config, http=client))
    finally:
        _run(client.aclose())
    assert error.value.code == "asset_download_failed"


def test_streamed_size_over_limit_is_visible(config) -> None:
    # Заявленный размер не указан, лимит превышается фактически при стриме.
    big = b"x" * (config.assets_max_bytes + 1)
    client = _client(_download_handler(payload=big))
    try:
        with pytest.raises(WorkerError) as error:
            _run(download_assets([_asset(sha256=None, size=None)], config, http=client))
    finally:
        _run(client.aclose())
    assert error.value.code == "asset_too_large"


def test_download_creates_input_dir(config) -> None:
    input_dir = Path(config.comfyui_input_dir)
    assert not input_dir.exists()
    client = _client(_download_handler())
    try:
        _run(download_assets([_asset()], config, http=client))
    finally:
        _run(client.aclose())
    assert input_dir.is_dir()
