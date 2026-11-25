"""Общие фикстуры тестов: конфиг без сети/GPU и клиент ComfyUI на MockTransport."""

from __future__ import annotations

import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from image_worker.comfy_client import ComfyClient  # noqa: E402
from image_worker.config import Config  # noqa: E402


@pytest.fixture
def config(tmp_path: Path) -> Config:
    """Быстрый конфиг: таймауты в миллисекундах, каталоги во tmp_path."""

    return Config.from_env(
        {
            "COMFYUI_INPUT_DIR": str(tmp_path / "comfyui" / "input"),
            "VOLUME_ROOT": str(tmp_path / "volume"),
            "MODELS_ROOT": str(tmp_path / "volume" / "models"),
            "COMFY_BOOT_TIMEOUT_SECONDS": "2",
            "JOB_TIMEOUT_SECONDS": "2",
            "COMFY_POLL_INITIAL_SECONDS": "0.001",
            "COMFY_POLL_MAX_SECONDS": "0.002",
            "COMFY_REQUEST_TIMEOUT_SECONDS": "2",
            "HEALTH_TIMEOUT_SECONDS": "1",
            "ASSETS_TIMEOUT_SECONDS": "2",
            "ASSETS_MAX_MB": "1",
            "ASSETS_MAX_TOTAL_MB": "2",
            "OUTPUT_INLINE_MAX_MB": "1",
            "OUTPUT_INLINE_BUDGET_MB": "2",
        }
    )


@pytest.fixture
def make_comfy(config: Config):
    """``make_comfy(mock)`` → реальный ComfyClient на httpx.MockTransport."""

    def _make(mock) -> ComfyClient:
        http = httpx.AsyncClient(transport=mock.transport(), base_url=config.comfy_url)
        return ComfyClient(config, http=http)

    return _make
