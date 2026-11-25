"""Env-конфигурация: дефолты, алиасы, видимая ошибка на невалидном значении."""

from __future__ import annotations

import pytest

from image_worker.config import Config
from image_worker.errors import WorkerError


def test_defaults() -> None:
    config = Config()
    assert config.comfy_url == "http://127.0.0.1:8188"
    assert config.job_timeout_seconds == 900
    assert config.comfy_boot_timeout_seconds == 600
    assert config.comfyui_input_dir == "/workspace/runpod-slim/ComfyUI/input"
    assert config.models_root == "/runpod-volume/models"
    assert config.assets_max_bytes == 64 * 1024 * 1024


def test_env_overrides() -> None:
    config = Config.from_env(
        {
            "COMFY_URL": "http://127.0.0.1:9999",
            "JOB_TIMEOUT_SECONDS": "600",
            "COMFY_BOOT_TIMEOUT_SECONDS": "120",
            "VOLUME_ROOT": "/volume",
            "COMFYUI_DIR": "/comfy",
            "OUTPUT_INLINE_MAX_MB": "2",
            "OUTPUT_INLINE_BUDGET_MB": "3",
            "ASSETS_MAX_MB": "8",
            "ASSETS_MAX_TOTAL_MB": "16",
        }
    )
    assert config.comfy_url == "http://127.0.0.1:9999"
    assert config.job_timeout_seconds == 600
    assert config.comfyui_input_dir == "/comfy/input"
    assert config.models_root == "/volume/models"
    assert config.inline_max_bytes == 2 * 1024 * 1024
    assert config.inline_budget_bytes == 3 * 1024 * 1024
    assert config.assets_max_bytes == 8 * 1024 * 1024
    assert config.assets_max_total_bytes == 16 * 1024 * 1024


def test_job_timeout_alias() -> None:
    assert Config.from_env({"COMFY_JOB_TIMEOUT_SECONDS": "555"}).job_timeout_seconds == 555


def test_s3_alias_and_secure_default() -> None:
    config = Config.from_env(
        {
            "S3_ENDPOINT": "http://minio:9000",
            "S3_ACCESS_KEY": "key",
            "S3_SECRET_KEY": "secret",
            "OUTPUT_BUCKET": "bucket",
        }
    )
    assert config.s3_bucket == "bucket"
    assert config.s3_configured is True
    assert config.s3_secure is False

    secure = Config.from_env({"S3_ENDPOINT": "https://s3.example.com"})
    assert secure.s3_secure is True


@pytest.mark.parametrize(
    "env",
    [
        {"COMFY_BOOT_TIMEOUT_SECONDS": "soon"},
        {"JOB_TIMEOUT_SECONDS": "0"},
        {"ASSETS_MAX_MB": "-1"},
        {"OUTPUT_INLINE_MAX_MB": "nan"},
        {"LINK_VOLUME_NODES": "maybe"},
        {"IMAGE_GPU_CC": "8.6"},
        {"IMAGE_GPU_CLASS": "RTX A6000"},
        {"IMAGE_GPU_CC": "abc", "IMAGE_GPU_CLASS": "RTX A6000"},
    ],
)
def test_invalid_env_is_visible(env: dict[str, str]) -> None:
    with pytest.raises(WorkerError) as error:
        Config.from_env(env)
    assert error.value.code == "config_invalid"


def test_gpu_override_pair() -> None:
    config = Config.from_env({"IMAGE_GPU_CC": "8.6", "IMAGE_GPU_CLASS": "RTX A6000"})
    assert config.image_gpu_cc == 8.6
    assert config.image_gpu_class == "RTX A6000"
