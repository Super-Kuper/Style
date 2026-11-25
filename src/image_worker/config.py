"""Env-конфигурация воркера (`contract/worker-contract.md` §5).

Все значения читаются из переменных окружения endpoint'а RunPod; секреты
(`S3_ACCESS_KEY`, `S3_SECRET_KEY`) не логируются и в образ не попадают.
Невалидное значение — видимая ошибка `config_invalid` с именем переменной,
а не тихий дефолт.
"""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .errors import WorkerError

DEFAULT_COMFY_URL = "http://127.0.0.1:8188"
DEFAULT_VOLUME_ROOT = "/runpod-volume"
# Корень ComfyUI в образе RunPod; входные файлы `input.assets` кладутся в
# `$COMFYUI_DIR/input`.
DEFAULT_COMFYUI_DIR = "/workspace/runpod-slim/ComfyUI"


# ---------------------------------------------------------------------------
# Разбор env-значений: любая ошибка разбора — WorkerError("config_invalid").
# ---------------------------------------------------------------------------


def _clean(value: Any) -> str | None:
    """Вернуть непустую строку значения или ``None`` (пустая строка = не задано)."""

    if value is None:
        return None
    if isinstance(value, str):
        text = value.strip()
        return text or None
    text = str(value).strip()
    return text or None


def _lookup(env: Mapping[str, str], names: tuple[str, ...]) -> tuple[str | None, str | None]:
    """Найти первое заданное имя из `names` → ``(имя, значение)``."""

    for name in names:
        raw = _clean(env.get(name))
        if raw is not None:
            return name, raw
    return None, None


def _invalid(name: str, detail: str) -> WorkerError:
    return WorkerError("config_invalid", f"{name}: {detail}")


def _env_float(
    env: Mapping[str, str],
    names: tuple[str, ...],
    default: float,
    *,
    positive: bool = False,
) -> float:
    name, raw = _lookup(env, names)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise _invalid(name or names[0], f"ожидалось число, получено {raw!r}") from exc
    if not math.isfinite(value):
        raise _invalid(name or names[0], f"ожидалось конечное число, получено {raw!r}")
    if positive and value <= 0:
        raise _invalid(name or names[0], f"должно быть > 0, получено {raw!r}")
    return value


def _env_int(
    env: Mapping[str, str],
    names: tuple[str, ...],
    default: int,
    *,
    positive: bool = False,
) -> int:
    name, raw = _lookup(env, names)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise _invalid(name or names[0], f"ожидалось целое число, получено {raw!r}") from exc
    if positive and value <= 0:
        raise _invalid(name or names[0], f"должно быть > 0, получено {raw!r}")
    return value


def _env_bool(env: Mapping[str, str], names: tuple[str, ...], default: bool) -> bool:
    name, raw = _lookup(env, names)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise _invalid(name or names[0], f"ожидалось true/false, получено {raw!r}")


def _env_str(env: Mapping[str, str], names: tuple[str, ...], default: str) -> str:
    _name, raw = _lookup(env, names)
    return default if raw is None else raw


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Config:
    """Параметры воркера; `Config.from_env()` — единственный штатный способ.

    Значения по умолчанию рассчитаны на image-эндпоинт с одной картой 48 ГБ:
    короткий граф (4–30 шагов), таймаут 900 с, картинка обычно < 10 МБ.
    """

    comfy_url: str = DEFAULT_COMFY_URL
    comfy_boot_timeout_seconds: float = 600.0
    job_timeout_seconds: float = 900.0
    comfy_poll_initial_seconds: float = 0.5
    comfy_poll_max_seconds: float = 3.0
    comfy_request_timeout_seconds: float = 60.0
    output_inline_max_mb: float = 10.0
    output_inline_budget_mb: float = 15.0
    output_url_ttl_seconds: int = 86400
    s3_endpoint: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_secure: bool = True
    volume_root: str = DEFAULT_VOLUME_ROOT
    link_volume_nodes: bool = False
    health_timeout_seconds: float = 10.0
    # Приём `input.assets` в ComfyUI/input: base image и маска edit-графа.
    comfyui_dir: str = DEFAULT_COMFYUI_DIR
    # None → `${COMFYUI_DIR}/input` подставляется в __post_init__ (env не задан).
    comfyui_input_dir: str | None = None
    assets_timeout_seconds: float = 120.0
    assets_max_mb: float = 64.0
    assets_max_total_mb: float = 128.0
    # None → `${VOLUME_ROOT}/models` подставляется в __post_init__.
    models_root: str | None = None
    # Фактические GPU-поля для окружений без torch (локальные тесты):
    # override нужен только чтобы заполнить gpu_cc/gpu_class без torch.
    image_gpu_cc: float | None = None
    image_gpu_class: str | None = None

    def __post_init__(self) -> None:
        """Дозаполнить вычисляемые каталоги (без env — от других полей)."""

        if not self.comfyui_input_dir:
            self.comfyui_input_dir = f"{self.comfyui_dir.rstrip('/')}/input"
        if not self.models_root:
            self.models_root = f"{self.volume_root.rstrip('/')}/models"

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Config:
        """Собрать конфигурацию из окружения (по умолчанию ``os.environ``)."""

        source: Mapping[str, str] = os.environ if env is None else env

        comfy_url = _env_str(source, ("COMFY_URL",), DEFAULT_COMFY_URL)
        comfy_boot_timeout = _env_float(source, ("COMFY_BOOT_TIMEOUT_SECONDS",), 600.0, positive=True)
        # Основной JOB_TIMEOUT_SECONDS; COMFY_JOB_TIMEOUT_SECONDS — алиас.
        job_timeout = _env_float(source, ("JOB_TIMEOUT_SECONDS", "COMFY_JOB_TIMEOUT_SECONDS"), 900.0, positive=True)
        poll_initial = _env_float(source, ("COMFY_POLL_INITIAL_SECONDS",), 0.5, positive=True)
        poll_max = _env_float(source, ("COMFY_POLL_MAX_SECONDS",), 3.0, positive=True)
        request_timeout = _env_float(source, ("COMFY_REQUEST_TIMEOUT_SECONDS",), 60.0, positive=True)
        inline_max_mb = _env_float(source, ("OUTPUT_INLINE_MAX_MB",), 10.0, positive=True)
        inline_budget_mb = _env_float(source, ("OUTPUT_INLINE_BUDGET_MB",), 15.0, positive=True)
        ttl = _env_int(source, ("OUTPUT_URL_TTL_SECONDS",), 86400, positive=True)
        health_timeout = _env_float(source, ("HEALTH_TIMEOUT_SECONDS",), 10.0, positive=True)

        s3_endpoint = _lookup(source, ("S3_ENDPOINT",))[1]
        s3_access_key = _lookup(source, ("S3_ACCESS_KEY",))[1]
        s3_secret_key = _lookup(source, ("S3_SECRET_KEY",))[1]
        s3_bucket = _lookup(source, ("S3_BUCKET", "OUTPUT_BUCKET"))[1]
        s3_region = _env_str(source, ("S3_REGION",), "us-east-1")
        # Дефолт S3_SECURE: true при https-endpoint (RunPod S3), иначе false
        # (локальный MinIO в docker-сети); явное значение env важнее.
        default_secure = not s3_endpoint or s3_endpoint.lower().startswith("https://")
        s3_secure = _env_bool(source, ("S3_SECURE",), default_secure)

        gpu_cc, gpu_class = _parse_gpu_override(source)
        volume_root = _env_str(source, ("VOLUME_ROOT",), DEFAULT_VOLUME_ROOT)
        comfyui_dir = _env_str(source, ("COMFYUI_DIR",), DEFAULT_COMFYUI_DIR)
        # Явный COMFYUI_INPUT_DIR важнее вычисленного; пустое значение = дефолт.
        comfyui_input_dir = _lookup(source, ("COMFYUI_INPUT_DIR",))[1]
        models_root = _lookup(source, ("MODELS_ROOT",))[1]
        assets_timeout = _env_float(source, ("ASSETS_TIMEOUT_SECONDS",), 120.0, positive=True)
        assets_max_mb = _env_float(source, ("ASSETS_MAX_MB",), 64.0, positive=True)
        assets_max_total_mb = _env_float(source, ("ASSETS_MAX_TOTAL_MB",), 128.0, positive=True)

        return cls(
            comfy_url=comfy_url,
            comfy_boot_timeout_seconds=comfy_boot_timeout,
            job_timeout_seconds=job_timeout,
            comfy_poll_initial_seconds=poll_initial,
            comfy_poll_max_seconds=poll_max,
            comfy_request_timeout_seconds=request_timeout,
            output_inline_max_mb=inline_max_mb,
            output_inline_budget_mb=inline_budget_mb,
            output_url_ttl_seconds=ttl,
            s3_endpoint=s3_endpoint,
            s3_access_key=s3_access_key,
            s3_secret_key=s3_secret_key,
            s3_bucket=s3_bucket,
            s3_region=s3_region,
            s3_secure=s3_secure,
            volume_root=volume_root,
            link_volume_nodes=_env_bool(source, ("LINK_VOLUME_NODES",), False),
            health_timeout_seconds=health_timeout,
            comfyui_dir=comfyui_dir,
            comfyui_input_dir=comfyui_input_dir,
            assets_timeout_seconds=assets_timeout,
            assets_max_mb=assets_max_mb,
            assets_max_total_mb=assets_max_total_mb,
            models_root=models_root,
            image_gpu_cc=gpu_cc,
            image_gpu_class=gpu_class,
        )

    @property
    def inline_max_bytes(self) -> int:
        """Порог «малого» файла для base64 (OUTPUT_INLINE_MAX_MB)."""

        return int(self.output_inline_max_mb * 1024 * 1024)

    @property
    def inline_budget_bytes(self) -> int:
        """Бюджет base64 в ответе RunPod (OUTPUT_INLINE_BUDGET_MB)."""

        return int(self.output_inline_budget_mb * 1024 * 1024)

    @property
    def assets_max_bytes(self) -> int:
        """Лимит одного файла `input.assets` в байтах (ASSETS_MAX_MB)."""

        return int(self.assets_max_mb * 1024 * 1024)

    @property
    def assets_max_total_bytes(self) -> int:
        """Суммарный лимит всех `input.assets` в байтах (ASSETS_MAX_TOTAL_MB)."""

        return int(self.assets_max_total_mb * 1024 * 1024)

    @property
    def s3_configured(self) -> bool:
        """Хватает ли env для выгрузки больших файлов в S3/MinIO."""

        return bool(self.s3_endpoint and self.s3_access_key and self.s3_secret_key and self.s3_bucket)


def _parse_gpu_override(env: Mapping[str, str]) -> tuple[float | None, str | None]:
    """IMAGE_GPU_CC/IMAGE_GPU_CLASS — только парой; иначе видимая config_invalid."""

    cc_raw = _clean(env.get("IMAGE_GPU_CC"))
    class_raw = _clean(env.get("IMAGE_GPU_CLASS"))
    if cc_raw is None and class_raw is None:
        return None, None
    if cc_raw is None or class_raw is None:
        raise _invalid(
            "IMAGE_GPU_CC/IMAGE_GPU_CLASS",
            "должны задаваться вместе (override для окружений без torch)",
        )
    try:
        cc = float(cc_raw)
    except ValueError as exc:
        raise _invalid("IMAGE_GPU_CC", f"ожидалось число, получено {cc_raw!r}") from exc
    if not math.isfinite(cc) or cc <= 0:
        raise _invalid("IMAGE_GPU_CC", f"ожидалось положительное число, получено {cc_raw!r}")
    return cc, class_raw


__all__ = ["Config", "DEFAULT_COMFY_URL", "DEFAULT_COMFYUI_DIR", "DEFAULT_VOLUME_ROOT"]
