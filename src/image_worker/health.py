"""Health-ответ воркера (`contract/worker-contract.md` §3).

Проверка не грузит модель и не запускает граф: только наличие тома
`VOLUME_ROOT` и короткий ``GET /system_stats`` к ComfyUI.

Статусы (документированы в контракте):
- ``comfyui=ready`` — ComfyUI ответил на /system_stats;
- ``comfyui=booting`` — ComfyUI ещё не принимает соединения/не ответил
  (старт или загрузка модели), это нормальное transient-состояние;
- ``comfyui=unavailable`` — ComfyUI ответил ошибкой/невалидным JSON либо
  клиент не сконфигурирован.
- ``status=ok`` — ready и том смонтирован; ``starting`` — booting и том есть;
  ``degraded`` — всё остальное (нет тома и/или ComfyUI недоступен).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from .config import Config

_LOG = logging.getLogger(__name__)


async def _system_stats(comfy: Any, timeout: float) -> dict[str, Any] | None:
    """Короткий запрос к ComfyUI; любые сбои health не валят."""

    method = getattr(comfy, "system_stats", None)
    if method is None:
        return None
    try:
        return await method(timeout=timeout)
    except TypeError:
        # Упрощённые заглушки/старые клиенты могут не принимать timeout.
        return await method()


async def health_payload(config: Config, comfy: Any) -> dict[str, Any]:
    """Собрать ``{status, comfyui, volume_mounted}`` без генерации."""

    volume_mounted = Path(config.volume_root).is_dir()
    comfyui = "unavailable"

    if comfy is not None:
        stats: dict[str, Any] | None = None
        try:
            stats = await _system_stats(comfy, config.health_timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - health не должен падать
            _LOG.warning("health: /system_stats упал: %s: %s", type(exc).__name__, exc)
            stats = None
        if isinstance(stats, dict) and stats:
            comfyui = "ready"
        else:
            last_error = getattr(comfy, "last_error", None)
            comfyui = "booting" if last_error in {"connect", "timeout"} else "unavailable"

    if comfyui == "ready" and volume_mounted:
        status = "ok"
    elif comfyui == "booting" and volume_mounted:
        status = "starting"
    else:
        status = "degraded"

    return {
        "status": status,
        "comfyui": comfyui,
        "volume_mounted": volume_mounted,
    }


__all__ = ["health_payload"]
