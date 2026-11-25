"""Health: статусы ok/starting/degraded по тому и /system_stats, без графа."""

from __future__ import annotations

import asyncio
from pathlib import Path

from image_worker.health import health_payload


class ComfyStub:
    def __init__(self, *, stats=None, last_error=None, raises=False) -> None:
        self._stats = stats if stats is not None else {"system": {}}
        self.last_error = last_error
        self._raises = raises
        self.calls = 0

    async def system_stats(self, timeout=None):
        self.calls += 1
        if self._raises:
            raise RuntimeError("comfy exploded")
        return self._stats


def _run(coro):
    return asyncio.run(coro)


def test_ready_and_volume_mounted(config, tmp_path) -> None:
    Path(config.volume_root).mkdir(parents=True)
    result = _run(health_payload(config, ComfyStub()))
    assert result == {"status": "ok", "comfyui": "ready", "volume_mounted": True}


def test_booting_with_volume_is_starting(config) -> None:
    Path(config.volume_root).mkdir(parents=True)
    result = _run(health_payload(config, ComfyStub(stats={}, last_error="connect")))
    assert result["comfyui"] == "booting"
    assert result["status"] == "starting"


def test_unavailable_without_volume_is_degraded(config) -> None:
    result = _run(health_payload(config, ComfyStub(stats={}, last_error="error")))
    assert result == {"status": "degraded", "comfyui": "unavailable", "volume_mounted": False}


def test_system_stats_exception_does_not_break_health(config) -> None:
    Path(config.volume_root).mkdir(parents=True)
    result = _run(health_payload(config, ComfyStub(raises=True)))
    assert result["status"] == "degraded"
    assert result["comfyui"] == "unavailable"


def test_no_comfy_client_is_degraded(config) -> None:
    Path(config.volume_root).mkdir(parents=True)
    result = _run(health_payload(config, None))
    assert result["comfyui"] == "unavailable"
    assert result["status"] == "degraded"


def test_health_action_does_not_touch_graph_or_assets(config) -> None:
    from image_worker.handler import Dependencies, execute

    class Comfy(ComfyStub):
        async def wait_ready(self, timeout=None):  # pragma: no cover - must not be called
            raise AssertionError("health не должен ждать ComfyUI")

        async def submit(self, prompt):  # pragma: no cover - must not be called
            raise AssertionError("health не должен запускать граф")

    Path(config.volume_root).mkdir(parents=True)
    deps = Dependencies(config=config, comfy=Comfy())
    result = _run(execute({"id": "h1", "input": {"action": "health"}}, deps))
    assert result["status"] == "ok"
    assert result["comfyui"] == "ready"
    assert result["volume_mounted"] is True
