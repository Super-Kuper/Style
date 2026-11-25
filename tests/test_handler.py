"""Handler: сквозной поток через httpx.MockTransport, passthrough графа, отмена."""

from __future__ import annotations

import asyncio
import base64
import dataclasses
import logging
import sys

import pytest

from helpers import MockComfyUI, png_bytes, t2i_graph
from image_worker.errors import WorkerError
from image_worker.handler import Dependencies, build_handler, execute
from image_worker.imaging import read_image_info

JOB_GRAPH = t2i_graph()
PARAMETER = {
    "mode": "generate",
    "scene_id": 2,
    "width": 768,
    "height": 1344,
    "seed": 20261113,
    "steps": 4,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
    "model": {"slug": "flux2-klein-4b", "graph_revision": 3},
}


def _job(**input_overrides):
    payload = {"prompt": JOB_GRAPH, "parameter": PARAMETER}
    payload.update(input_overrides)
    return {"id": "job-1", "input": payload}


def _run(coro):
    return asyncio.run(coro)


def _gpu():
    return 8.6, "RTX A6000"


def _handler(config, comfy, **kwargs):
    return build_handler(config, comfy, gpu_provider=_gpu, **kwargs)


def test_success_returns_image_and_measured_telemetry(config, make_comfy) -> None:
    mock = MockComfyUI(image_bytes=png_bytes(768, 1344))
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        result = _run(handler(_job()))
    finally:
        _run(comfy.close())

    assert set(result) == {"image", "telemetry"}
    assert base64.b64decode(result["image"]) == mock.image_bytes
    telemetry = result["telemetry"]
    assert telemetry["gpu_cc"] == 8.6
    assert telemetry["gpu_class"] == "RTX A6000"
    assert telemetry["width"] == 768
    assert telemetry["height"] == 1344
    assert telemetry["model_name"] == "flux-2-klein-4b.safetensors"
    assert telemetry["seed"] == 20261113
    assert telemetry["steps"] == 4
    assert telemetry["cfg"] == 1.0
    assert telemetry["sampler"] == "euler"
    assert telemetry["scheduler"] == "simple"
    assert telemetry["seconds"] > 0
    assert "text_detected" not in telemetry
    assert mock.interrupts == 0


def test_graph_is_submitted_unchanged(config, make_comfy) -> None:
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    graph = t2i_graph(seed=123, steps=12)
    try:
        _run(handler({"id": "j", "input": {"prompt": graph, "parameter": {}}}))
    finally:
        _run(comfy.close())
    assert mock.submitted_graphs == [graph]


def test_assets_delivered_before_prompt(config, make_comfy) -> None:
    import httpx

    from image_worker.assets import download_assets

    asset = b"base-image"
    mock = MockComfyUI(http_files={"/base.png": asset})
    comfy = make_comfy(mock)

    async def assets_downloader(assets, cfg):
        # Assets идут через тот же MockTransport: сеть в тестах не нужна.
        async with httpx.AsyncClient(transport=mock.transport()) as client:
            return await download_assets(assets, cfg, http=client)

    handler = build_handler(config, comfy, gpu_provider=_gpu, assets_downloader=assets_downloader)
    job = _job(assets=[{"name": "base.png", "url": "https://files.example/base.png", "kind": "image"}])
    try:
        _run(handler(job))
    finally:
        _run(comfy.close())

    paths = mock.paths()
    assert "/base.png" in paths and "/prompt" in paths
    assert paths.index("/base.png") < paths.index("/prompt")


def test_health_action(config, make_comfy, tmp_path) -> None:
    from pathlib import Path

    Path(config.volume_root).mkdir(parents=True)
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        result = _run(handler({"id": "h", "input": {"action": "health"}}))
    finally:
        _run(comfy.close())

    assert result == {"status": "ok", "comfyui": "ready", "volume_mounted": True}
    assert "/prompt" not in mock.paths()
    assert "/view" not in mock.paths()


def test_unknown_action_is_input_invalid(config, make_comfy) -> None:
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler({"id": "x", "input": {"action": "warmup"}}))
    finally:
        _run(comfy.close())
    assert error.value.code == "input_invalid"


def test_missing_prompt_is_visible(config, make_comfy) -> None:
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler({"id": "x", "input": {"parameter": {}}}))
    finally:
        _run(comfy.close())
    assert error.value.code == "input_prompt_missing"


@pytest.mark.parametrize("job", [
    None,
    {"id": "x"},
    {"id": "x", "input": "not-an-object"},
])
def test_invalid_job_is_input_invalid(config, make_comfy, job) -> None:
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler(job))
    finally:
        _run(comfy.close())
    assert error.value.code == "input_invalid"


def test_comfy_execution_error(config, make_comfy) -> None:
    mock = MockComfyUI(history_status="error", history_error_message="node exploded")
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler(_job()))
    finally:
        _run(comfy.close())
    assert error.value.code == "comfy_execution_failed"
    assert "node exploded" in error.value.message


def test_comfy_submit_error(config, make_comfy) -> None:
    mock = MockComfyUI(submit_status=400, submit_body={"error": {"message": "invalid prompt"}})
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler(_job()))
    finally:
        _run(comfy.close())
    assert error.value.code == "comfy_submit_failed"
    assert "invalid prompt" in error.value.message


def test_output_missing(config, make_comfy) -> None:
    mock = MockComfyUI(history_outputs={})
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler(_job()))
    finally:
        _run(comfy.close())
    assert error.value.code == "comfy_output_missing"


def test_unreadable_output_is_output_invalid(config, make_comfy) -> None:
    mock = MockComfyUI(image_bytes=b"not-an-image")
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with pytest.raises(WorkerError) as error:
            _run(handler(_job()))
    finally:
        _run(comfy.close())
    assert error.value.code == "output_invalid"


def test_large_image_goes_to_storage_url(config, make_comfy) -> None:
    from image_worker.config import Config

    small_inline = Config.from_env({
        "COMFYUI_INPUT_DIR": config.comfyui_input_dir,
        "VOLUME_ROOT": config.volume_root,
        "COMFY_BOOT_TIMEOUT_SECONDS": "2",
        "JOB_TIMEOUT_SECONDS": "2",
        "COMFY_POLL_INITIAL_SECONDS": "0.001",
        "COMFY_POLL_MAX_SECONDS": "0.002",
        "OUTPUT_INLINE_MAX_MB": "0.0001",
        "OUTPUT_INLINE_BUDGET_MB": "0.0001",
    })

    class Storage:
        def __init__(self) -> None:
            self.uploaded = 0

        async def upload(self, data: bytes, key: str, content_type: str) -> str:
            self.uploaded += 1
            return key

        async def presigned_url(self, key: str, ttl_seconds: int) -> str:
            return f"https://s3.example/{key}"

    storage = Storage()
    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = build_handler(small_inline, comfy, storage=storage, gpu_provider=_gpu)
    try:
        result = _run(handler(_job()))
    finally:
        _run(comfy.close())
    assert set(result) == {"image_url", "telemetry"}
    assert storage.uploaded == 1


def test_timeout_interrupts_graph_and_drops_it_from_queue(config, make_comfy) -> None:
    """Таймаут ожидания освобождает GPU: иначе граф считается, а следующий job ждёт."""

    fast = dataclasses.replace(config, job_timeout_seconds=0.05)
    mock = MockComfyUI(not_ready_polls=10**9)
    comfy = make_comfy(mock)
    handler = build_handler(fast, comfy, gpu_provider=_gpu)
    try:
        with pytest.raises(WorkerError) as caught:
            _run(handler(_job()))
    finally:
        _run(comfy.close())

    assert caught.value.code == "comfy_timeout"
    assert mock.interrupts == 1
    assert mock.queue_deletes == [mock.prompt_id]


def test_job_logs_graph_summary_before_submit(config, make_comfy, caplog) -> None:
    """До submit в лог уходят шаги, размер и файлы моделей из графа."""

    mock = MockComfyUI()
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with caplog.at_level(logging.INFO, logger="image_worker.handler"):
            _run(handler(_job()))
    finally:
        _run(comfy.close())

    assert "нод 9" in caplog.text
    assert "шагов [4]" in caplog.text
    assert "размер 768x1344" in caplog.text
    assert "flux-2-klein-4b.safetensors" in caplog.text
    assert "ожидание до 2s" in caplog.text


def test_wait_logs_queue_transition(config, make_comfy, caplog) -> None:
    """Переход «в очереди → выполняется» виден в логе ожидания."""

    mock = MockComfyUI(not_ready_polls=2)
    mock.queue_running = [mock.prompt_id]
    comfy = make_comfy(mock)
    handler = _handler(config, comfy)
    try:
        with caplog.at_level(logging.INFO, logger="image_worker.comfy_client"):
            _run(handler(_job()))
    finally:
        _run(comfy.close())

    assert "выполняется" in caplog.text
    assert f"ComfyUI {mock.prompt_id}" in caplog.text


def test_cancellation_interrupts_comfy(config) -> None:
    class CancelComfy:
        def __init__(self) -> None:
            self.interrupted = 0

        async def wait_ready(self, timeout=None):
            return {}

        async def submit(self, prompt):
            return "prompt-1"

        async def wait(self, prompt_id, timeout=None):
            raise asyncio.CancelledError()

        async def fetch_output(self, history_entry):  # pragma: no cover - not reached
            raise AssertionError("отмена не должна доходить до /view")

        async def interrupt(self):
            self.interrupted += 1

    async def assets_downloader(assets, cfg):
        return []

    comfy = CancelComfy()
    deps = Dependencies(
        config=config,
        comfy=comfy,
        assets_downloader=assets_downloader,
        image_reader=read_image_info,
        gpu_provider=_gpu,
    )
    with pytest.raises(asyncio.CancelledError):
        _run(execute(_job(), deps))
    assert comfy.interrupted == 1


def test_contract_request_examples_end_to_end(config, make_comfy) -> None:
    """Примеры контракта проходят handler целиком и валидируются output-схемой."""

    import hashlib

    import httpx
    from jsonschema import Draft202012Validator

    from helpers import contract_dir, load_json
    from image_worker.assets import download_assets

    output_schema = load_json(contract_dir() / "output.schema.json")

    base = png_bytes(768, 1344)

    def asset_entry(name: str, kind: str, content: bytes) -> dict:
        return {
            "name": name,
            "url": f"https://files.example/{name}",
            "sha256": hashlib.sha256(content).hexdigest(),
            "kind": kind,
            "size": len(content),
        }

    assets = [asset_entry("base.png", "image", base), asset_entry("mask.png", "mask", base)]
    payload = load_json(contract_dir() / "examples" / "request-edit.json")
    payload["input"]["assets"] = assets

    mock = MockComfyUI(image_bytes=base, http_files={"/base.png": base, "/mask.png": base})
    comfy = make_comfy(mock)

    async def assets_downloader(items, cfg):
        async with httpx.AsyncClient(transport=mock.transport()) as client:
            return await download_assets(items, cfg, http=client)

    handler = build_handler(config, comfy, gpu_provider=_gpu, assets_downloader=assets_downloader)
    try:
        result = _run(handler({"id": "example", "input": payload["input"]}))
    finally:
        _run(comfy.close())

    Draft202012Validator(output_schema).validate(result)
    telemetry = result["telemetry"]
    assert telemetry["model_name"] == "qwen_image_edit_2511_fp8mixed.safetensors"
    assert telemetry["steps"] == 20
    assert telemetry["cfg"] == 2.5
    assert telemetry["seed"] == 20261113
    assert (telemetry["width"], telemetry["height"]) == (768, 1344)
    assert set(mock.submitted_asset_paths()) == {"/base.png", "/mask.png"}


def test_import_without_runpod() -> None:
    # Тесты обязаны идти без runpod SDK: пакет не импортирует его на уровне
    # модулей, а main.py делает ленивый импорт внутри main().
    sys.modules.pop("runpod", None)
    import image_worker  # noqa: F401
    import image_worker.handler  # noqa: F401
    import image_worker.main  # noqa: F401

    assert "runpod" not in sys.modules
