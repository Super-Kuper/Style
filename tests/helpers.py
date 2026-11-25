"""Test helpers: image bytes, ComfyUI graphs and an httpx.MockTransport ComfyUI.

No GPU, no network and no runpod SDK are needed: the ComfyUI HTTP API is mocked
at the transport level (`httpx.MockTransport`), which is exactly how the worker
talks to it in production.
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path
from typing import Any

import httpx

DEFAULT_PNG_SIZE = (768, 1344)


def png_bytes(width: int = 4, height: int = 3) -> bytes:
    """Valid PNG with IHDR and a compressed solid-colour raster."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    signature = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    raw = b"".join(b"\x00" + b"\x00\x00\x00" * width for _ in range(height))
    return signature + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")


def jpeg_bytes(width: int = 7, height: int = 5) -> bytes:
    """Minimal JPEG header with SOF0 (enough to read dimensions)."""

    app0 = b"\xff\xe0" + struct.pack(">H", 16) + b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    sof0 = b"\xff\xc0" + struct.pack(">HBHHB", 11, 8, height, width, 1) + b"\x01\x11\x00"
    return b"\xff\xd8" + app0 + sof0 + b"\xff\xd9"


def t2i_graph(
    *,
    seed: int = 20261113,
    steps: int = 4,
    cfg: float = 1.0,
    sampler: str = "euler",
    scheduler: str = "simple",
    unet_name: str = "flux-2-klein-4b.safetensors",
) -> dict[str, Any]:
    """Generic KSampler-based graph fixture (the refs profile ships no t2i example)."""

    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": unet_name, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "4": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": "a rainy street"}},
        "5": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["2", 0], "text": ""}},
        "6": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
                "seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": 1.0,
            },
        },
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["7", 0], "vae": ["3", 0]}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "keyframe"}},
    }


def custom_advanced_graph(*, seed: int = 42, steps: int = 8, cfg: float = 3.5) -> dict[str, Any]:
    """SamplerCustomAdvanced graph: facts live in linked RandomNoise/BasicScheduler/CFGGuider."""

    return {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors"}},
        "2": {"class_type": "RandomNoise", "inputs": {"noise_seed": seed}},
        "3": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": steps, "denoise": 1.0}},
        "4": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "5": {
            "class_type": "CFGGuider",
            "inputs": {"model": ["1", 0], "positive": ["7", 0], "negative": ["8", 0], "cfg": cfg},
        },
        "6": {"class_type": "SamplerCustomAdvanced", "inputs": {"noise": ["2", 0], "guider": ["5", 0], "sampler": ["4", 0], "sigmas": ["3", 0], "latent_image": ["9", 0]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 0], "text": "a cat"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["10", 0], "text": ""}},
        "9": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
        "10": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen_3_4b.safetensors", "type": "flux2"}},
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["6", 0], "vae": ["12", 0]}},
        "12": {"class_type": "VAELoader", "inputs": {"vae_name": "flux2-vae.safetensors"}},
        "13": {"class_type": "SaveImage", "inputs": {"images": ["11", 0], "filename_prefix": "keyframe"}},
    }


def contract_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "contract"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


class MockComfyUI:
    """In-process ComfyUI HTTP API used through ``httpx.MockTransport``.

    Records requests, captures the submitted graph and returns a configurable
    history/output. All endpoints are real HTTP paths the worker calls.
    """

    def __init__(
        self,
        *,
        image_bytes: bytes | None = None,
        history_outputs: dict[str, Any] | None = None,
        submit_status: int = 200,
        submit_body: Any = None,
        history_status: str = "success",
        history_error_message: str = "boom",
        view_status: int = 200,
        stats_status: int = 200,
        stats_body: Any = None,
        prompt_id: str = "prompt-1",
        http_files: dict[str, bytes] | None = None,
        not_ready_polls: int = 0,
    ) -> None:
        self.image_bytes = image_bytes if image_bytes is not None else png_bytes(*DEFAULT_PNG_SIZE)
        self.history_outputs = history_outputs
        self.submit_status = submit_status
        self.submit_body = submit_body
        self.history_status = history_status
        self.history_error_message = history_error_message
        self.view_status = view_status
        self.stats_status = stats_status
        self.stats_body = stats_body if stats_body is not None else {"system": {"comfyui_version": "0.44.0"}}
        self.prompt_id = prompt_id
        self.http_files = http_files or {}
        self.requests: list[tuple[str, str]] = []
        self.submitted_graphs: list[dict[str, Any]] = []
        self.interrupts = 0
        # GET /queue: какие графы ComfyUI выполняет и держит в очереди.
        self.queue_running: list[str] = []
        self.queue_pending: list[str] = []
        self.queue_deletes: list[str] = []
        self._history_polls = 0
        self.not_ready_polls = not_ready_polls

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def _default_outputs(self) -> dict[str, Any]:
        return {
            "9": {
                "images": [
                    {"filename": "keyframe_00001_.png", "subfolder": "", "type": "output"},
                ]
            }
        }

    def _handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))

        if path == "/system_stats":
            if self.stats_status >= 400:
                return httpx.Response(self.stats_status, text="unavailable")
            return httpx.Response(200, json=self.stats_body)

        if path == "/prompt":
            body = json.loads(request.content.decode("utf-8"))
            self.submitted_graphs.append(body.get("prompt"))
            if self.submit_status >= 400:
                if self.submit_body is not None:
                    return httpx.Response(self.submit_status, json=self.submit_body)
                return httpx.Response(self.submit_status, json={"error": {"message": "bad graph"}})
            return httpx.Response(200, json={"prompt_id": self.prompt_id})

        if path == "/queue":
            if request.method == "POST":
                body = json.loads(request.content.decode("utf-8") or "{}")
                self.queue_deletes.extend(str(item) for item in (body.get("delete") or []))
                return httpx.Response(200, json={})
            return httpx.Response(
                200,
                json={
                    "queue_running": [[index, pid, {}, {}, []] for index, pid in enumerate(self.queue_running)],
                    "queue_pending": [[index, pid, {}, {}, []] for index, pid in enumerate(self.queue_pending)],
                },
            )

        if path.startswith("/history/"):
            self._history_polls += 1
            if self._history_polls <= self.not_ready_polls:
                return httpx.Response(200, json={})
            status = {
                "status_str": self.history_status,
                "completed": self.history_status == "success",
                "messages": [],
            }
            if self.history_status == "error":
                status["messages"] = [["execution_error", {"exception_message": self.history_error_message}]]
            outputs = self.history_outputs if self.history_outputs is not None else self._default_outputs()
            return httpx.Response(200, json={self.prompt_id: {"status": status, "outputs": outputs}})

        if path == "/view":
            if self.view_status >= 400:
                return httpx.Response(self.view_status, text="no file")
            return httpx.Response(200, content=self.image_bytes, headers={"content-type": "image/png"})

        if path == "/interrupt":
            self.interrupts += 1
            return httpx.Response(200, json={})

        # Presigned asset URLs (host/path/query arbitrary).
        if path in self.http_files:
            return httpx.Response(200, content=self.http_files[path])

        return httpx.Response(404, json={"error": f"unknown path {path}"})

    # -- assertions helpers -------------------------------------------------

    def paths(self) -> list[str]:
        return [path for _method, path in self.requests]

    def submitted_asset_paths(self) -> list[str]:
        return [path for path in self.paths() if path in self.http_files]
