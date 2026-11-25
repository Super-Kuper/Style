"""RunPod serverless entrypoint for the image worker.

The container entrypoint (docker/entrypoint.sh) boots ComfyUI on loopback,
downloads the missing models to the network volume and then runs this file.
The job handler itself lives in ``src/image_worker/handler.py``; this module only
makes the ``image_worker`` package importable (``PYTHONPATH`` is already set in
the image, the ``sys.path`` line covers a plain repository checkout) and starts
the RunPod serverless loop.

Job contract: ``contract/worker-contract.md`` (``image-serverless/v1``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# В образе PYTHONPATH уже указывает на /opt/image-worker/src; при локальном
# запуске `python handler.py` из чистого клона пакет лежит рядом со скриптом.
_SRC_DIR = Path(__file__).resolve().parent / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


def main() -> None:
    """Start the RunPod serverless loop with the worker handler."""

    # Импорты внутри функции: `import handler` не требует runpod/httpx, если
    # нужен только сам модуль (проверки структуры образа, smoke).
    import runpod

    from image_worker.handler import handler

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
