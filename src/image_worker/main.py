"""Точка входа RunPod serverless.

RunPod запускает ``main()``; handler — `image_worker.handler.handler`. Импорт
`runpod` ленивый: пакет и тесты живут без него (SDK ставится только в образ,
runpod==2.2.1 из pyproject).
"""

from __future__ import annotations


def main() -> None:
    """Запустить serverless-цикл RunPod с нашим handler'ом."""

    import runpod  # noqa: PLC0415 - ленивый импорт: import пакета без runpod

    from .handler import handler

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
