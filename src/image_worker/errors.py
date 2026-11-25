"""Ошибки воркера: стабильный код + человекочитаемый текст.

Коды — латиницей snake_case, список зафиксирован в
`contract/worker-contract.md` §4: ошибка handler'а → job `FAILED` → приложение
видит `serverless_failed` и текст ошибки. Молчаливый фолбэк запрещён: любое
расхождение контракта — видимая ошибка.
"""

from __future__ import annotations


class WorkerError(RuntimeError):
    """Ошибка воркера с машинным `code` и текстом `message`.

    Пример:
        >>> err = WorkerError("comfy_timeout", "ComfyUI не ответил за 60 с")
        >>> err.code, err.message
        ('comfy_timeout', 'ComfyUI не ответил за 60 с')
        >>> str(err)
        'comfy_timeout: ComfyUI не ответил за 60 с'
    """

    def __init__(self, code: str, message: str = "") -> None:
        self.code = str(code)
        self.message = str(message or "")
        text = f"{self.code}: {self.message}" if self.message else self.code
        self._text = text
        super().__init__(text)

    def __str__(self) -> str:
        return self._text

    def __repr__(self) -> str:
        return f"WorkerError(code={self.code!r}, message={self.message!r})"
