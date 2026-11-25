"""Serverless-воркер изображений: готовый ComfyUI-граф → картинка + телеметрия.

Контракт `image-serverless/v1`: вход `{input: {prompt, parameter, acceleration?,
assets?, action?}}`, выход `{image | image_url, telemetry}`; источник истины —
`contract/worker-contract.md`.

Пакет намеренно не импортирует тяжёлые зависимости (`torch`, `boto3`, `runpod`)
на уровне модулей: тесты должны идти без GPU, сети и runpod SDK.
"""

__version__ = "1.0.0"
