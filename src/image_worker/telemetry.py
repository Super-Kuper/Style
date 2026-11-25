"""Телеметрия факта `image-serverless/v1`: исполненный граф + картинка + замер.

Источник истины — **факты исполнения** (`contract/worker-contract.md` §2):

- `gpu_cc`/`gpu_class` — ``torch.cuda`` (или env-override `IMAGE_GPU_CC`/
  `IMAGE_GPU_CLASS` для окружений без torch);
- `seconds` — замер handler'а (submit → history completed);
- `width`/`height` — заголовок готового файла (`imaging.read_image_info`), а не
  снимок `parameter`;
- `model_name`, `seed`, `steps`, `cfg`, `sampler`, `scheduler` — по фактически
  исполненному графу `prompt` (API-format ComfyUI).

Ожидаемые значения из `parameter`/`acceleration` **не подставляются**: при
невозможности измерить обязательный факт — видимая ошибка
``telemetry_missing``, при расхождении фактов между узлами —
``telemetry_ambiguous``.

`text_detected` (OCR) воркер не измеряет: OCR-движка в образе нет, поэтому поле
в ответ не попадает. Это не молчаливый `false`; OCR-факт фиксирует сервис
(`docs/tasks/2026-11-12-keyframe-implementation.md`, `keyframe_text_detected`).
"""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from typing import Any

from .errors import WorkerError
from .imaging import ImageInfo

_KNOWN_GPU_CLASSES = (
    "RTX A6000",
    "A40",
    "L40S",
    "L40",
    "RTX 6000 Ada",
    "RTX PRO 4500",
    "RTX 5090",
    "RTX 4090",
)

# Узлы-сэмплеры: у KSampler*/факты в inputs, у SamplerCustom* — в связанных
# узлах (noise/guider/sampler/sigmas).
_SAMPLER_CLASSES = ("KSampler", "KSamplerAdvanced", "SamplerCustom", "SamplerCustomAdvanced")
_SCHEDULER_CLASSES = ("BasicScheduler",)
_SELECT_CLASSES = ("KSamplerSelect",)
_NOISE_CLASSES = ("RandomNoise",)
_CFG_GUIDER_CLASSES = ("CFGGuider",)
# Узлы-загрузчики весов: имя файла диффузионной модели — это model_name.
_UNET_LOADER_CLASSES = ("UNETLoader", "UnetLoaderGGUF", "UnetLoaderGGUFAdvanced")
_CHECKPOINT_LOADER_CLASSES = ("CheckpointLoaderSimple", "CheckpointLoader")

# Факты, которые даёт сэмплер-узел (model_name ищется отдельно по загрузчикам).
_SAMPLER_FACT_KEYS = ("seed", "steps", "cfg", "sampler", "scheduler")

# Поля обязательного минимума (`contract/output.schema.json`).
_REQUIRED = (
    ("gpu_cc", "number"),
    ("gpu_class", "string"),
    ("seconds", "number"),
    ("width", "integer"),
    ("height", "integer"),
    ("model_name", "string"),
    ("seed", "integer"),
    ("steps", "integer"),
    ("cfg", "number"),
    ("sampler", "string"),
    ("scheduler", "string"),
)


# ---------------------------------------------------------------------------
# GPU
# ---------------------------------------------------------------------------


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _torch_gpu() -> tuple[float, str]:
    """Фактические cc/class из torch.cuda (ленивый импорт — тесты без torch)."""

    try:
        import torch  # noqa: PLC0415 - намеренно ленивый импорт
    except Exception as exc:  # noqa: BLE001 - нет torch/сломанное окружение
        raise WorkerError(
            "gpu_unavailable",
            f"torch недоступен, а IMAGE_GPU_CC/IMAGE_GPU_CLASS не заданы: {type(exc).__name__}: {exc}",
        ) from exc

    cuda = getattr(torch, "cuda", None)
    try:
        available = bool(cuda is not None and cuda.is_available())
    except Exception as exc:  # noqa: BLE001 - CUDA может бросить на старте
        raise WorkerError("gpu_unavailable", f"torch.cuda.is_available() упал: {exc}") from exc
    if not available:
        raise WorkerError("gpu_unavailable", "CUDA недоступна: torch.cuda.is_available() = False")

    try:
        major, minor = cuda.get_device_capability(0)
        name = str(cuda.get_device_name(0))
    except Exception as exc:  # noqa: BLE001
        raise WorkerError("gpu_unavailable", f"не удалось прочитать GPU через torch.cuda: {exc}") from exc
    return float(f"{int(major)}.{int(minor)}"), _gpu_class_from_name(name)


def _gpu_class_from_name(name: str) -> str:
    """Класс GPU по имени устройства; неизвестное имя возвращаем как факт."""

    upper = name.upper()
    for gpu_class in _KNOWN_GPU_CLASSES:
        if gpu_class.upper() in upper:
            return gpu_class
    return name


def detect_gpu(env: Mapping[str, str] | None = None) -> tuple[float, str]:
    """Вернуть фактическую пару ``(gpu_cc, gpu_class)``.

    Приоритет: явный override ``IMAGE_GPU_CC``/``IMAGE_GPU_CLASS`` (окружения без
    torch/для E2E-диагностики) → ``torch.cuda`` (capability + имя). Недоступно →
    ``WorkerError(code="gpu_unavailable")``; выдумывать класс GPU нельзя.
    """

    source: Mapping[str, str] = os.environ if env is None else env
    cc_raw = _clean(source.get("IMAGE_GPU_CC"))
    class_raw = _clean(source.get("IMAGE_GPU_CLASS"))
    if cc_raw is not None or class_raw is not None:
        if cc_raw is None or class_raw is None:
            raise WorkerError(
                "config_invalid",
                "IMAGE_GPU_CC и IMAGE_GPU_CLASS должны задаваться вместе",
            )
        try:
            cc = float(cc_raw)
        except ValueError as exc:
            raise WorkerError(
                "config_invalid",
                f"IMAGE_GPU_CC: ожидалось число, получено {cc_raw!r}",
            ) from exc
        if not math.isfinite(cc) or cc <= 0:
            raise WorkerError("config_invalid", f"IMAGE_GPU_CC: ожидалось положительное число, получено {cc_raw!r}")
        return cc, class_raw
    return _torch_gpu()


# ---------------------------------------------------------------------------
# Значения из графа.
# ---------------------------------------------------------------------------


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _int_value(value: Any) -> int | None:
    """Целое из inputs графа (int или целочисленный float), иначе ``None``."""

    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return None


def _float_value(value: Any) -> float | None:
    """Конечное число из inputs графа (int/float/числовая строка)."""

    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _str_value(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _iter_nodes(prompt: Any) -> Iterator[tuple[str, Mapping[str, Any]]]:
    """Узлы графа в порядке графа: ``(id, {class_type, inputs})``."""

    if not isinstance(prompt, Mapping):
        return
    for node_id, node in prompt.items():
        if isinstance(node, Mapping):
            yield str(node_id), node


def _node_class(node: Mapping[str, Any]) -> str | None:
    class_type = node.get("class_type")
    return class_type if isinstance(class_type, str) else None


def _node_inputs(node: Mapping[str, Any]) -> Mapping[str, Any]:
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, Mapping) else {}


def _link_target(value: Any) -> str | None:
    """Цель связи ComfyUI ``[node_id, slot]`` → ``str(node_id)``."""

    if isinstance(value, (list, tuple)) and len(value) == 2 and not isinstance(value[0], bool):
        return str(value[0])
    return None


def _linked_node(
    nodes: Mapping[str, Mapping[str, Any]],
    value: Any,
    expected: tuple[str, ...],
) -> Mapping[str, Any] | None:
    """Узел по связи, если его class_type входит в `expected`."""

    target = _link_target(value)
    if target is None:
        return None
    node = nodes.get(target)
    if node is None or _node_class(node) not in expected:
        return None
    return node


# ---------------------------------------------------------------------------
# Факты сэмплера.
# ---------------------------------------------------------------------------


def _sampler_node_facts(node: Mapping[str, Any], nodes: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    """Извлечь факты одного сэмплер-узла; отсутствующие — ``None``."""

    inputs = _node_inputs(node)
    seed = _int_value(inputs.get("seed"))
    if seed is None:
        # SamplerCustom хранит seed прямо в noise_seed; RandomNoise — в линке noise.
        seed = _int_value(inputs.get("noise_seed"))
    facts: dict[str, Any] = {
        "seed": seed,
        "steps": _int_value(inputs.get("steps")),
        "cfg": _float_value(inputs.get("cfg")),
        "sampler": _str_value(inputs.get("sampler_name")),
        "scheduler": _str_value(inputs.get("scheduler")),
    }

    if facts["seed"] is None:
        noise = _linked_node(nodes, inputs.get("noise"), _NOISE_CLASSES)
        if noise is not None:
            noise_inputs = _node_inputs(noise)
            facts["seed"] = _int_value(noise_inputs.get("noise_seed"))
            if facts["seed"] is None:
                facts["seed"] = _int_value(noise_inputs.get("seed"))

    if facts["steps"] is None or facts["scheduler"] is None:
        scheduler_node = _linked_node(nodes, inputs.get("sigmas"), _SCHEDULER_CLASSES)
        if scheduler_node is not None:
            scheduler_inputs = _node_inputs(scheduler_node)
            if facts["steps"] is None:
                facts["steps"] = _int_value(scheduler_inputs.get("steps"))
            if facts["scheduler"] is None:
                facts["scheduler"] = _str_value(scheduler_inputs.get("scheduler"))

    if facts["sampler"] is None:
        select_node = _linked_node(nodes, inputs.get("sampler"), _SELECT_CLASSES)
        if select_node is not None:
            facts["sampler"] = _str_value(_node_inputs(select_node).get("sampler_name"))

    if facts["cfg"] is None:
        guider_node = _linked_node(nodes, inputs.get("guider"), _CFG_GUIDER_CLASSES)
        if guider_node is not None:
            facts["cfg"] = _float_value(_node_inputs(guider_node).get("cfg"))

    return facts


def _sampler_facts(prompt: Mapping[str, Any]) -> dict[str, Any]:
    """Согласованные факты по всем сэмплер-узлам графа.

    Логика: по каждому ключу собираем значения всех сэмплер-узлов, которые его
    дают. Нет ни одного значения — ``telemetry_missing``; разные значения —
    ``telemetry_ambiguous`` (у телеметрии контракта одно поле на ключ, а
    «первый попавшийся» узел — не факт).
    """

    nodes = dict(_iter_nodes(prompt))
    samplers: list[tuple[str, Mapping[str, Any]]] = [
        (node_id, node) for node_id, node in nodes.items() if _node_class(node) in _SAMPLER_CLASSES
    ]
    if not samplers:
        raise WorkerError(
            "telemetry_missing",
            "факты исполнения: в графе нет сэмплер-узла "
            f"({', '.join(_SAMPLER_CLASSES)}) — seed/steps/cfg/sampler/scheduler не измерить",
        )

    collected: dict[str, list[tuple[str, Any]]] = {key: [] for key in _SAMPLER_FACT_KEYS}
    for node_id, node in samplers:
        facts = _sampler_node_facts(node, nodes)
        for key in _SAMPLER_FACT_KEYS:
            value = facts.get(key)
            if value is not None:
                collected[key].append((node_id, value))

    result: dict[str, Any] = {}
    for key in _SAMPLER_FACT_KEYS:
        found = collected[key]
        if not found:
            raise WorkerError(
                "telemetry_missing",
                f"факты исполнения: в графе нет значения {key!r} "
                f"({'/'.join(_SAMPLER_CLASSES)}); ожидаемые значения не подставляются",
            )
        values = sorted({value for _, value in found})
        if len(values) > 1:
            detail = "; ".join(f"node {node_id}: {key}={value!r}" for node_id, value in found)
            raise WorkerError(
                "telemetry_ambiguous",
                f"факты исполнения: {key!r} в исполненном графе расходится: {detail}",
            )
        result[key] = values[0]
    return result


def _model_name_from_graph(prompt: Mapping[str, Any]) -> str:
    """Имя файла диффузионной модели по узлам-загрузчикам графа."""

    unet_names: list[tuple[str, str]] = []
    checkpoint_names: list[tuple[str, str]] = []
    for node_id, node in _iter_nodes(prompt):
        class_type = _node_class(node)
        inputs = _node_inputs(node)
        if class_type in _UNET_LOADER_CLASSES:
            name = _str_value(inputs.get("unet_name"))
            if name is not None:
                unet_names.append((node_id, name))
        elif class_type in _CHECKPOINT_LOADER_CLASSES:
            name = _str_value(inputs.get("ckpt_name"))
            if name is not None:
                checkpoint_names.append((node_id, name))

    for label, found in (("unet_name", unet_names), ("ckpt_name", checkpoint_names)):
        values = sorted({name for _, name in found})
        if len(values) > 1:
            detail = "; ".join(f"node {node_id}: {name}" for node_id, name in found)
            raise WorkerError(
                "telemetry_ambiguous",
                f"факты исполнения: model_name по {label} расходится: {detail}",
            )
        if len(values) == 1:
            return values[0]

    raise WorkerError(
        "telemetry_missing",
        "факты исполнения: в графе нет узла-загрузчика весов "
        f"({', '.join(_UNET_LOADER_CLASSES + _CHECKPOINT_LOADER_CLASSES)}) — "
        "model_name не измерить; ожидаемые значения не подставляются",
    )


# ---------------------------------------------------------------------------
# Проверки обязательных полей и сборка.
# ---------------------------------------------------------------------------


def _check_telemetry(telemetry: Mapping[str, Any]) -> dict[str, Any]:
    """Проверить обязательный минимум/типы; нарушения — ``telemetry_missing``."""

    problems: list[str] = []
    for name, kind in _REQUIRED:
        if name not in telemetry or telemetry[name] is None:
            problems.append(name)
            continue
        value = telemetry[name]
        if kind == "number":
            ok = (isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)))
        elif kind == "integer":
            ok = _is_int(value)
        else:
            ok = isinstance(value, str) and bool(value.strip())
        if not ok:
            problems.append(f"{name} (неверный тип/значение)")
    if problems:
        raise WorkerError(
            "telemetry_missing",
            "факты исполнения: отсутствуют или некорректны обязательные поля: "
            + ", ".join(problems)
            + "; ожидаемые значения не подставляются",
        )
    if telemetry["width"] <= 0 or telemetry["height"] <= 0:
        raise WorkerError("telemetry_missing", "факты исполнения: width/height должны быть > 0")
    if telemetry["steps"] <= 0:
        raise WorkerError("telemetry_missing", "факты исполнения: steps должен быть > 0")
    if telemetry["seed"] < 0:
        raise WorkerError("telemetry_missing", "факты исполнения: seed должен быть >= 0")
    if telemetry["cfg"] < 0:
        raise WorkerError("telemetry_missing", "факты исполнения: cfg должен быть >= 0")
    return dict(telemetry)


def build_telemetry(
    prompt: Mapping[str, Any] | None,
    parameter: Mapping[str, Any] | None = None,
    acceleration: Mapping[str, Any] | None = None,
    *,
    image: ImageInfo | None = None,
    gpu: tuple[float, str] | None = None,
    elapsed_seconds: float | None = None,
) -> dict[str, Any]:
    """Собрать телеметрию ответа из измеренных фактов.

    ``parameter``/``acceleration`` — снимок приложения; приняты для единой
    подписи и **не** используются для подстановки ожидаемых значений.

    Факты: `prompt` (исполненный граф), `image` (заголовок готового файла),
    `elapsed_seconds` (submit → history completed) и `gpu` (torch.cuda).
    Любой неизмеримый обязательный факт — видимая ``telemetry_missing``.
    """

    if not isinstance(prompt, Mapping) or not prompt:
        raise WorkerError(
            "telemetry_missing",
            "факты исполнения: prompt (исполненный граф) отсутствует или пуст",
        )
    if image is None:
        raise WorkerError(
            "telemetry_missing",
            "факты исполнения: image отсутствует — width/height не измерить "
            "(ожидаемые значения не подставляются)",
        )
    if not _is_int(getattr(image, "width", None)) or not _is_int(getattr(image, "height", None)):
        raise WorkerError("telemetry_missing", "факты исполнения: image.width/height — ожидались целые")
    if elapsed_seconds is None or not math.isfinite(float(elapsed_seconds)) or float(elapsed_seconds) <= 0:
        raise WorkerError(
            "telemetry_missing",
            f"факты исполнения: elapsed_seconds={elapsed_seconds!r} — замер submit→history "
            "completed не выполнен (seconds > 0)",
        )

    cc, gpu_class = gpu if gpu is not None else detect_gpu()
    if cc is None or not math.isfinite(float(cc)) or float(cc) <= 0 or not _str_value(gpu_class):
        raise WorkerError(
            "telemetry_missing",
            f"факты исполнения: gpu=({cc!r}, {gpu_class!r}) — ожидалась пара (cc>0, имя)",
        )

    facts = _sampler_facts(prompt)
    model_name = _model_name_from_graph(prompt)

    telemetry: dict[str, Any] = {
        "gpu_cc": float(cc),
        "gpu_class": str(gpu_class),
        "seconds": float(elapsed_seconds),
        "width": int(image.width),
        "height": int(image.height),
        "model_name": model_name,
        "seed": int(facts["seed"]),
        "steps": int(facts["steps"]),
        "cfg": float(facts["cfg"]),
        "sampler": str(facts["sampler"]),
        "scheduler": str(facts["scheduler"]),
    }
    return _check_telemetry(telemetry)


__all__ = ["build_telemetry", "detect_gpu"]
