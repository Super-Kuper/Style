"""Схемы контракта: примеры валидны, нарушения видны, схемы — draft 2020-12."""

from __future__ import annotations

import copy

import pytest
from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError

from helpers import contract_dir, load_json


@pytest.fixture(scope="module")
def input_schema() -> dict:
    return load_json(contract_dir() / "input.schema.json")


@pytest.fixture(scope="module")
def output_schema() -> dict:
    return load_json(contract_dir() / "output.schema.json")


@pytest.fixture(scope="module")
def input_validator(input_schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(input_schema)
    return Draft202012Validator(input_schema)


@pytest.fixture(scope="module")
def output_validator(output_schema: dict) -> Draft202012Validator:
    Draft202012Validator.check_schema(output_schema)
    return Draft202012Validator(output_schema)


def _assert_invalid(validator: Draft202012Validator, payload: object) -> None:
    with pytest.raises(ValidationError):
        validator.validate(payload)


# --- Примеры -----------------------------------------------------------------


@pytest.mark.parametrize("name", ["request-edit.json", "request-i2i.json", "health.request.json"])
def test_input_examples_are_valid(input_validator: Draft202012Validator, name: str) -> None:
    input_validator.validate(load_json(contract_dir() / "examples" / name))


@pytest.mark.parametrize("name", ["response.json", "response-url.json"])
def test_output_examples_are_valid(output_validator: Draft202012Validator, name: str) -> None:
    output_validator.validate(load_json(contract_dir() / "examples" / name))


def test_health_response_shape() -> None:
    payload = load_json(contract_dir() / "examples" / "health.response.json")
    assert payload == {"status": "ok", "comfyui": "ready", "volume_mounted": True}


# --- Вход --------------------------------------------------------------------


def test_prompt_is_required_without_health(input_validator: Draft202012Validator) -> None:
    _assert_invalid(input_validator, {"input": {"parameter": {"mode": "generate"}}})
    _assert_invalid(input_validator, {"input": {}})


def test_health_allows_no_prompt(input_validator: Draft202012Validator) -> None:
    input_validator.validate({"input": {"action": "health"}})


def test_unknown_action_is_invalid(input_validator: Draft202012Validator) -> None:
    _assert_invalid(input_validator, {"input": {"action": "warmup"}})


@pytest.mark.parametrize("assets", [
    [{"name": "a.png", "url": "https://files.example/a.png", "extra": 1}],
    [{"name": "../a.png", "url": "https://files.example/a.png"}],
    [{"name": "a.png", "url": "ftp://files.example/a.png"}],
    [{"name": "a.png", "url": "https://files.example/a.png", "sha256": "ABC"}],
    [{"name": "a.png", "url": "https://files.example/a.png", "kind": "video"}],
    [{"name": "a.png", "url": "https://files.example/a.png", "size": 0}],
    [{"url": "https://files.example/a.png"}],
])
def test_invalid_assets(input_validator: Draft202012Validator, assets: list[dict]) -> None:
    graph = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x.safetensors"}}}
    _assert_invalid(input_validator, {"input": {"prompt": graph, "assets": assets}})


def test_edit_request_assets_are_the_graph_names(input_validator: Draft202012Validator) -> None:
    payload = load_json(contract_dir() / "examples" / "request-edit.json")
    names = {asset["name"] for asset in payload["input"]["assets"]}
    graph = payload["input"]["prompt"]
    referenced = {
        node["inputs"].get("image")
        for node in graph.values()
        if node["class_type"] in {"LoadImage", "LoadImageMask", "LoadImageOutput"}
    }
    assert names == referenced == {"base.png", "mask.png"}


def test_i2i_request_uses_style_reference_as_init_image() -> None:
    """i2i: референс стиля — входной ``LoadImage``, граф остаётся API-графом."""
    payload = load_json(contract_dir() / "examples" / "request-i2i.json")
    names = {asset["name"] for asset in payload["input"]["assets"]}
    graph = payload["input"]["prompt"]
    loads = [
        node["inputs"].get("image")
        for node in graph.values()
        if node["class_type"] == "LoadImage"
    ]
    assert names == {"style_reference.png"}
    assert loads == ["style_reference.png"]
    classes = {node["class_type"] for node in graph.values()}
    assert {"LoadImage", "ImageScale", "VAEEncode", "KSampler", "SaveImage"} <= classes


# --- Выход -------------------------------------------------------------------


VALID_TELEMETRY = {
    "gpu_cc": 8.6,
    "gpu_class": "RTX A6000",
    "seconds": 7.4,
    "width": 768,
    "height": 1344,
    "model_name": "flux-2-klein-4b.safetensors",
    "seed": 20261113,
    "steps": 4,
    "cfg": 1.0,
    "sampler": "euler",
    "scheduler": "simple",
}


def _output(**overrides: object) -> dict:
    payload = {"image": "aGVsbG8=", "telemetry": copy.deepcopy(VALID_TELEMETRY)}
    payload.update(overrides)
    return payload


def test_valid_inline_output(output_validator: Draft202012Validator) -> None:
    output_validator.validate(_output())


def test_valid_url_output(output_validator: Draft202012Validator) -> None:
    payload = {"image_url": "https://s3.example/out.png?X-Amz-Signature=x", "telemetry": copy.deepcopy(VALID_TELEMETRY)}
    output_validator.validate(payload)


def test_image_and_image_url_are_mutually_exclusive(output_validator: Draft202012Validator) -> None:
    _assert_invalid(output_validator, _output(image_url="https://s3.example/out.png"))
    _assert_invalid(output_validator, {"telemetry": copy.deepcopy(VALID_TELEMETRY)})


@pytest.mark.parametrize("missing", sorted(VALID_TELEMETRY))
def test_telemetry_required_fields(output_validator: Draft202012Validator, missing: str) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    telemetry.pop(missing)
    _assert_invalid(output_validator, {"image": "aGVsbG8=", "telemetry": telemetry})


@pytest.mark.parametrize("field,value", [
    ("gpu_cc", -1),
    ("seconds", 0),
    ("width", 0),
    ("height", -3),
    ("steps", 0),
    ("cfg", -1),
    ("seed", -1),
    ("model_name", ""),
    ("sampler", ""),
])
def test_telemetry_bad_values(output_validator: Draft202012Validator, field: str, value: object) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    telemetry[field] = value
    _assert_invalid(output_validator, {"image": "aGVsbG8=", "telemetry": telemetry})


def test_text_detected_is_optional_and_omitted(output_validator: Draft202012Validator) -> None:
    telemetry = copy.deepcopy(VALID_TELEMETRY)
    assert "text_detected" not in telemetry
    output_validator.validate({"image": "aGVsbG8=", "telemetry": telemetry})
    telemetry["text_detected"] = False
    output_validator.validate({"image": "aGVsbG8=", "telemetry": telemetry})
