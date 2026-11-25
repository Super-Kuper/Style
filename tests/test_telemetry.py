"""Телеметрия: факты из графа картинки, без подстановки ожиданий."""

from __future__ import annotations

import copy

import pytest

from helpers import custom_advanced_graph, png_bytes, t2i_graph
from image_worker.errors import WorkerError
from image_worker.imaging import ImageInfo, read_image_info
from image_worker.telemetry import build_telemetry, detect_gpu

IMAGE = ImageInfo(width=768, height=1344, format="png", content_type="image/png")
GPU = (8.6, "RTX A6000")


def test_ksampler_facts() -> None:
    telemetry = build_telemetry(t2i_graph(seed=7, steps=6, cfg=2.5, sampler="dpmpp_2m", scheduler="karras"),
                                image=IMAGE, gpu=GPU, elapsed_seconds=3.5)
    assert telemetry == {
        "gpu_cc": 8.6,
        "gpu_class": "RTX A6000",
        "seconds": 3.5,
        "width": 768,
        "height": 1344,
        "model_name": "flux-2-klein-4b.safetensors",
        "seed": 7,
        "steps": 6,
        "cfg": 2.5,
        "sampler": "dpmpp_2m",
        "scheduler": "karras",
    }


def test_sampler_custom_advanced_facts() -> None:
    telemetry = build_telemetry(custom_advanced_graph(seed=42, steps=8, cfg=3.5),
                                image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert telemetry["seed"] == 42
    assert telemetry["steps"] == 8
    assert telemetry["cfg"] == 3.5
    assert telemetry["sampler"] == "euler"
    assert telemetry["scheduler"] == "simple"
    assert telemetry["model_name"] == "flux-2-klein-4b.safetensors"


def test_sampler_custom_direct_noise_seed() -> None:
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors"}},
        "2": {"class_type": "KSamplerSelect", "inputs": {"sampler_name": "euler"}},
        "3": {"class_type": "BasicScheduler", "inputs": {"model": ["1", 0], "scheduler": "simple", "steps": 12, "denoise": 1.0}},
        "4": {
            "class_type": "SamplerCustom",
            "inputs": {"model": ["1", 0], "add_noise": True, "noise_seed": 11, "cfg": 2.0,
                       "sampler": ["2", 0], "sigmas": ["3", 0], "latent_image": ["5", 0]},
        },
        "5": {"class_type": "EmptySD3LatentImage", "inputs": {"width": 768, "height": 1344, "batch_size": 1}},
    }
    telemetry = build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert telemetry["seed"] == 11
    assert telemetry["steps"] == 12
    assert telemetry["cfg"] == 2.0
    assert telemetry["sampler"] == "euler"
    assert telemetry["scheduler"] == "simple"


def test_dimensions_are_facts_not_parameter() -> None:
    info = read_image_info(png_bytes(512, 640))
    telemetry = build_telemetry(
        t2i_graph(),
        {"width": 768, "height": 1344, "steps": 999, "seed": 1},
        {"gpu_class": "L40S"},
        image=info,
        gpu=(8.9, "L40S"),
        elapsed_seconds=2.0,
    )
    assert (telemetry["width"], telemetry["height"]) == (512, 640)
    assert telemetry["steps"] == 4
    assert telemetry["seed"] == 20261113
    assert telemetry["gpu_class"] == "L40S"


def test_text_detected_is_not_invented() -> None:
    telemetry = build_telemetry(t2i_graph(), image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert "text_detected" not in telemetry


def test_missing_sampler_is_visible() -> None:
    graph = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors"}},
        "2": {"class_type": "SaveImage", "inputs": {"images": ["1", 0]}},
    }
    with pytest.raises(WorkerError) as error:
        build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert error.value.code == "telemetry_missing"
    assert "сэмплер" in error.value.message


def test_missing_loader_is_visible() -> None:
    graph = {"1": {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 4, "cfg": 1.0,
                                                       "sampler_name": "euler", "scheduler": "simple"}}}
    with pytest.raises(WorkerError) as error:
        build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert error.value.code == "telemetry_missing"
    assert "model_name" in error.value.message


def test_ambiguous_steps_between_samplers() -> None:
    graph = t2i_graph(steps=4)
    graph["20"] = copy.deepcopy(graph["7"])
    graph["20"]["inputs"]["steps"] = 20
    with pytest.raises(WorkerError) as error:
        build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert error.value.code == "telemetry_ambiguous"
    assert "steps" in error.value.message


def test_ambiguous_model_between_loaders() -> None:
    graph = t2i_graph()
    graph["20"] = {"class_type": "UNETLoader", "inputs": {"unet_name": "other.safetensors"}}
    with pytest.raises(WorkerError) as error:
        build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert error.value.code == "telemetry_ambiguous"
    assert "model_name" in error.value.message


def test_checkpoint_loader_is_accepted() -> None:
    graph = {"1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "model.safetensors"}},
             "2": {"class_type": "KSampler", "inputs": {"seed": 3, "steps": 5, "cfg": 3.0,
                                                        "sampler_name": "euler", "scheduler": "normal"}}}
    telemetry = build_telemetry(graph, image=IMAGE, gpu=GPU, elapsed_seconds=1.0)
    assert telemetry["model_name"] == "model.safetensors"
    assert telemetry["sampler"] == "euler"
    assert telemetry["scheduler"] == "normal"


def test_image_is_required() -> None:
    with pytest.raises(WorkerError) as error:
        build_telemetry(t2i_graph(), gpu=GPU, elapsed_seconds=1.0)
    assert error.value.code == "telemetry_missing"


def test_positive_seconds_required() -> None:
    with pytest.raises(WorkerError) as error:
        build_telemetry(t2i_graph(), image=IMAGE, gpu=GPU, elapsed_seconds=0)
    assert error.value.code == "telemetry_missing"


def test_gpu_override_pair_and_single_value() -> None:
    assert detect_gpu({"IMAGE_GPU_CC": "8.6", "IMAGE_GPU_CLASS": "RTX A6000"}) == (8.6, "RTX A6000")
    with pytest.raises(WorkerError) as error:
        detect_gpu({"IMAGE_GPU_CC": "8.6"})
    assert error.value.code == "config_invalid"
