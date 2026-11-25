"""Размеры картинки — факт из заголовка файла, а не снимок parameter."""

from __future__ import annotations

import struct

import pytest

from helpers import jpeg_bytes, png_bytes
from image_worker.errors import WorkerError
from image_worker.imaging import read_image_info


def test_png_dimensions() -> None:
    info = read_image_info(png_bytes(768, 1344))
    assert (info.width, info.height) == (768, 1344)
    assert info.format == "png"
    assert info.content_type == "image/png"


def test_jpeg_dimensions() -> None:
    info = read_image_info(jpeg_bytes(640, 480))
    assert (info.width, info.height) == (640, 480)
    assert info.format == "jpeg"
    assert info.content_type == "image/jpeg"


def test_webp_vp8x_dimensions() -> None:
    payload = b"\x00" * 4 + (300 - 1).to_bytes(3, "little") + (200 - 1).to_bytes(3, "little")
    data = b"RIFF" + struct.pack("<I", 4 + 8 + len(payload)) + b"WEBP" + b"VP8X" + struct.pack("<I", len(payload)) + payload
    info = read_image_info(data)
    assert (info.width, info.height) == (300, 200)
    assert info.format == "webp"


def test_empty_file_is_output_empty() -> None:
    with pytest.raises(WorkerError) as error:
        read_image_info(b"")
    assert error.value.code == "output_empty"


def test_unknown_format_is_output_invalid() -> None:
    with pytest.raises(WorkerError) as error:
        read_image_info(b"GIF89a\x01\x00\x01\x00")
    assert error.value.code == "output_invalid"


def test_truncated_png_is_output_invalid() -> None:
    data = png_bytes(10, 10)[:18]
    with pytest.raises(WorkerError) as error:
        read_image_info(data)
    assert error.value.code == "output_invalid"


def test_zero_size_png_is_output_invalid() -> None:
    data = png_bytes(0, 0)
    with pytest.raises(WorkerError) as error:
        read_image_info(data)
    assert error.value.code == "output_invalid"
