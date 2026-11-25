"""Размер готовой картинки из её байтов (факт, не снимок параметров).

Телеметрия контракта требует `width`/`height` фактического результата, а не
ожидаемых `parameter.width/height`. Поэтому размеры читаются из заголовка
готового файла, который вернул ComfyUI: PNG (IHDR), JPEG (SOF) или WebP
(VP8/VP8L/VP8X). Никаких зависимостей (PIL) не нужно.

Неизвестный/битый формат — видимая ошибка ``output_invalid``: пустая или
нечитаемая картинка не считается успешным результатом.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

from .errors import WorkerError

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)


@dataclass(frozen=True)
class ImageInfo:
    """Факт готовой картинки: формат и размеры в пикселях."""

    width: int
    height: int
    format: str
    content_type: str


def _fail(detail: str) -> WorkerError:
    return WorkerError("output_invalid", detail)


def _positive_size(width: int, height: int, fmt: str) -> ImageInfo:
    if width <= 0 or height <= 0:
        raise _fail(f"размер картинки {fmt}: {width}x{height} — ожидались положительные значения")
    content_type = {
        "png": "image/png",
        "jpeg": "image/jpeg",
        "webp": "image/webp",
    }.get(fmt, "application/octet-stream")
    return ImageInfo(width=int(width), height=int(height), format=fmt, content_type=content_type)


def _png_info(data: bytes) -> ImageInfo:
    # signature(8) + length(4) + "IHDR"(4) + width(4) + height(4)
    if len(data) < 24 or not data.startswith(_PNG_SIGNATURE) or data[12:16] != b"IHDR":
        raise _fail("PNG: нет сигнатуры/IHDR — файл не является PNG")
    width, height = struct.unpack(">II", data[16:24])
    return _positive_size(width, height, "png")


def _jpeg_info(data: bytes) -> ImageInfo:
    if len(data) < 4 or data[0:2] != b"\xff\xd8":
        raise _fail("JPEG: нет SOI — файл не является JPEG")
    offset = 2
    total = len(data)
    while offset + 4 <= total:
        if data[offset] != 0xFF:
            raise _fail(f"JPEG: ожидался маркер в смещении {offset}")
        marker = data[offset + 1]
        # Padding-байты 0xFF и standalone-маркеры без длины.
        if marker == 0xFF:
            offset += 1
            continue
        if marker in {0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            offset += 2
            continue
        if marker == 0xD9:  # EOI: до SOF не дошли
            break
        if offset + 4 > total:
            break
        (length,) = struct.unpack(">H", data[offset + 2 : offset + 4])
        if length < 2:
            raise _fail(f"JPEG: некорректная длина сегмента маркера 0x{marker:02x}")
        if marker in _JPEG_SOF_MARKERS:
            if offset + 9 > total:
                raise _fail("JPEG: обрезанный SOF-сегмент")
            height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
            return _positive_size(width, height, "jpeg")
        offset += 2 + length
    raise _fail("JPEG: не найден SOF-сегмент — размеры не измерить")


def _webp_info(data: bytes) -> ImageInfo:
    if len(data) < 16 or data[0:4] != b"RIFF" or data[8:12] != b"WEBP":
        raise _fail("WebP: нет RIFF/WEBP — файл не является WebP")
    chunk = data[12:16]
    if chunk == b"VP8X":
        if len(data) < 30:
            raise _fail("WebP VP8X: обрезанный заголовок")
        width = 1 + int.from_bytes(data[24:27], "little")
        height = 1 + int.from_bytes(data[27:30], "little")
        return _positive_size(width, height, "webp")
    if chunk == b"VP8L":
        if len(data) < 25 or data[20] != 0x2F:
            raise _fail("WebP VP8L: обрезанный или неизвестный заголовок")
        bits = int.from_bytes(data[21:25], "little")
        width = 1 + (bits & 0x3FFF)
        height = 1 + ((bits >> 14) & 0x3FFF)
        return _positive_size(width, height, "webp")
    if chunk == b"VP8 ":
        if len(data) < 30 or data[23:26] != b"\x9d\x01\x2a":
            raise _fail("WebP VP8: обрезанный или неизвестный заголовок")
        width, height = struct.unpack("<HH", data[26:30])
        return _positive_size(width & 0x3FFF, height & 0x3FFF, "webp")
    raise _fail(f"WebP: неизвестный chunk {chunk!r}")


def read_image_info(data: bytes) -> ImageInfo:
    """Прочитать ``ImageInfo`` из байтов результата; иначе ``output_invalid``."""

    if not data:
        raise WorkerError("output_empty", "ComfyUI вернул пустой файл: картинка не сформирована")
    if data.startswith(_PNG_SIGNATURE):
        return _png_info(data)
    if data.startswith(b"\xff\xd8"):
        return _jpeg_info(data)
    if data[0:4] == b"RIFF" and data[8:12] == b"WEBP":
        return _webp_info(data)
    raise _fail(
        "ComfyUI вернул файл неизвестного формата (поддержаны PNG/JPEG/WebP): "
        f"первые байты {data[:8]!r}"
    )


__all__ = ["ImageInfo", "read_image_info"]
