"""Доставка ``input.assets`` в ComfyUI входной каталог (`contract §1`).

Приложение отдаёт presigned-URL'ы входных файлов (base image и маска
edit-графа, референсы t2i); воркер валидирует записи (`name` — basename без
путей, `url` — http(s), `sha256` — 64 hex, `kind` — image|mask), качает их
стримом в ``COMFYUI_INPUT_DIR`` под именем `name`, проверяет sha256 и размер и
делает ``os.replace`` (атомарная замена, повторный job идемпотентен). Граф
ссылается на эти имена в ``LoadImage``/``LoadImageMask``.

Ошибки — видимые (job FAILED, `contract §4`): ``asset_invalid``,
``asset_download_failed``, ``asset_hash_mismatch``, ``asset_too_large``.
Временный файл удаляется при любой ошибке; секретов здесь нет (URL приходит от
приложения и не логируется).
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

from .config import Config
from .errors import WorkerError

_LOG = logging.getLogger(__name__)

# `contract/input.schema.json`: не больше 8 файлов (base image + маска +
# запас на референсы); имя — basename без `/` и `..`.
MAX_ASSETS = 8
_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_KINDS = frozenset({"image", "mask"})
_KNOWN_FIELDS = frozenset({"name", "url", "sha256", "kind", "size"})
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class Asset:
    """Проверенная запись ``input.assets`` (после валидации, до скачивания)."""

    name: str
    url: str
    sha256: str | None = None
    kind: str | None = None
    size: int | None = None


def _invalid(detail: str) -> WorkerError:
    return WorkerError("asset_invalid", detail)


def _parse_asset(raw: Any, index: int) -> Asset:
    """Проверить одну запись ``input.assets[i]`` (типы/паттерны схемы)."""

    where = f"input.assets[{index}]"
    if not isinstance(raw, Mapping):
        raise _invalid(f"{where}: ожидался объект {{name,url,sha256?,kind?,size?}}")
    unknown = sorted(set(raw) - _KNOWN_FIELDS)
    if unknown:
        # `contract/input.schema.json`: additionalProperties=false — лишние поля
        # не угадываем, а показываем видимую ошибку.
        raise _invalid(f"{where}: неизвестные поля {unknown}")

    name = raw.get("name")
    if not isinstance(name, str) or not _NAME_RE.fullmatch(name):
        raise _invalid(
            f"{where}.name: ожидалось имя файла (basename, ^[A-Za-z0-9][A-Za-z0-9._-]{{0,127}}$), получено {name!r}"
        )

    url = raw.get("url")
    if not isinstance(url, str) or not url:
        raise _invalid(f"{where}.url: ожидалась строка http(s) URL")
    try:
        parts = urlsplit(url)
    except ValueError as exc:
        raise _invalid(f"{where}.url: невалидный URL {url!r}: {exc}") from exc
    if parts.scheme.lower() not in {"http", "https"} or not parts.netloc:
        raise _invalid(f"{where}.url: ожидался http(s) URL, получено {url!r}")

    sha256 = raw.get("sha256")
    if sha256 is not None and not (isinstance(sha256, str) and _SHA256_RE.fullmatch(sha256)):
        raise _invalid(f"{where}.sha256: ожидались 64 hex-символа, получено {sha256!r}")

    kind = raw.get("kind")
    if kind is not None and (not isinstance(kind, str) or kind not in _KINDS):
        raise _invalid(f"{where}.kind: ожидалось image|mask, получено {kind!r}")

    size = raw.get("size")
    if size is not None and (not isinstance(size, int) or isinstance(size, bool) or size <= 0):
        raise _invalid(f"{where}.size: ожидалось целое > 0, получено {size!r}")

    return Asset(name=name, url=url, sha256=sha256, kind=kind, size=size)


def validate_assets(assets: Any, config: Config) -> list[Asset]:
    """Разобрать и валидировать ``input.assets``; вернуть список ``Asset``.

    Ошибки структуры — ``asset_invalid``; заявленный размер больше
    ``ASSETS_MAX_MB``/``ASSETS_MAX_TOTAL_MB`` — ``asset_too_large`` ещё до
    скачивания. Повторный вызов идемпотентен (принимает и готовые ``Asset``).
    Дубликаты имён запрещены: два файла с одним именем — неоднозначная подмена.
    """

    if not isinstance(assets, Sequence) or isinstance(assets, (str, bytes, bytearray, Mapping)):
        raise _invalid("input.assets: ожидался массив объектов {name,url,...}")
    if len(assets) > MAX_ASSETS:
        raise _invalid(f"input.assets: не больше {MAX_ASSETS} файлов, получено {len(assets)}")

    parsed: list[Asset] = []
    declared_total = 0
    seen_names: set[str] = set()
    for index, item in enumerate(assets):
        asset = item if isinstance(item, Asset) else _parse_asset(item, index)
        if asset.name in seen_names:
            raise _invalid(f"input.assets: имя {asset.name!r} встречается больше одного раза")
        seen_names.add(asset.name)
        if asset.size is not None:
            if asset.size > config.assets_max_bytes:
                raise WorkerError(
                    "asset_too_large",
                    f"{asset.name}: заявленный размер {asset.size} байт больше "
                    f"ASSETS_MAX_MB={config.assets_max_mb:g} МБ ({config.assets_max_bytes} байт)",
                )
            declared_total += asset.size
        parsed.append(asset)

    if declared_total > config.assets_max_total_bytes:
        raise WorkerError(
            "asset_too_large",
            f"input.assets: заявленный суммарный размер {declared_total} байт больше "
            f"ASSETS_MAX_TOTAL_MB={config.assets_max_total_mb:g} МБ "
            f"({config.assets_max_total_bytes} байт)",
        )
    return parsed


async def _download_one(
    client: httpx.AsyncClient,
    asset: Asset,
    input_dir: Path,
    config: Config,
    *,
    total_before: int,
) -> tuple[Path, int]:
    """Скачать один asset в temp-файл → проверки → ``os.replace`` на имя."""

    final_path = input_dir / asset.name
    # Temp-файл в том же каталоге: os.replace атомарен только в пределах ФС.
    try:
        fd, temp_name = tempfile.mkstemp(prefix=f".{asset.name}.", suffix=".part", dir=input_dir)
    except OSError as exc:
        raise WorkerError(
            "asset_download_failed",
            f"{asset.name}: не удалось создать temp-файл в COMFYUI_INPUT_DIR={input_dir}: {exc}",
        ) from exc
    temp_path = Path(temp_name)
    hasher = hashlib.sha256()
    written = 0
    try:
        with os.fdopen(fd, "wb") as handle:
            async with client.stream("GET", asset.url) as response:
                if response.status_code >= 400:
                    raise WorkerError(
                        "asset_download_failed",
                        f"{asset.name}: GET {asset.url}: HTTP {response.status_code}",
                    )
                async for chunk in response.aiter_bytes():
                    if not chunk:
                        continue
                    written += len(chunk)
                    if written > config.assets_max_bytes:
                        raise WorkerError(
                            "asset_too_large",
                            f"{asset.name}: файл больше ASSETS_MAX_MB="
                            f"{config.assets_max_mb:g} МБ ({config.assets_max_bytes} байт)",
                        )
                    if total_before + written > config.assets_max_total_bytes:
                        raise WorkerError(
                            "asset_too_large",
                            f"{asset.name}: суммарный размер input.assets больше "
                            f"ASSETS_MAX_TOTAL_MB={config.assets_max_total_mb:g} МБ "
                            f"({config.assets_max_total_bytes} байт)",
                        )
                    hasher.update(chunk)
                    handle.write(chunk)
            handle.flush()
            os.fsync(handle.fileno())

        if written == 0:
            raise WorkerError("asset_download_failed", f"{asset.name}: скачан пустой файл")
        if asset.size is not None and written != asset.size:
            raise WorkerError(
                "asset_download_failed",
                f"{asset.name}: размер скачанного файла {written} байт не совпал с заявленным size={asset.size}",
            )
        if asset.sha256 is not None:
            actual = hasher.hexdigest()
            if actual != asset.sha256:
                raise WorkerError(
                    "asset_hash_mismatch",
                    f"{asset.name}: sha256 {actual} не совпал с ожидаемым {asset.sha256}",
                )
        os.replace(temp_path, final_path)
        _LOG.info("asset %s: %s байт → %s", asset.name, written, final_path)
        return final_path, written
    except WorkerError:
        temp_path.unlink(missing_ok=True)
        raise
    except httpx.HTTPError as exc:
        temp_path.unlink(missing_ok=True)
        raise WorkerError(
            "asset_download_failed",
            f"{asset.name}: GET {asset.url}: {type(exc).__name__}: {exc}",
        ) from exc
    except OSError as exc:
        temp_path.unlink(missing_ok=True)
        raise WorkerError(
            "asset_download_failed",
            f"{asset.name}: не удалось сохранить в COMFYUI_INPUT_DIR={input_dir}: {exc}",
        ) from exc
    except BaseException:
        # Отмена/неожиданная ошибка не должна оставлять .part в ComfyUI/input.
        temp_path.unlink(missing_ok=True)
        raise


async def download_assets(
    assets: Any,
    config: Config,
    http: httpx.AsyncClient | None = None,
) -> list[Path]:
    """Скачать ``input.assets`` в ``COMFYUI_INPUT_DIR``; вернуть пути файлов.

    ``http`` инъектируется для тестов (``httpx.MockTransport``): штатный клиент
    создаётся с ``follow_redirects=True`` и таймаутом ``ASSETS_TIMEOUT_SECONDS``.
    Порядок путей совпадает с порядком ``input.assets``. Пустой список — ``[]``.
    """

    parsed = validate_assets(assets, config)
    if not parsed:
        return []

    input_dir = Path(str(config.comfyui_input_dir))
    try:
        # ComfyUI/input может отсутствовать в свежем контейнере до первого старта.
        input_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise WorkerError(
            "asset_download_failed",
            f"COMFYUI_INPUT_DIR={input_dir}: не удалось создать каталог: {exc}",
        ) from exc

    own_client = http is None
    client = (
        http
        if http is not None
        else httpx.AsyncClient(
            follow_redirects=True,
            timeout=config.assets_timeout_seconds,
        )
    )
    try:
        paths: list[Path] = []
        total = 0
        for asset in parsed:
            path, written = await _download_one(client, asset, input_dir, config, total_before=total)
            total += written
            paths.append(path)
        return paths
    finally:
        if own_client:
            await client.aclose()


__all__ = ["MAX_ASSETS", "Asset", "download_assets", "validate_assets"]
