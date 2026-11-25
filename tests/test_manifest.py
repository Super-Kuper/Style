"""models-manifest.txt: только commit-pinned URL, sha256, имена совпадают с графами."""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "models-manifest.txt"

PIN_RE = re.compile(r"/resolve/([0-9a-f]{40})/")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def _rows() -> list[list[str]]:
    rows = []
    for line in MANIFEST.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        rows.append(line.split("\t"))
    return rows


def test_all_rows_have_four_columns() -> None:
    rows = _rows()
    assert rows, "манифест пуст"
    for row in rows:
        assert len(row) == 4, f"ожидался <subdir>\\t<filename>\\t<URL>\\t<sha256>: {row!r}"


def test_urls_are_commit_pinned() -> None:
    for subdir, name, url, sha in _rows():
        assert "/resolve/main/" not in url and "/resolve/master/" not in url, (subdir, name, url)
        match = PIN_RE.search(url)
        assert match is not None, f"{name}: URL не запинен на commit: {url}"
        assert name in url, f"{name}: имя файла не совпадает с URL {url}"
        assert SHA_RE.fullmatch(sha), f"{name}: sha256 не 64 hex"


def test_no_duplicate_names() -> None:
    rows = _rows()
    keys = [(row[0], row[1]) for row in rows]
    assert len(keys) == len(set(keys))


def test_expected_layout_and_models() -> None:
    rows = _rows()
    by_name = {row[1]: row[0] for row in rows}
    # Refs profile: only the Qwen-Image-Edit 2511 compose/identity/clean trio.
    assert len(rows) == 3
    assert by_name["qwen_image_edit_2511_fp8mixed.safetensors"] == "diffusion_models"
    assert by_name["qwen_2.5_vl_7b_fp8_scaled.safetensors"] == "text_encoders"
    assert by_name["qwen_image_vae.safetensors"] == "vae"


def test_contract_example_graphs_reference_manifest_files() -> None:
    """Графы каталога ссылаются на файлы, которые populate-volume.sh кладёт на том."""

    import json

    from helpers import contract_dir

    volume_files = {row[1] for row in _rows()}
    for example in ("request-edit.json", "request-i2i.json"):
        payload = json.loads((contract_dir() / "examples" / example).read_text(encoding="utf-8"))
        graph = payload["input"]["prompt"]
        for node in graph.values():
            inputs = node.get("inputs", {})
            for key in ("unet_name", "ckpt_name", "clip_name", "vae_name"):
                value = inputs.get(key)
                if isinstance(value, str):
                    assert value in volume_files, f"{example}: {value} нет в models-manifest.txt"
