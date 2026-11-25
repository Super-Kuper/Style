"""Скрипты образа: синтаксис, обязательные COPY в Dockerfile, populate-volume.sh.

Проверки не требуют Docker/GPU/сети: ``bash -n`` для синтаксиса и прогоны
populate-volume.sh на временном «томе» с локальным манифестом.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BASH = shutil.which("bash")

pytestmark = pytest.mark.skipif(BASH is None, reason="нужен bash для проверки shell-скриптов")


def _run(args: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, text=True, cwd=ROOT, **kwargs)


@pytest.mark.parametrize("script", [
    "populate-volume.sh",
    "docker/entrypoint.sh",
    "docker/install-comfy-packages.sh",
    "scripts/wait-comfy.sh",
])
def test_shell_syntax(script: str) -> None:
    result = _run([BASH, "-n", script])
    assert result.returncode == 0, result.stderr


def _manifest(tmp_path: Path, rows: list[str]) -> Path:
    path = tmp_path / "models-manifest.txt"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _populate(tmp_path: Path, manifest: Path, extra_env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    (tmp_path / "volume").mkdir(exist_ok=True)
    env = {
        **os.environ,
        "VOL_ROOT": str(tmp_path / "volume"),
        "MANIFEST": str(manifest),
    }
    env.pop("HF_TOKEN", None)
    if extra_env:
        env.update(extra_env)
    return _run([BASH, "populate-volume.sh"], env=env)


def _pinned_url(name: str, commit: str = "a" * 40) -> str:
    return f"https://huggingface.co/org/repo/resolve/{commit}/split_files/diffusion_models/{name}"


def test_populate_rejects_main(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\thttps://huggingface.co/org/repo/resolve/main/x.safetensors\t{'b' * 64}",
    ])
    result = _populate(tmp_path, manifest)
    assert result.returncode != 0
    assert "main" in result.stderr.lower()


def test_populate_rejects_unresolved_placeholder(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [
        "diffusion_models\tx.safetensors\t"
        "https://huggingface.co/org/repo/resolve/${MODEL_REV}/x.safetensors\t" + "b" * 64,
    ])
    result = _populate(tmp_path, manifest)
    assert result.returncode != 0
    assert "MODEL_REV" in result.stderr
    assert "не заданы переменные" in result.stderr


def test_populate_skips_existing_verified_file(tmp_path: Path) -> None:
    content = b"already-downloaded-model"
    sha = hashlib.sha256(content).hexdigest()
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\t{_pinned_url('x.safetensors')}\t{sha}",
    ])
    target = tmp_path / "volume" / "models" / "diffusion_models" / "x.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)

    result = _populate(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    assert "[populate]" in result.stdout
    assert "есть diffusion_models/x.safetensors (sha256 ok)" in result.stdout
    assert target.read_bytes() == content
    # Layout тома создаётся заранее.
    assert (tmp_path / "volume" / "custom_nodes").is_dir()
    assert (tmp_path / "volume" / "encoder-cache").is_dir()
    assert (tmp_path / "volume" / "models" / ".populate-state.tsv").is_file()


def test_populate_reports_manifest_format_error(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\t{_pinned_url('x.safetensors')}\t{'b' * 64}\tEXTRA",
    ])
    result = _populate(tmp_path, manifest)
    assert result.returncode != 0
    assert "лишняя колонка" in result.stderr


def test_populate_downloads_verifies_and_renames_atomically(tmp_path: Path) -> None:
    # Локальный file:// источник с commit-пином в пути: проверяем полный путь
    # скачивания (curl), sha256, атомарный mv и запись состояния тома.
    source_dir = tmp_path / "source" / "resolve" / ("c" * 40)
    source_dir.mkdir(parents=True)
    content = b"fresh-model-bytes"
    (source_dir / "x.safetensors").write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    url = f"file://{(source_dir / 'x.safetensors').as_posix()}"
    manifest = _manifest(tmp_path, [f"diffusion_models\tx.safetensors\t{url}\t{sha}"])

    result = _populate(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    target = tmp_path / "volume" / "models" / "diffusion_models" / "x.safetensors"
    assert target.read_bytes() == content
    assert not (target.parent / "x.safetensors.tmp").exists()
    state = (tmp_path / "volume" / "models" / ".populate-state.tsv").read_text(encoding="utf-8")
    assert f"diffusion_models/x.safetensors\t{sha}" in state


def test_populate_expands_pinned_revision_placeholder(tmp_path: Path) -> None:
    content = b"model-bytes"
    sha = hashlib.sha256(content).hexdigest()
    manifest = _manifest(tmp_path, [
        "diffusion_models\tx.safetensors\t"
        "https://huggingface.co/org/repo/resolve/${MODEL_REV}/x.safetensors\t" + sha,
    ])
    target = tmp_path / "volume" / "models" / "diffusion_models" / "x.safetensors"
    target.parent.mkdir(parents=True)
    target.write_bytes(content)

    result = _populate(tmp_path, manifest, {"MODEL_REV": "d" * 40})
    assert result.returncode == 0, result.stderr
    assert "есть diffusion_models/x.safetensors (sha256 ok)" in result.stdout


def test_dockerfile_copy_sources_exist() -> None:
    text = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    copies = re.findall(r"^COPY\s+(\S+)\s+\S+", text, flags=re.MULTILINE)
    assert copies, "в Dockerfile нет COPY"
    for source in copies:
        assert (ROOT / source).exists(), f"COPY {source}: источника нет в репозитории"


def test_runpod_metadata_is_valid_json() -> None:
    hub = json.loads((ROOT / ".runpod" / "hub.json").read_text(encoding="utf-8"))
    tests = json.loads((ROOT / ".runpod" / "tests.json").read_text(encoding="utf-8"))
    assert hub["type"] == "serverless"
    assert hub["config"]["gpuIds"]
    assert hub["config"]["env"]
    assert tests["tests"][0]["input"] == {"action": "health"}
    assert "VOLUME_AUTOPOPULATE" in {item["key"] for item in tests["config"]["env"]}


def test_hub_gpu_pools_are_48gb_with_full_card_fallback() -> None:
    """Профиль refs: 48 GB остаются первыми, запас — полная A100 80 GB.

    MIG-срез (`BLACKWELL_96` / PRO 6000 MIG 48GB) в этом профиле не нужен:
    latency первых кадров важнее доступности. `BLACKWELL_96` поэтому не
    добавляется; `AMPERE_80` — полная карта, а не срез, и только фолбэк.
    """

    hub = json.loads((ROOT / ".runpod" / "hub.json").read_text(encoding="utf-8"))
    pools = [item.strip() for item in hub["config"]["gpuIds"].split(",")]
    assert pools[0] in {"AMPERE_48", "ADA_48_PRO"}, pools
    assert {"AMPERE_48", "ADA_48_PRO"} <= set(pools), pools
    assert "AMPERE_80" in pools, pools
    assert "BLACKWELL_96" not in pools, pools

    tests = json.loads((ROOT / ".runpod" / "tests.json").read_text(encoding="utf-8"))
    assert tests["config"]["gpuTypeId"].startswith("NVIDIA ")


def test_version_is_aligned_across_metadata() -> None:
    """Версия в hub.json, pyproject, Dockerfile, __init__ и README совпадает."""

    version = json.loads((ROOT / ".runpod" / "hub.json").read_text(encoding="utf-8"))["version"].lstrip("v")
    assert re.search(rf'^version = "{re.escape(version)}"$', (ROOT / "pyproject.toml").read_text(encoding="utf-8"), re.M)
    assert re.search(
        rf"^ARG IMAGE_WORKER_VERSION={re.escape(version)}$",
        (ROOT / "Dockerfile").read_text(encoding="utf-8"),
        re.M,
    )
    assert f'__version__ = "{version}"' in (ROOT / "src" / "image_worker" / "__init__.py").read_text(encoding="utf-8")
    assert f"image-refs-serverless:{version}" in (ROOT / "README.md").read_text(encoding="utf-8")


# --- populate-volume.sh: режимы сверки VERIFY=fast|full|off --------------------
#
# Мотивация: полная сверка 66.7 GB на каждом cold start занимает ~4.5 минуты
# (log run_pod_logs.txt 08:29:05 → 08:33:39). fast-path доверяет паре
# (размер, mtime) из состояния тома, а полная сверка уходит в фон после старта
# handler'а — поэтому тесты фиксируют и выигрыш, и условия, при которых
# fast-path обязан прочитать файл целиком.


def _state_file(tmp_path: Path) -> Path:
    return tmp_path / "volume" / "models" / ".populate-state.tsv"


def _state_rows(tmp_path: Path) -> dict[str, list[str]]:
    rows: dict[str, list[str]] = {}
    for line in _state_file(tmp_path).read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        parts = line.split("\t")
        rows[parts[0]] = parts
    return rows


def _target(tmp_path: Path) -> Path:
    return tmp_path / "volume" / "models" / "diffusion_models" / "x.safetensors"


def _download_manifest(tmp_path: Path, content: bytes) -> Path:
    """Манифест с локальным file:// источником и pinned sha256 содержимого."""
    directory = tmp_path / "source" / "resolve" / ("c" * 40)
    directory.mkdir(parents=True, exist_ok=True)
    source = directory / "x.safetensors"
    source.write_bytes(content)
    sha = hashlib.sha256(content).hexdigest()
    return _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\tfile://{source.as_posix()}\t{sha}",
    ])


def test_second_boot_takes_fast_path_without_hashing(tmp_path: Path) -> None:
    content = b"fast-path-model"
    manifest = _download_manifest(tmp_path, content)
    first = _populate(tmp_path, manifest)
    assert first.returncode == 0, first.stderr

    second = _populate(tmp_path, manifest)  # VERIFY по умолчанию — fast
    assert second.returncode == 0, second.stderr
    assert "метаданные совпадают" in second.stdout
    assert "sha не читаю" in second.stdout
    assert "проверяю sha256" not in second.stdout
    assert "(sha256 ok)" not in second.stdout


def test_full_mode_reads_sha_even_when_metadata_matches(tmp_path: Path) -> None:
    content = b"full-mode-model"
    manifest = _download_manifest(tmp_path, content)
    assert _populate(tmp_path, manifest).returncode == 0

    second = _populate(tmp_path, manifest, {"VERIFY": "full"})
    assert second.returncode == 0, second.stderr
    assert "проверяю sha256" in second.stdout
    assert "(sha256 ok)" in second.stdout
    assert "метаданные совпадают" not in second.stdout


def test_state_records_sha_size_and_mtime(tmp_path: Path) -> None:
    content = b"state-shape"
    manifest = _download_manifest(tmp_path, content)
    assert _populate(tmp_path, manifest).returncode == 0

    target = _target(tmp_path)
    row = _state_rows(tmp_path)["diffusion_models/x.safetensors"]
    assert len(row) == 4, row
    assert row[1] == hashlib.sha256(content).hexdigest()
    assert row[2] == str(len(content))
    assert row[3] == str(int(target.stat().st_mtime))


def test_legacy_state_row_without_mtime_is_hashed_once_and_upgraded(tmp_path: Path) -> None:
    content = b"legacy-state-model"
    sha = hashlib.sha256(content).hexdigest()
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(content)
    _state_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    _state_file(tmp_path).write_text(
        "# subdir/filename\tsha256\tsize\n"
        f"diffusion_models/x.safetensors\t{sha}\t{len(content)}\n",
        encoding="utf-8",
    )
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\t{_pinned_url('x.safetensors')}\t{sha}",
    ])

    # Строка старого формата: mtime неизвестен, значит fast-path не применим.
    first = _populate(tmp_path, manifest)
    assert first.returncode == 0, first.stderr
    assert "проверяю sha256" in first.stdout
    assert "(sha256 ok)" in first.stdout
    assert len(_state_rows(tmp_path)["diffusion_models/x.safetensors"]) == 4

    # После апгрейда строки — уже fast-path.
    second = _populate(tmp_path, manifest)
    assert "метаданные совпадают" in second.stdout


def test_mtime_change_forces_rehash(tmp_path: Path) -> None:
    content = b"mtime-changed-model"
    manifest = _download_manifest(tmp_path, content)
    assert _populate(tmp_path, manifest).returncode == 0

    target = _target(tmp_path)
    old = target.stat().st_mtime
    os.utime(target, (old - 120, old - 120))

    second = _populate(tmp_path, manifest)
    assert second.returncode == 0, second.stderr
    assert "проверяю sha256" in second.stdout
    assert "(sha256 ok)" in second.stdout


def test_size_change_forces_rehash(tmp_path: Path) -> None:
    content = b"size-changed-model"
    manifest = _download_manifest(tmp_path, content)
    assert _populate(tmp_path, manifest).returncode == 0

    target = _target(tmp_path)
    target.write_bytes(b"truncated")
    os.utime(target, None)  # mtime мог остаться в той же секунде — размер уже другой

    second = _populate(tmp_path, manifest)
    assert second.returncode == 0, second.stderr
    assert "проверяю sha256" in second.stdout
    assert "отвожу в карантин" in second.stderr
    assert target.read_bytes() == content


def test_corrupt_file_is_quarantined_then_replaced(tmp_path: Path) -> None:
    content = b"correct-model-bytes"
    manifest = _download_manifest(tmp_path, content)
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"silently-corrupted-bytes")

    result = _populate(tmp_path, manifest)
    assert result.returncode == 0, result.stderr
    assert "отвожу в карантин" in result.stderr
    assert target.read_bytes() == content
    # Карантин удаляется только после того, как проверенная замена встала на место.
    assert not list(target.parent.glob("x.safetensors.corrupt-*"))
    assert _state_rows(tmp_path)["diffusion_models/x.safetensors"][1] == hashlib.sha256(content).hexdigest()


def test_quarantine_is_kept_when_redownload_fails(tmp_path: Path) -> None:
    sha = hashlib.sha256(b"expected-model").hexdigest()
    missing = tmp_path / "source" / "resolve" / ("c" * 40) / "x.safetensors"  # файла нет
    manifest = _manifest(tmp_path, [f"diffusion_models\tx.safetensors\tfile://{missing.as_posix()}\t{sha}"])
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"corrupted-and-unreplaceable")

    result = _populate(tmp_path, manifest)
    assert result.returncode != 0
    assert "отвожу в карантин" in result.stderr
    # Код выхода curl должен попадать в лог как есть (регрессия: было «curl exit 0»).
    assert re.search(r"curl exit [1-9]\d*", result.stderr), result.stderr
    # Модель исчезла из layout'а (job упадёт видимо), но байты сохранены для разбора.
    assert not target.exists()
    quarantined = list(target.parent.glob("x.safetensors.corrupt-*"))
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == b"corrupted-and-unreplaceable"


def test_verify_off_skips_hashing_and_keeps_file(tmp_path: Path) -> None:
    sha = hashlib.sha256(b"expected-model").hexdigest()
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\t{_pinned_url('x.safetensors')}\t{sha}",
    ])
    target = _target(tmp_path)
    target.parent.mkdir(parents=True)
    target.write_bytes(b"whatever-bytes")

    result = _populate(tmp_path, manifest, {"VERIFY": "off"})
    assert result.returncode == 0, result.stderr
    assert "VERIFY=off" in result.stdout
    assert "проверяю sha256" not in result.stdout
    assert target.read_bytes() == b"whatever-bytes"


def test_second_worker_does_not_redownload_after_file_appeared(tmp_path: Path) -> None:
    """Второй воркер на том же томе не качает файл повторно.

    Сценарий боевого холодного старта: два воркера стартуют одновременно, файла
    ещё нет. Второй ждёт flock, видит уже скачанный файл с совпавшим состоянием
    тома и пропускает докачку (иначе 20 GB качаются дважды).
    """
    content = b"appeared-while-waiting"
    sha = hashlib.sha256(content).hexdigest()
    directory = tmp_path / "source" / "resolve" / ("c" * 40)
    directory.mkdir(parents=True)
    source = directory / "x.safetensors"
    source.write_bytes(content)
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\tfile://{source.as_posix()}\t{sha}",
    ])
    volume = tmp_path / "volume"
    volume.mkdir()
    _state_file(tmp_path).parent.mkdir(parents=True, exist_ok=True)

    with (volume / ".populate.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        proc = subprocess.Popen(
            [BASH, "populate-volume.sh"],
            cwd=ROOT,
            env={**os.environ, "VOL_ROOT": str(volume), "MANIFEST": str(manifest), "VERIFY": "fast"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            time.sleep(0.4)  # воркер успевает дойти до проверки файла/блокировки
            target = _target(tmp_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            _state_file(tmp_path).write_text(
                "# subdir/filename\tsha256\tsize\tmtime\n"
                f"diffusion_models/x.safetensors\t{sha}\t{len(content)}\t{int(target.stat().st_mtime)}\n",
                encoding="utf-8",
            )
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)
        out, err = proc.communicate(timeout=60)

    assert proc.returncode == 0, err
    assert "качаю" not in out
    assert "метаданные совпадают" in out or "появился, пока ждал блокировку" in out


def test_unknown_verify_mode_is_rejected(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path, [
        f"diffusion_models\tx.safetensors\t{_pinned_url('x.safetensors')}\t{'b' * 64}",
    ])
    result = _populate(tmp_path, manifest, {"VERIFY": "sometimes"})
    assert result.returncode != 0
    assert "VERIFY=sometimes" in result.stderr


def test_entrypoint_moves_full_verify_off_the_boot_path() -> None:
    text = (ROOT / "docker" / "entrypoint.sh").read_text(encoding="utf-8")
    assert 'VOLUME_VERIFY="${VOLUME_VERIFY:-fast}"' in text
    assert 'VOLUME_BACKGROUND_VERIFY="${VOLUME_BACKGROUND_VERIFY:-true}"' in text

    main_body = text.split("main() {", 1)[1]
    assert "_start_background_verify" in main_body
    assert main_body.index("_start_background_verify") < main_body.index("_start_handler")

    cleanup_body = text.split("_cleanup() {", 1)[1].split("trap _cleanup", 1)[0]
    assert "_stop_background_verify" in cleanup_body
    # Фоновая сверка — именно full: fast-режим не читает файл и не поймал бы порчу.
    assert "VERIFY=full POPULATE_LABEL=populate-verify" in text


_VERIFY_FUNCS = r"""
set -uo pipefail
log()  { printf '[entrypoint] %s\n' "$*"; }
warn() { printf '[entrypoint] WARN: %s\n' "$*" >&2; }
eval "$(awk '/^# --- 5b\./{f=1} /^# --- 6\./{f=0} f' "$1")"
VOLUME_AUTOPOPULATE=true
VOLUME_VERIFY=fast
VOLUME_ROOT="$2"
POPULATE_SH="$3"
VOLUME_BACKGROUND_VERIFY="$4"
MANIFEST=/dev/null
HF_TOKEN=""
VERIFY_PID=""
_start_background_verify
if [[ -z "${VERIFY_PID}" ]]; then
    echo NO_PID
    exit 0
fi
sleep 0.5  # даём фоновому процессу отметиться, прежде чем гасить
if kill -0 "${VERIFY_PID}" 2>/dev/null; then echo "STARTED"; else echo DIED; fi
_stop_background_verify
if kill -0 "${VERIFY_PID}" 2>/dev/null; then echo STILL_ALIVE; else echo STOPPED; fi
"""


def _sleeper(tmp_path: Path, marker: Path) -> Path:
    """Фоновая «сверка»: пишет полученный env в файл и висит, как долгий sha256."""
    script = tmp_path / "sleeper.sh"
    script.write_text(
        "#!/usr/bin/env bash\n"
        f"printf '%s|%s\\n' \"${{VERIFY:-}}\" \"${{POPULATE_LABEL:-}}\" > {shlex.quote(str(marker))}\n"
        "sleep 60\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    return script


def test_background_verify_passes_full_mode_and_is_stoppable(tmp_path: Path) -> None:
    marker = tmp_path / "seen.txt"
    sleeper = _sleeper(tmp_path, marker)
    volume = tmp_path / "volume"
    volume.mkdir()
    harness = tmp_path / "harness.sh"
    harness.write_text(_VERIFY_FUNCS, encoding="utf-8")

    result = _run([BASH, str(harness), str(ROOT / "docker" / "entrypoint.sh"), str(volume), str(sleeper), "true"])
    assert result.returncode == 0, result.stderr
    assert "STARTED" in result.stdout
    assert "STOPPED" in result.stdout
    assert "фоновая сверка sha256" in result.stdout
    # Фоновая сверка обязана быть полной: fast-режим файл не читает.
    assert marker.read_text(encoding="utf-8").strip() == "full|populate-verify"


def test_background_verify_can_be_disabled(tmp_path: Path) -> None:
    marker = tmp_path / "seen.txt"
    sleeper = _sleeper(tmp_path, marker)
    volume = tmp_path / "volume"
    volume.mkdir()
    harness = tmp_path / "harness.sh"
    harness.write_text(_VERIFY_FUNCS, encoding="utf-8")

    result = _run([BASH, str(harness), str(ROOT / "docker" / "entrypoint.sh"), str(volume), str(sleeper), "false"])
    assert result.returncode == 0, result.stderr
    assert "NO_PID" in result.stdout
    assert "фоновая сверка выключена" in result.stdout
    assert not marker.exists()
