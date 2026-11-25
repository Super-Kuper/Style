#!/usr/bin/env bash
# =============================================================================
# Подготовка ComfyUI v0.44.0 в образе serverless-воркера изображений.
#
# ВАЖНО: официальный runpod/comfyui — «baked»-образ: готовый ComfyUI лежит в
# /opt/comfyui-baked, а /start.sh раскладывает его в /workspace/runpod-slim при
# старте пода (и создаёт venv с --system-site-packages, SSH/Jupyter/FileBrowser,
# ComfyUI на 0.0.0.0). Для serverless это неприемлемо (долгий сетевой bootstrap,
# лишние сервисы). Поэтому раскладка **материализуется на этапе сборки**:
#   /opt/comfyui-baked/  →  $COMFYUI_DIR/
#   venv                 →  $COMFYUI_PYTHON (--system-site-packages,
#                            torch/CUDA из системных пакетов базы)
# Наш docker/entrypoint.sh стартует ComfyUI на 127.0.0.1:8188 и НЕ вызывает
# /start.sh базового образа.
#
# Custom nodes НЕ ставятся: профиль refs (Qwen-Image-Edit 2511 compose/
# identity/clean) покрыт core-нодами ComfyUI v0.44.0. Pod-ноды базового образа
# (Manager/KJNodes/Civicomfy/RunpodDirect) удаляются ниже: воркеру они не
# нужны. Если будущей модели нужен пинованный пакет — добавьте его сюда по
# коммиту (SHA), не по `main`, и перечислите в README.
#
# Скрипт вызывается из Dockerfile ПОСЛЕ COPY и ДО кода воркера. Env:
#   COMFYUI_DIR          каталог ComfyUI (раскладка материализуется сюда)
#   COMFYUI_PYTHON       интерпретатор venv (torch/CUDA из system-site-packages)
#   COMFYUI_BAKED_DIR    baked-копия ComfyUI в базовом образе
#   COMFYUI_VERSION      ожидаемый тег ComfyUI (v0.44.0)
#   SKIP_COMFY_PACKAGES  =1 — не ставить пакеты (stub-сборка на python:3.12-slim)
#
# Stub-сборка: в python:3.12-slim нет /opt/comfyui-baked → шаг пропускается с
# логом и кодом 0 (проверка структуры слоя воркера).
# =============================================================================

set -euo pipefail

COMFYUI_DIR="${COMFYUI_DIR:-/workspace/runpod-slim/ComfyUI}"
COMFYUI_PYTHON="${COMFYUI_PYTHON:-/workspace/runpod-slim/.venv-cu130/bin/python}"
COMFYUI_BAKED_DIR="${COMFYUI_BAKED_DIR:-/opt/comfyui-baked}"
COMFYUI_VERSION="${COMFYUI_VERSION:-v0.44.0}"
SKIP_COMFY_PACKAGES="${SKIP_COMFY_PACKAGES:-0}"
# uv (если появится в образе) не должен писать в read-only HOME при сборке.
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

log() { printf '[packages] %s\n' "$*"; }
warn() { printf '[packages] WARN: %s\n' "$*" >&2; }
die() {
    printf '[packages] ERROR: %s\n' "$*" >&2
    exit 1
}

# Держим torch/vision/audio в версиях базового образа (cu130) при любых pip
# install: файл есть в официальном runpod/comfyui. См. /start.sh базового образа.
if [[ -f /opt/comfyui-runtime-constraints.txt ]]; then
    export PIP_CONSTRAINT=/opt/comfyui-runtime-constraints.txt
    export PIP_BUILD_CONSTRAINT=/opt/comfyui-runtime-constraints.txt
    log "PIP_CONSTRAINT=/opt/comfyui-runtime-constraints.txt ($(tr '\n' ' ' </opt/comfyui-runtime-constraints.txt))"
fi

# --- 0. Применимость --------------------------------------------------------
if [[ ! -d "${COMFYUI_BAKED_DIR}" ]]; then
    log "SKIP: ${COMFYUI_BAKED_DIR} нет в базовом образе (stub-сборка без ComfyUI) — пакеты не ставятся"
    exit 0
fi
if [[ "${SKIP_COMFY_PACKAGES}" == "1" ]]; then
    log "SKIP_COMFY_PACKAGES=1: установка ComfyUI/пакетов пропущена (проверка структуры образа)"
    exit 0
fi

# --- 1. Версия ComfyUI в базовом образе -------------------------------------
# Digest базового образа закреплён под ComfyUI v0.44.0. Версию читаем из
# comfyui_version.py (offline, есть и в baked-копии с .git): если совпадает —
# checkout/fetch не нужен. Иначе сверяемся с git-тегом; расхождение — die.
# Проверка версии остаётся в логе сборки: это доказательство, что раскладка
# соответствует заявленной версии.
version_ok=0
version_file="${COMFYUI_BAKED_DIR}/comfyui_version.py"
if [[ -f "${version_file}" ]]; then
    py_version="$(awk -F'"' '/^__version__/{print $2; exit}' "${version_file}")"
    if [[ -z "${py_version}" ]]; then
        py_version="$(awk -F"'" '/^__version__/{print $2; exit}' "${version_file}")"
    fi
    log "ComfyUI в базовом образе: ${py_version:-unknown} из comfyui_version.py (ожидается ${COMFYUI_VERSION#v})"
    if [[ "${py_version}" == "${COMFYUI_VERSION#v}" ]]; then
        version_ok=1
    fi
fi

if [[ "${version_ok}" == "1" ]]; then
    log "ComfyUI: версия совпадает с ${COMFYUI_VERSION} — checkout не требуется"
elif [[ -d "${COMFYUI_BAKED_DIR}/.git" ]]; then
    current="$(git -C "${COMFYUI_BAKED_DIR}" describe --tags --always 2>/dev/null || echo unknown)"
    log "ComfyUI git describe: ${current} (ожидается ${COMFYUI_VERSION})"
    if [[ "${current}" == "${COMFYUI_VERSION}" || "${current}" == "${COMFYUI_VERSION}-"* ]]; then
        log "ComfyUI: версия совпадает с ${COMFYUI_VERSION} — checkout не требуется"
    else
        warn "ComfyUI: версия ${current} != ${COMFYUI_VERSION} — пробую checkout тега"
        git -C "${COMFYUI_BAKED_DIR}" fetch --tags origin
        git -C "${COMFYUI_BAKED_DIR}" checkout --detach "${COMFYUI_VERSION}"
        after="$(git -C "${COMFYUI_BAKED_DIR}" describe --tags --always 2>/dev/null || echo unknown)"
        log "ComfyUI после checkout: ${after}"
        if [[ "${after}" != "${COMFYUI_VERSION}" && "${after}" != "${COMFYUI_VERSION}-"* ]]; then
            die "не удалось привести ComfyUI к ${COMFYUI_VERSION} (сейчас ${after}); base-образ не соответствует пину"
        fi
    fi
else
    die "${COMFYUI_BAKED_DIR}: нет ни comfyui_version.py с версией ${COMFYUI_VERSION}, ни .git — не могу проверить версию ComfyUI"
fi

# --- 2. Раскладка: baked → /workspace/runpod-slim/ComfyUI --------------------
# .git не копируем (20 МБ, для запуска не нужен; версия уже проверена и
# залогирована). Baked custom_nodes базового образа переносятся, чтобы сразу
# удалить их в 2b: профилю refs они не нужны (см. ниже).
mkdir -p "$(dirname "${COMFYUI_DIR}")"
log "раскладка ComfyUI: ${COMFYUI_BAKED_DIR}/ → ${COMFYUI_DIR}/ (без .git)"
rsync -a --exclude='/.git' "${COMFYUI_BAKED_DIR}/" "${COMFYUI_DIR}/"

# --- 2b. Профиль refs: не тащим baked-копию и pod-ноды ---
# Материализованная копия выше — единственное дерево ComfyUI, нужное воркеру.
# /start.sh базового образа не вызывается, /opt/comfyui-baked больше не нужен
# (106 МБ). Custom nodes базового образа (Manager/KJNodes/Civicomfy/
# RunpodDirect) воркеру не нужны: графы каталога — core-ноды ComfyUI. Удаление
# убирает их импорт и фоновые сетевые задачи ComfyUI-Manager на старте.
rm -rf "${COMFYUI_BAKED_DIR}"
for node in ComfyUI-Manager ComfyUI-KJNodes Civicomfy ComfyUI-RunpodDirect; do
    rm -rf "${COMFYUI_DIR}/custom_nodes/${node}"
done
log "профиль refs: baked-копия и pod-ноды удалены (${COMFYUI_DIR}/custom_nodes)"

# --- 3. venv ComfyUI ---------------------------------------------------------
# --system-site-packages: torch и CUDA-стек уже в системных пакетах базового
# образа (constraints /opt/comfyui-runtime-constraints.txt); --without-pip: pip
# берётся из system site-packages и ставит пакеты в venv (sys.prefix), т.е. не
# тянем второй pip. Так же делает /start.sh базового образа.
if [[ ! -x "${COMFYUI_PYTHON}" ]]; then
    log "создаю venv ${COMFYUI_PYTHON%/bin/python} (--system-site-packages --without-pip)"
    python3 -m venv --system-site-packages --without-pip "${COMFYUI_PYTHON%/bin/python}"
fi
[[ -x "${COMFYUI_PYTHON}" ]] || die "не удалось создать venv ${COMFYUI_PYTHON}"
log "Python ComfyUI: $("${COMFYUI_PYTHON}" -V 2>&1)"
log "torch из base: $("${COMFYUI_PYTHON}" -c 'import torch; print(torch.__version__, "cuda", torch.version.cuda)' 2>&1 || echo 'import torch FAILED')"

# --- 4. Проверка core-нод моделей и i2i -------------------------------------
# На этапе сборки ComfyUI ещё не запущен, но имена core-нод можно проверить по
# исходникам: для двух моделей стенда нужны UNETLoader, CLIPLoader, VAELoader,
# KSampler, SaveImage, LoadImage, LoadImageMask. Для i2i/референса — ImageScale,
# VAEEncode и TextEncodeQwenImageEdit (Qwen-Image-Edit). Отсутствие —
# предупреждение (в собранной вручную базе layout может отличаться), но версия
# ниже проверит совместимость при первом job через comfy_submit_failed.
_core_nodes_dir="${COMFYUI_DIR}/comfy_extras"
for node in UNETLoader CLIPLoader VAELoader KSampler SaveImage LoadImage LoadImageMask VAEEncode ImageScale ImageScaleToTotalPixels TextEncodeQwenImageEdit; do
    if ! grep -Rqs "${node}" "${COMFYUI_DIR}/nodes.py" "${_core_nodes_dir}" 2>/dev/null; then
        warn "core-нода ${node} не найдена в ${COMFYUI_DIR}: проверьте версию ComfyUI"
    fi
done

log "готово: ComfyUI ${COMFYUI_VERSION} в ${COMFYUI_DIR}, python ${COMFYUI_PYTHON}"
