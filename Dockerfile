# =============================================================================
# Refs image serverless worker — RunPod serverless (contract image-serverless/v1).
#
# Dedicated profile for the first-frame-by-reference flow:
# Qwen-Image-Edit 2511 compose/identity/clean on the existing network volume.
# The manifest is the three pinned files of that model; t2i is not shipped.
#
# Base image — the official ComfyUI v0.44.0 CUDA 13.0 image, pinned by digest
# (the same base the H3 worker uses; read-only reference):
#   runpod/comfyui:2.2.0-comfyuiv0.44.0-cuda13.0
#   @sha256:adccffa4dc2151b48dd7110b7b03a5f9450b059bc8524bc493b93e5155aea3e2
#
# The base image is a "baked" pod image: ComfyUI lives in /opt/comfyui-baked and
# /start.sh copies it to /workspace/runpod-slim at pod boot (plus SSH, Jupyter
# and FileBrowser). That is not what a serverless worker needs, so the layout is
# materialized at build time (docker/install-comfy-packages.sh):
#   /opt/comfyui-baked/            → $COMFYUI_DIR (/workspace/runpod-slim/ComfyUI)
#   venv --system-site-packages    → $COMFYUI_PYTHON
# Our ENTRYPOINT (docker/entrypoint.sh) starts ComfyUI on 127.0.0.1:8188 and
# never calls the base /start.sh.
#
# Image profile: Qwen-Image-Edit 2511 (compose / identity / clean) only. The
# graph is built by the video_gen catalog; the image carries no weights and no
# t2i models (FLUX.2 klein 4B / Qwen-Image-2512 are out of scope). Core ComfyUI
# nodes cover the profile; pod custom nodes are removed at build time.
#
# Structural build without ComfyUI/GPU (CI-like check):
#   docker build --build-arg BASE_IMAGE=python:3.12-slim \
#     --build-arg SKIP_COMFY_PACKAGES=1 -t image-refs-serverless:stub .
#   docker run --rm --entrypoint python image-refs-serverless:stub -c "import image_worker"
# That image is NOT meant to run: it has no ComfyUI.
#
# Boundaries:
#   - ComfyUI listens on loopback only and is never exposed: no EXPOSE on purpose;
#   - no secrets, tokens or model weights in the image: secrets come from the
#     endpoint env, weights from the network volume at /runpod-volume;
#   - models are downloaded to the volume at boot by populate-volume.sh
#     (models-manifest.txt, VOLUME_AUTOPOPULATE=true by default);
#   - USER root is intentional: ComfyUI, custom nodes and RunPod volumes work
#     as root.
# =============================================================================

# Official runpod/comfyui with ComfyUI v0.44.0, pinned by digest.
# Overridable with --build-arg for the structural stub build or a re-pinned base.
ARG BASE_IMAGE=runpod/comfyui:2.2.0-comfyuiv0.44.0-cuda13.0@sha256:adccffa4dc2151b48dd7110b7b03a5f9450b059bc8524bc493b93e5155aea3e2

FROM ${BASE_IMAGE}

# Expected ComfyUI tag: the install script prints the baked version and only
# checks out the tag if it differs.
ARG COMFYUI_VERSION=v0.44.0

# =1 — skip ComfyUI installation (structural stub build).
ARG SKIP_COMFY_PACKAGES=0

# Worker version for logs/labels: --build-arg IMAGE_WORKER_VERSION=1.0.0.
ARG IMAGE_WORKER_VERSION=1.0.0

# ARG PYTHONPATH is declared on purpose: it preserves the base image PYTHONPATH
# when our path is appended below and removes the UndefinedVar build warning.
ARG PYTHONPATH

# --- Image environment --------------------------------------------------------
# PYTHONUNBUFFERED=1 — worker/ComfyUI logs go to RunPod stdout immediately.
# COMFYUI_DIR/COMFYUI_PYTHON — final image layout (materialized by the install
# script); the endpoint env can override them (docker/entrypoint.sh).
# COMFYUI_BAKED_DIR — baked ComfyUI copy in the base image (layout source).
# PYTHONPATH — worker package without installation.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    COMFYUI_DIR=/workspace/runpod-slim/ComfyUI \
    COMFYUI_PYTHON=/workspace/runpod-slim/.venv-cu130/bin/python \
    COMFYUI_BAKED_DIR=/opt/comfyui-baked \
    COMFY_HOST=127.0.0.1 \
    COMFY_PORT=8188 \
    IMAGE_WORKER_VERSION=${IMAGE_WORKER_VERSION}

ENV PYTHONPATH=/opt/image-worker/src${PYTHONPATH:+:$PYTHONPATH}

# --- System utilities ---------------------------------------------------------
# curl is used by scripts/wait-comfy.sh and populate-volume.sh; git by the
# optional custom-node installer; rsync to materialize the baked ComfyUI.
USER root

RUN set -eux; \
    apt-get update; \
    apt-get install -y --no-install-recommends curl git ca-certificates rsync; \
    rm -rf /var/lib/apt/lists/*

# --- ComfyUI v0.44.0 layout (core nodes only) --------------------------------
COPY docker/install-comfy-packages.sh /opt/image-worker/install-comfy-packages.sh

RUN set -eux; \
    chmod 0755 /opt/image-worker/install-comfy-packages.sh; \
    COMFYUI_VERSION="${COMFYUI_VERSION}" \
    SKIP_COMFY_PACKAGES="${SKIP_COMFY_PACKAGES}" \
    bash /opt/image-worker/install-comfy-packages.sh

# --- Worker Python dependencies ----------------------------------------------
# runpod SDK, httpx (ComfyUI client/assets) and boto3 (presigned image_url
# delivery) go into the ComfyUI venv: the entrypoint runs the handler with the
# same interpreter, so no package installation is needed at runtime.
# PIP_CONSTRAINT from the base image keeps the torch/CUDA stack unchanged.
RUN set -eux; \
    py="${COMFYUI_PYTHON}"; \
    if [ ! -x "${py}" ]; then py="$(command -v python3)"; fi; \
    echo "worker deps python: ${py}"; \
    if [ -f /opt/comfyui-runtime-constraints.txt ]; then \
        export PIP_CONSTRAINT=/opt/comfyui-runtime-constraints.txt; \
    fi; \
    if "${py}" -m pip --version >/dev/null 2>&1; then \
        "${py}" -m pip install --no-cache-dir "runpod==2.2.1" "httpx>=0.27,<0.29" "boto3>=1.35"; \
    elif command -v uv >/dev/null 2>&1; then \
        UV_CACHE_DIR=/tmp/uv-cache uv pip install --python "${py}" "runpod==2.2.1" "httpx>=0.27,<0.29" "boto3>=1.35"; \
    else \
        "${py}" -m ensurepip --upgrade; \
        "${py}" -m pip install --no-cache-dir "runpod==2.2.1" "httpx>=0.27,<0.29" "boto3>=1.35"; \
    fi; \
    "${py}" -c "import boto3, httpx, runpod; print('worker deps ok, httpx', httpx.__version__)"

# --- Worker package and runtime files ----------------------------------------
WORKDIR /opt/image-worker

# image_worker package (imported via PYTHONPATH above).
COPY src/ /opt/image-worker/src/

# RunPod serverless entry script.
COPY handler.py /opt/image-worker/handler.py

# Entrypoint: boot ComfyUI → wait for /system_stats → populate volume → handler.
COPY docker/entrypoint.sh /opt/image-worker/entrypoint.sh

# Waiter: polls http://127.0.0.1:8188/system_stats with a timeout.
COPY scripts/wait-comfy.sh /opt/image-worker/wait-comfy.sh

# Model paths shared through the network volume: the entrypoint copies this
# file to ${COMFYUI_DIR}/extra_model_paths.yaml before starting ComfyUI.
COPY docker/extra_model_paths.yaml /opt/image-worker/extra_model_paths.yaml

# Volume auto-population: model manifest + downloader (VOLUME_AUTOPOPULATE).
COPY models-manifest.txt /opt/image-worker/models-manifest.txt
COPY populate-volume.sh /opt/image-worker/populate-volume.sh

RUN chmod 0755 /opt/image-worker/entrypoint.sh /opt/image-worker/wait-comfy.sh /opt/image-worker/populate-volume.sh

# ComfyUI is never exposed (loopback only), so there is no EXPOSE on purpose.
#
# ENTRYPOINT is the single entry point: RunPod serverless only starts the
# container, and the handler is started by the entrypoint script. It overrides
# the base image /start.sh (pod bootstrap is not used, see the file header).
ENTRYPOINT ["/opt/image-worker/entrypoint.sh"]
