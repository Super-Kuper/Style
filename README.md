# image-refs-serverless — Qwen-Image-Edit 2511 compose worker

RunPod serverless worker for the **first-frame-by-reference** flow. It runs
ComfyUI v0.44.0 on one 48 GB GPU, executes a **prepared ComfyUI API graph** built
by the video_gen catalog (Qwen-Image-Edit 2511 compose / identity / clean), and
returns the generated image plus execution-fact telemetry. The contract is
**`image-serverless/v1`** and is unchanged:
[`contract/worker-contract.md`](contract/worker-contract.md).

This is a dedicated profile, not the general image stand:

| Model | Mode | Files |
|---|---|---|
| Qwen-Image-Edit 2511 | compose (style-sheet + location + characters), identity, clean | fp8mixed diffusion model + Qwen2.5-VL text encoder + Qwen-Image VAE |

The manifest contains **only these three pinned files** (~30.16 GB):

| Volume path | Size |
|---|---|
| `diffusion_models/qwen_image_edit_2511_fp8mixed.safetensors` | 20.53 GB |
| `text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors` | 9.38 GB |
| `vae/qwen_image_vae.safetensors` | 0.25 GB |

FLUX.2 klein 4B and Qwen-Image-2512 (t2i) are **out of scope**: they are not in
the manifest and t2i jobs are not sent to this endpoint. The worker itself stays
model-agnostic — it executes whatever graph the service sends.

## Volume: reuse the old one

The image is designed to run on the **existing** `image-serverless` network
volume:

- `populate-volume.sh` checks/downloads only the three rows above; on a warm
  volume this is a few `stat` calls (`метаданные совпадают`), no download;
- ComfyUI still sees every file on the volume through `extra_model_paths.yaml`
  (FLUX/Qwen-2512 remain visible); they are **not verified** by this manifest.
  Do not rely on them here — the profile is t2i-free;
- the background sha256 pass reads ~30.16 GB instead of ~66.7 GB.

A dedicated 40 GB volume also fits the profile; a new empty volume downloads
~30.16 GB on first boot (about 9 minutes at ~56 MB/s).

## Boot

- The baked pod copy (`/opt/comfyui-baked`) and the base pod custom nodes
  (`ComfyUI-Manager`, `ComfyUI-KJNodes`, `Civicomfy`, `ComfyUI-RunpodDirect`)
  are **removed at build time** (`docker/install-comfy-packages.sh`); the image
  is ~127 MB smaller and ComfyUI has no Manager network tasks on boot.
- ComfyUI starts with `--disable-all-custom-nodes --disable-metadata` on
  loopback only (`docker/entrypoint.sh`). The profile is core ComfyUI nodes
  (`TextEncodeQwenImageEditPlus`, `ReferenceMultiLatent` is not needed for this
  graph, `ImageScale`, `VAEEncode`, `KSampler`, `SaveImage`).
- `VOLUME_VERIFY=fast` (default) skips unchanged files by size+mtime;
  `VOLUME_BACKGROUND_VERIFY=true` runs the full sha256 pass in the background
  after the handler starts. Set it to `false` for latency-critical sessions.
- The first job on a cold worker still stages the diffusion model + encoder
  (~30.16 GB) from the volume; a split profile does not remove that cost. Keep
  the worker warm (`idleTimeout` 300–600, `min=1` for a session) or pre-warm.

## Build

```bash
docker build -t <registry>/image-refs-serverless:1.0.0 .
docker push <registry>/image-refs-serverless:1.0.0
docker inspect --format='{{index .RepoDigests 0}}' <registry>/image-refs-serverless:1.0.0
```

Pin the digest. RunPod Hub can build the image from this repository
([`.runpod/hub.json`](.runpod/hub.json)); otherwise use your registry.

Structural build without ComfyUI/GPU (CI-like check):

```bash
docker build --build-arg BASE_IMAGE=python:3.12-slim \
  --build-arg SKIP_COMFY_PACKAGES=1 -t image-refs-serverless:stub .
docker run --rm --entrypoint python image-refs-serverless:stub -c "import image_worker"
```

## Endpoint

| Parameter | Value |
|---|---|
| Type | `QUEUE` |
| GPU | 48 GB primary: `AMPERE_48`, `ADA_48_PRO`; fallback full A100 80 GB (`AMPERE_80`). `BLACKWELL_96` is not used (MIG 2g.48gb slices are slow for this profile). |
| Workers | `max=1`, `min=0` (or `1` for a working session), `idleTimeout=300–600` |
| Endpoint timeout | 1200 s (a warm frame is ~70 s; a cold first frame is minutes) |
| `RUNPOD_INIT_TIMEOUT` | 1800 (first populate of a new volume; not needed on a warm one) |
| Volume | `/runpod-volume` (the existing image-serverless volume) |
| Env | `VOLUME_VERIFY=fast`, `VOLUME_BACKGROUND_VERIFY=true` (or `false` per session), optional `S3_*` for `image_url` |

Assign the endpoint to modality `image` in the studio («Настройки → Запуск GPU»).
Only one image endpoint is active per modality: assigning this one disables the
previous image stand; the old endpoint/image stays for rollback. Side-by-side
(compose + t2i) would need model routing and is out of scope.

## Job input / output

Submit `POST /v2/{endpoint_id}/run` with a prepared ComfyUI API graph:

```json
{
  "input": {
    "prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "qwen_image_edit_2511_fp8mixed.safetensors", "weight_dtype": "default"}}},
    "parameter": {"mode": "edit", "width": 768, "height": 1344, "seed": 20261125},
    "assets": [
      {"name": "style.png", "url": "https://...", "sha256": "...", "kind": "image"}
    ]
  }
}
```

`parameter`/`acceleration` are tracing only; the worker never builds or edits the
graph and never substitutes expected values for measured telemetry. Examples:
[`request-edit.json`](contract/examples/request-edit.json) (base image + optional
mask) and [`request-i2i.json`](contract/examples/request-i2i.json) (style
reference as init image). The t2i example was removed: this profile ships no t2i
weights.

Output: exactly one of `image` (base64) or `image_url` (presigned S3) plus
`telemetry` (`gpu_class`, `gpu_cc`, `seconds`, width/height, model_name, seed,
steps, cfg, sampler, scheduler). Missing/disagreeing facts fail visibly with
`telemetry_missing`/`telemetry_ambiguous`.

## Health, errors, cancellation

`{"input": {"action": "health"}}` → `{"status": "ok", "comfyui": "ready",
"volume_mounted": true}`; health never loads a model.

Stable error codes: `input_invalid`, `input_prompt_missing`, `config_invalid`,
`asset_*`, `comfy_boot_timeout`, `comfy_submit_failed`, `comfy_execution_failed`,
`comfy_timeout`, `comfy_output_missing`, `telemetry_*`, `gpu_unavailable`,
`output_*`. A timeout interrupts the graph and drops it from the ComfyUI queue so
the GPU is freed.

## Steps and quality

The Qwen-Image-Edit-2511 model card uses **40 steps** (`true_cfg_scale=4.0`,
`guidance_scale=1.0`). The video_gen catalog uses **20 steps, cfg 2.5** for
compose; that is already half the card's steps and is the profile default.
`quality=draft` halves it to 10 for iteration. 15 steps is a measurement
candidate only after a live A/B on faces, hands, text and geometry: the first
frame is fed to H3 and its degradation carries into the video. Step changes are
catalog data (`defaults.steps`), not worker code; the snapshot records the steps.

## Tests

Offline (no GPU, ComfyUI or RunPod SDK; ComfyUI HTTP is mocked):

```bash
PYTHONPATH=src ../.deps/testenv/bin/python -m pytest tests -q
```

Coverage: contract schemas/examples, asset validation/download, graph passthrough,
telemetry extraction/ambiguity, PNG/JPEG/WebP dimensions, inline/URL delivery,
health, cancellation, manifest pinning (three refs files), shell syntax.

## Limitations

- No t2i: FLUX.2 klein 4B / Qwen-Image-2512 are not in the manifest and t2i jobs
  must not be sent here.
- The old volume still holds the t2i files; ComfyUI sees them but this manifest
  does not verify them.
- One active image endpoint per modality; the old endpoint remains for rollback.
- Cold first frame still stages ~30.16 GB; warm worker/pre-warm is the real
  latency lever.
- The endpoint image is the versioned copy of `models-manifest.txt`; adding a
  model means rebuilding the image (or overriding `MANIFEST` on the endpoint).
- MIG slices (e.g. `BLACKWELL_96` 2g.48gb) are visible only in the ComfyUI log,
  not in `telemetry.gpu_class`; pin 48 GB full cards if latency matters.

## Licenses

Worker code is MIT (see [LICENSE](LICENSE)). The image includes ComfyUI (GPL-3.0)
and NVIDIA CUDA/PyTorch components under their own terms. Model weights are not
part of the image; Qwen-Image-Edit-2511 is Apache-2.0 (commercial use allowed),
downloaded from pinned Hugging Face commits to the network volume.
