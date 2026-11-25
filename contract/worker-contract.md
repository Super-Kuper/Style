# Worker contract — image-serverless/v1

This document is the source of truth for the job input, output and errors of the
worker in this repository. The contract is versioned separately from the worker
release: an incompatible change is a new contract version (`v2`).

The worker executes a prepared ComfyUI API graph on a local ComfyUI instance and
returns the generated image plus execution-fact telemetry. It **never builds or
edits the graph**: the application (video_gen) renders the graph from the
`image_models` catalog template, substitutes `{{...}}` placeholders and sends the
finished API graph. An empty or unreadable image is never a successful result.
JSON Schemas and examples:
[`input.schema.json`](input.schema.json), [`output.schema.json`](output.schema.json),
[`examples/`](examples/).

## 1. Job input

`POST {RUNPOD_SERVERLESS_API_BASE}/v2/{endpoint_id}/run`

```json
{
  "input": {
    "prompt": {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "flux-2-klein-4b.safetensors", "weight_dtype": "default"}}},
    "parameter": {"mode": "generate", "width": 768, "height": 1344, "seed": 20261113},
    "acceleration": {"gpu_class": "RTX A6000", "use_turbo": false}
  },
  "policy": {"executionTimeout": 900000, "ttl": 3600000}
}
```

| Field | Meaning |
|---|---|
| `prompt` | Prepared ComfyUI API graph (`node id -> {class_type, inputs}`), submitted unchanged; the worker does not build or edit it, and ComfyUI rejects invalid graphs (`comfy_submit_failed`). Required unless `action` is `health`. |
| `parameter` | Snapshot of generation parameters (`mode`, `scene_id`, `width`, `height`, `seed`, `steps`, `cfg`, `sampler`, `scheduler`, `prompt`, `negative_prompt`, `model`, `edit`, ...). Recorded for tracing; **never** used to build the graph or to fill telemetry. |
| `acceleration` | Snapshot of acceleration settings (`gpu_class`, `use_turbo`, ...). Recorded for tracing; never used to fill telemetry. |
| `action` | `"health"` runs the health check without a prompt, a model load or a graph. Any other value fails with `input_invalid`. |
| `assets` | Optional input files delivered as presigned GET URLs; see below. |
| `policy` | Optional RunPod policy (`executionTimeout`, `ttl`). The worker does not read it; RunPod applies it before the handler runs. |

### 1.1. Input assets

`input.assets` is a list of at most 8 entries. Each entry has a required `name`
(basename without path segments, `^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$`) and `url`
(HTTP(S) presigned GET URL), plus optional `sha256` (64 lowercase hex), `kind`
(`image`/`mask`) and `size` in bytes; no other fields are accepted. Names must be
unique. The worker validates the list before waiting for ComfyUI, downloads each
file into the ComfyUI input directory under `name`, verifies `sha256` and the
declared size, and atomically replaces the target file; the graph references
those names in `LoadImage` / `LoadImageMask` / `LoadImageOutput`. Limits:
`ASSETS_MAX_MB` per file and `ASSETS_MAX_TOTAL_MB` in total (defaults 64 MB /
128 MB); declared sizes are checked before download, actual sizes while
streaming.

For an **edit** graph the application sends the base image and, when the model
supports masks, the mask as two assets whose names are exactly the ones the graph
expects. The application picks the names (the service uses
`asset_<sha16>.png`); the graph template must reference those same names. The
examples below use short illustrative names:

```json
"assets": [
  {"name": "base.png", "url": "https://.../base.png", "sha256": "<64 hex>", "kind": "image", "size": 184320},
  {"name": "mask.png", "url": "https://.../mask.png", "sha256": "<64 hex>", "kind": "mask", "size": 18432}
]
```

and the edit graph references `"image": "base.png"` in `LoadImage`,
`"image": "mask.png"` in `LoadImageMask`. `kind` is advisory: the worker only
validates it, the graph decides how the file is used. The base image is never
overwritten by an edit result: the worker only writes the files listed in
`assets` and never writes back to the application asset store.

### 1.2. Image-to-image / style reference

Image-to-image uses the same `assets` mechanism: the application sends the style
reference image as an `image` asset under a fixed name and the graph loads it
with `LoadImage`. The graph is responsible for scaling (`ImageScale`), encoding
(`VAEEncode`) and partial denoising (`KSampler.denoise` between `0` and `1`);
the worker does not change the graph or the denoise. There is no separate `mode`
field in the contract — `parameter.mode` is tracing data only.

For Qwen-Image-Edit, `TextEncodeQwenImageEdit` takes the reference image and the
prompt: in the i2i/style-reference mode the reference supplies style, the prompt
describes the new frame. The example is
[`examples/request-i2i.json`](examples/request-i2i.json). The application
attaches that reference only when the catalog model declares
`capabilities.style_reference` and the graph template has the
`{{style_reference}}` placeholder; otherwise the reference is not sent. The
worker verifies and writes the file exactly like any other asset.

For the style-container two-pass flow the graph uses
`TextEncodeQwenImageEditPlus` with named image roles, not the i2i mode:
`style_reference` + `location` for the compose pass, then `base` +
`character_1`/`character_2` for the identity pass. The application sends each
role as an `image` asset with the graph's filename; the worker does not apply
masks, crops or denoise — the graph owns `ImageScaleToTotalPixels`, latent
encoding and `KSampler.denoise`. The reference-cleanup pass sends one
`source` image and receives a cleaned variant; approve happens in the app and
creates a new style/location version, never overwrites the original.

## 2. Job output

The handler returns one object that RunPod places in `output` of the status
response.

```json
{
  "image": "<base64 png>",
  "telemetry": {
    "gpu_cc": 8.6, "gpu_class": "RTX A6000", "seconds": 7.4,
    "width": 768, "height": 1344, "model_name": "flux-2-klein-4b.safetensors",
    "seed": 20261113, "steps": 4, "cfg": 1.0, "sampler": "euler", "scheduler": "simple"
  }
}
```

- Exactly one of `image` (base64 PNG/JPEG/WebP) or `image_url` (presigned URL)
  is present. Results up to `OUTPUT_INLINE_MAX_MB` and within
  `OUTPUT_INLINE_BUDGET_MB` of encoded payload are returned inline; larger files
  go to S3-compatible storage as `image_url` with lifetime
  `OUTPUT_URL_TTL_SECONDS`.
- `telemetry` is required; the minimum is defined by
  [`output.schema.json`](output.schema.json): `gpu_cc`, `gpu_class`, `seconds`,
  `width`, `height`, `model_name`, `seed`, `steps`, `cfg`, `sampler`,
  `scheduler`.
- Telemetry comes from **execution facts**, never from `parameter`/`acceleration`:

| Field | Measured from |
|---|---|
| `gpu_cc`, `gpu_class` | `torch.cuda` (capability + device name) inside the worker; env override `IMAGE_GPU_CC`/`IMAGE_GPU_CLASS` only as a pair for environments without torch. |
| `seconds` | Wall clock from `POST /prompt` to the completed `/history` entry. |
| `width`, `height` | Header of the produced file (PNG IHDR, JPEG SOF, WebP VP8/VP8L/VP8X). |
| `model_name` | Loader node in the executed graph: `UNETLoader.unet_name` / `UnetLoaderGGUF(unet_name)` / `UnetLoaderGGUFAdvanced(unet_name)` or `CheckpointLoaderSimple.ckpt_name` / `CheckpointLoader.ckpt_name`. |
| `seed` | `KSampler.seed` / `KSamplerAdvanced.seed`; for `SamplerCustom*` — linked `RandomNoise.noise_seed`. |
| `steps` | `KSampler.steps` / `KSamplerAdvanced.steps`; for `SamplerCustom*` — linked `BasicScheduler.steps`. |
| `cfg` | `KSampler.cfg` / `KSamplerAdvanced.cfg` / `SamplerCustom.cfg`; for `SamplerCustomAdvanced` — linked `CFGGuider.cfg`. |
| `sampler` | `sampler_name` of `KSampler*` or linked `KSamplerSelect.sampler_name`. |
| `scheduler` | `KSampler.scheduler` or linked `BasicScheduler.scheduler`. |

- If a fact cannot be measured the job fails with `telemetry_missing`; if the
  executed graph contains several sampler/loader nodes that disagree, the job
  fails with `telemetry_ambiguous`. Expected values are never substituted.
- `text_detected` (OCR) is **not** reported by this worker: there is no OCR
  engine in the image, so the field is omitted rather than set to a silent
  `false`. The application runs OCR itself and records the fact
  (`keyframe_text_detected` in the pipeline). If a future worker release adds
  OCR, it must report `text_detected` (and may add `text`); the pipeline then
  stops the frame instead of canonicalizing it.
- `model_name` is the **diffusion model file name** from the executed graph. The
  application snapshot stores the model `slug`, so the service matches the
  returned file name against the catalog's pinned
  `image_models.model_files[].name/path`. The catalog row must list the exact
  volume file name (for example `qwen_image_edit_2511_fp8mixed.safetensors`),
  otherwise the frame is rejected with `keyframe_model_mismatch`.

## 3. Health

```json
{"status": "ok", "comfyui": "ready", "volume_mounted": true}
```

- `status`: `ok` (ComfyUI ready and the volume mounted), `starting` (ComfyUI
  booting, volume mounted; normal transient state) or `degraded` (volume missing
  and/or ComfyUI unavailable).
- `comfyui`: `ready`, `booting` or `unavailable` — result of a short
  `GET /system_stats` probe.
- `volume_mounted`: whether `VOLUME_ROOT` exists. Health never loads a model or a
  graph, so it is safe as a smoke test and for diagnostics.

## 4. Errors and cancellation

Any handler error fails the RunPod job (`FAILED`); error codes are stable
lowercase identifiers:

| Group | Codes |
|---|---|
| Input/config | `input_invalid`, `input_prompt_missing`, `config_invalid` |
| Assets | `asset_invalid`, `asset_download_failed`, `asset_hash_mismatch`, `asset_too_large` |
| ComfyUI | `comfy_boot_timeout`, `comfy_submit_failed`, `comfy_execution_failed`, `comfy_timeout`, `comfy_output_missing` |
| Telemetry | `telemetry_missing`, `telemetry_ambiguous`, `gpu_unavailable` |
| Output | `output_empty`, `output_invalid`, `output_storage_missing`, `output_storage_failed` |

- Cancellation (`POST /v2/{endpoint_id}/cancel/{run_id}`) interrupts the running
  graph in ComfyUI before the job terminates. `JOB_TIMEOUT_SECONDS` (default
  900) bounds the graph run inside the handler; the endpoint `timeout` and
  `policy.executionTimeout` must be greater than or equal to it, otherwise
  RunPod aborts the job before the worker can return an explicit error.
- `output_invalid` means the produced file is not a decodable PNG/JPEG/WebP (or
  its size header cannot be read); `output_empty` means zero bytes. Neither is a
  success.

## 5. Environment

The worker reads its settings from endpoint environment variables. Invalid values
fail the job with `config_invalid` and the variable name.

| Variable | Default | Purpose |
|---|---|---|
| `COMFY_URL` | `http://127.0.0.1:8188` | Local ComfyUI address; loopback only. |
| `COMFYUI_DIR` / `COMFYUI_INPUT_DIR` | `/workspace/runpod-slim/ComfyUI` / `${COMFYUI_DIR}/input` | ComfyUI root and directory for `input.assets`. |
| `COMFY_BOOT_TIMEOUT_SECONDS` | `600` | Wait for ComfyUI to start. |
| `JOB_TIMEOUT_SECONDS` | `900` | Maximum graph run time (alias `COMFY_JOB_TIMEOUT_SECONDS`). |
| `COMFY_POLL_INITIAL_SECONDS` / `COMFY_POLL_MAX_SECONDS` / `COMFY_REQUEST_TIMEOUT_SECONDS` | `0.5` / `3` / `60` | `/history` polling backoff and per-request HTTP timeout. |
| `VOLUME_ROOT` / `MODELS_ROOT` | `/runpod-volume` / `${VOLUME_ROOT}/models` | Volume mount point and root of model weights. |
| `OUTPUT_INLINE_MAX_MB` / `OUTPUT_INLINE_BUDGET_MB` | `10` / `15` | Inline base64 file-size limit and encoded-payload budget. |
| `OUTPUT_URL_TTL_SECONDS` | `86400` | Lifetime of the presigned `image_url`. |
| `ASSETS_MAX_MB` / `ASSETS_MAX_TOTAL_MB` / `ASSETS_TIMEOUT_SECONDS` | `64` / `128` / `120` | Per-file limit, total limit and download timeout for `input.assets`. |
| `HEALTH_TIMEOUT_SECONDS` | `10` | `/system_stats` timeout for the health action. |
| `LINK_VOLUME_NODES` | `false` | Link `custom_nodes` from the volume at boot. |
| `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_REGION`, `S3_SECURE` | — / `us-east-1` / auto | S3-compatible storage for large results; bucket alias `OUTPUT_BUCKET`. |
| `IMAGE_GPU_CC` / `IMAGE_GPU_CLASS` | — | Optional GPU fact override for environments without torch; set only as a pair. |

`S3_ACCESS_KEY` and `S3_SECRET_KEY` are secrets: set them only in the endpoint
environment, never in the image, the repository or logs. The image sets its own
boot variables; endpoint operators normally need to change only the variables in
this table.

Container boot variables are read by `docker/entrypoint.sh` and the volume
scripts, not by the handler: `VOLUME_AUTOPOPULATE` (default `true`) and
`HF_ENDPOINT`/`HF_TOKEN` control the model download, `VOLUME_VERIFY` (`fast`,
`full` or `off`) and `VOLUME_BACKGROUND_VERIFY` (default `true`) select how the
existing volume is checked — `fast` trusts the recorded size+mtime and the full
sha256 pass runs in the background after the handler starts — `COMFYUI_PYTHON`
selects the interpreter, and `COMFY_START_CMD` overrides how ComfyUI is started.
None of them changes the request/response contract.

## 6. Model graphs and the `image_models` catalog

The worker is model-agnostic by design: adding a model means adding files to the
volume and a graph template to the `image_models` catalog, not changing this
repository. Constraints the application must respect when rendering a graph:

1. The graph is a complete ComfyUI **API-format** graph (`GET /prompt` shape),
   with no `{{...}}` placeholders left, and it ends in a save node whose
   `outputs` contain an `images` list (`SaveImage` or equivalent).
2. The sampler must be one of `KSampler`, `KSamplerAdvanced`, `SamplerCustom`,
   `SamplerCustomAdvanced` with the linked-core nodes listed in §2, otherwise
   telemetry cannot be measured (`telemetry_missing`).
3. The diffusion model must be loaded by a known loader node (§2), otherwise
   `model_name` cannot be measured.
4. `parameter.steps/cfg/sampler/scheduler/seed/width/height/model_name` in the
   snapshot must equal the values actually rendered into the graph; the service
   compares them with the returned telemetry and stops the step on a deviation.
5. Input files (base image, mask, references) are referenced by the names sent in
   `input.assets`; the worker writes exactly those names to the ComfyUI input
   directory.

## 7. Version

Current contract version: **`image-serverless/v1`**. Input and output are
additive within a minor version. An incompatible change requires a new contract
version and a matching update of the schemas, examples and this document.
