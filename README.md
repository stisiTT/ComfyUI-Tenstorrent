# Tenstorrent Custom Nodes for ComfyUI

Custom nodes for running Stable Diffusion XL (image) and Wan 2.2 (text-to-video)
on Tenstorrent hardware (N150 / N300 / T3K / p150 / p300x2 / Galaxy) via a
tt-metal HTTP inference server.

## How it works

These nodes do **not** run the model in the ComfyUI process. Instead, the
`TT_CheckpointLoader` node stands up (and supervises) a tt-metal FastAPI
inference server and talks to it over HTTP. All compute — CLIP, UNet/DiT, VAE —
happens on the Tenstorrent device inside that server.

```
ComfyUI (this process, ComfyUI venv)
  └─ TT_CheckpointLoader
       └─ server_manager.py
            ├─ Popen ──▶ <tt-metal>/launch_server.sh   (spawned subprocess, tt-metal python_env)
            └─ HTTP  ──▶ http://127.0.0.1:8000          (/health, inference endpoints)
                              └─ tt-metal server.py ──▶ Tenstorrent hardware
```

The two repos use **separate Python virtual environments on purpose** (so
tt-metal's native/NumPy ABI never contaminates ComfyUI). The only coupling is
the subprocess launch + the local HTTP socket.

- **Single model at a time.** The server serves one model; selecting a different
  `model_type` stops the running server and starts the new one (a multi-minute
  warmup each switch).
- **First launch is slow.** Initial warmup includes trace capture: ~5–10 min for
  SDXL, ~15–25 min for Wan 2.2. The node blocks until `/health` reports healthy.

## Prerequisites

1. **Tenstorrent hardware** with drivers/firmware installed (e.g. via
   `tt-installer`), and `tt-smi` available on `PATH`.
2. **A built tt-metal checkout** on the branch that contains the standalone media
   server (`server.py`, `launch_server.sh`, `requirements-server.txt`,
   `device_specs.py`, `worker.py`, the SDXL/Wan runners), with its `python_env`
   created (`./create_venv.sh`). By default these nodes look for tt-metal as a
   sibling of the ComfyUI checkout (`../tt-metal`); override with `TT_METAL_DIR`.

   There are two self-consistent stacks and they cannot be mixed — the relocated
   server needs `fuse_lora(lora_scale, clip_scale)`, which only the
   per-component-LoRA tt-metal branch accepts. **Read
   [`INTEGRATION.md`](./INTEGRATION.md) for the version matrix** before pinning
   anything; it also has the PR breakdown and where the server is headed
   (tt-inference-server).
3. **Model weights** reachable by the tt-metal server (downloaded into its
   `HF_HOME` / ttnn model cache).

## Launch modes

The node can stand the server up two ways, selected by `TT_LAUNCH_MODE`:

- **`subprocess` (default)** — spawns `<TT_METAL_DIR>/launch_server.sh` directly on
  the host, no container. Needs nothing built beyond tt-metal itself, which is why
  it is the default.
- **`docker`** — drives `tt-inference-server`'s
  `run.py --workflow server --docker-server --dev-mode --override-docker-image`
  to start the `comfyui-media-server` container. Opt-in: build the image first
  (see `tt-inference-server/comfyui-media-server/LOCAL_TESTING.md`).

```
subprocess mode:   node → <TT_METAL_DIR>/launch_server.sh → server.py → TT HW
docker mode:       node → run.py --docker-server → container (comfyui-media-server) → TT HW
```

## Configuration (environment variables)

All read by `server_manager.py`; every default is overridable:

| Variable            | Default                                  | Purpose |
|---------------------|------------------------------------------|---------|
| `TT_LAUNCH_MODE`    | `subprocess`                             | `subprocess` (launch_server.sh on the host) or `docker` (run.py container) |
| `TT_INFERENCE_SERVER_DIR` | `../tt-inference-server`            | Checkout holding `run.py` (docker mode) |
| `TT_INFERENCE_PY`   | `<TT_INFERENCE_SERVER_DIR>/venv/bin/python` | Python that runs `run.py` (docker mode) |
| `TT_COMFYUI_IMAGE`  | `comfyui-media-server:dev`               | Image passed to `--override-docker-image` (docker mode) |
| `TT_RUNPY_EXTRA_ARGS` | (empty)                                | Extra args appended to `run.py` (e.g. weight mounts) |
| `TT_METAL_DIR`      | `../tt-metal` (sibling of ComfyUI)       | tt-metal checkout that holds `launch_server.sh` (subprocess mode) |
| `TT_SMI_BIN`        | `tt-smi` resolved from `PATH`            | tt-smi console script (board reset / detection) |
| `TT_SERVER_HOST`    | `127.0.0.1`                              | Host the tt-metal server binds / is reached on |
| `TT_SERVER_PORT`    | `8000`                                   | Port for the tt-metal server |
| `TT_SDXL_BOARD`     | `p150`                                   | Board passed to `launch_server.sh` for SDXL |
| `TT_WAN22_BOARD`    | `p300x2`                                 | Board passed to `launch_server.sh` for Wan 2.2 |
| `TT_SERVER_READY_TIMEOUT` | `1800` (seconds)                   | How long to wait for `/health` during warmup |
| `TT_SERVER_PID_FILE`| `/tmp/tt_comfy_server.pid`               | Lock file used to reap an orphaned server |

## Launching ComfyUI

From the ComfyUI repo root:

```bash
./launch_with_http.sh            # defaults: port 8188, listen 127.0.0.1
./launch_with_http.sh --port 8188 --listen 0.0.0.0
```

This activates the ComfyUI venv and runs `main.py --tenstorrent`. You do **not**
start the tt-metal server yourself — pick a model in the **TT Checkpoint Loader**
node and it will spawn and supervise the server for you. On exit, the launcher
backstops a kill of any tt-metal server the node left running.

## Nodes

| Node | Category | Purpose |
|------|----------|---------|
| **TT Checkpoint Loader** (`TT_CheckpointLoader`) | Tenstorrent | Stand up a tt-metal model (auto-launch server) and return `MODEL` / `CLIP` / `VAE` handles. Inputs: `model_type` (`sdxl`, `wan22`, `ltx` or `ltx_pro`); optional `board` override and `server_url` (connect to an already-running server instead of auto-standup). |
| **TT LoRA Loader** (`TT_LoraLoader`) | Tenstorrent | Attach a LoRA with separate UNet (`strength_model`) and CLIP (`strength_clip`) scales. Returns `MODEL` / `CLIP`. |
| **TT Wan LoRA Loader** (`TT_WanLoraLoader`) | Tenstorrent/video | Attach per-expert Wan 2.2 LoRA paths (high/low) applied server-side. Returns `MODEL`. |
| **TT KSampler** (`TT_KSampler`) | Tenstorrent/sampling | Run SDXL denoising on the server; returns `LATENT`. |
| **TT VAE Decode** (`TT_VAEDecode`) | Tenstorrent/latent | Decode latents to images using the tt-metal VAE (SDXL and Wan 2.2). |
| **TT VAE Encode** (`TT_VAEEncode`) | Tenstorrent/latent | Encode images to latents using the tt-metal VAE. |
| **TT Wan Sampler** (`TT_WanSampler`) | Tenstorrent/video | Run Wan 2.2 denoising; returns a video `LATENT` for `TT_VAEDecode`. |
| **TT Text To Video** (`TT_TextToVideo`) | Tenstorrent/video | One-shot Wan 2.2 text-to-video; returns image frames. |
| **TT LTX Video (AV)** (`TT_LTXVideo`) | Tenstorrent/video | One-shot LTX-2.3 text-to-audio+video; returns a native `VIDEO` (muxed h264 + AAC) for `Save Video`. Clip geometry and step count are fixed by the running server. The negative input is accepted but ignored — the distilled pipeline has no CFG. |
| **TT LTX Video Pro (AV)** (`TT_LTXVideoPro`) | Tenstorrent/video | Guided one-stage LTX-2.3. Same `VIDEO` output, but takes `steps`, `video_cfg` / `audio_cfg`, `video_stg` / `audio_stg`, `stg_block`, and a **live** negative prompt. Several times slower than the distilled node. Needs `model_type=ltx_pro`. |
| **TT LTX LoRA Loader** (`TT_LTXLoraLoader`) | Tenstorrent/video | Attach an LTX-2.3 LoRA, applied on device server-side. **Chainable** — wire several in series to stack them, each with its own strength. Adapters live in `models/loras/ltx/`. |
| **TT Preview Video** (`TT_PreviewVideo`) | Tenstorrent/video | Show a `VIDEO` in the graph without writing to `output/`. The video counterpart of `Preview Image`: writes to ComfyUI's temp directory and renders a player. Never re-encodes. Works with any `VIDEO`, not just the TT nodes. |
| **TT Model Info** (`TT_ModelInfo`) | Tenstorrent/utils | Display information about a TT model handle. |
| **TT Unload Model** (`TT_UnloadModel`) | Tenstorrent/utils | Stop the tt-metal server; optionally reset all Tenstorrent boards. |

### Example: SDXL text-to-image

```
[TT Checkpoint Loader (sdxl)] ─model─▶ [TT KSampler] ─samples─▶ [TT VAE Decode] ─▶ [Save Image]
            │ clip ─▶ [CLIP Text Encode] ─▶ TT KSampler (positive/negative)
            └ vae  ─────────────────────────────────────────▶ TT VAE Decode
```

### Example: Wan 2.2 text-to-video

```
[TT Checkpoint Loader (wan22)] ─model─▶ [TT Wan Sampler] ─samples─▶ [TT VAE Decode] ─▶ [Save / VHS combine]
   (optional) └─▶ [TT Wan LoRA Loader] ─▶ TT Wan Sampler
```

### Example: LTX-2.3 text-to-audio+video

```
[TT Checkpoint Loader (ltx)] ─model─▶ [TT LTX Video (AV)] ─video─▶ [TT Preview Video]
            └ clip ─▶ [CLIP Text Encode] ×2 ─▶ TT LTX Video      └─────▶ [Save Video]
```

`TT Preview Video` shows the clip in the graph without writing to `output/`;
`Save Video` is the keeper. The shipped workflow has the preview active and
`Save Video` muted.

Unlike the Wan graph there is no separate decode step: LTX-2.3 decodes video and
audio together and muxes them server-side, so the node hands `Save Video` a
finished clip. The `vae` output of the loader is unused here.

Everything runs on device, text encoding included — LTX's text encoder is
Gemma-3-12B, which is why the prompt travels to the server as a string rather
than as embeddings computed on the host.

### LoRA on LTX

Adapters go in `ComfyUI/models/loras/ltx/` and must be built for LTX-2.3. Chain
`TT LTX LoRA Loader` nodes to stack them:

```
[TT Checkpoint Loader (ltx_pro)] ─model─▶ [TT LTX LoRA Loader]  ─▶ [TT LTX LoRA Loader]  ─▶ [TT LTX Video Pro]
                                              style @ 0.6              distillation @ 1.0
```

Stacking is the point rather than a nicety. LTX-2.3 style adapters are trained
against the **dev** checkpoint, so they do not work on the distilled one — and
`ltx_pro` is the server that runs dev. The documented way to get a custom look
*and* the distilled step count is a style adapter at its normal strength
together with Lightricks' official distillation adapter at 1.0, sampled at
**8 steps with CFG 1** (set `steps`, `video_cfg` and `audio_cfg` on
`TT LTX Video Pro`).

### The two Pro guidance profiles

`TT LTX Video Pro` has five guidance knobs and **their neutral values differ**, which
is the easy mistake. Every term left enabled costs an extra transformer forward per
step, so a run meant to be unguided can silently cost 3-4x what it should.

| | reference (no LoRA) | distilled (with the distillation LoRA) |
|---|---|---|
| `steps` | 30 | 8 |
| `video_cfg` / `audio_cfg` | 3.0 / 7.0 | 1.0 / 1.0 |
| `video_stg` / `audio_stg` | 1.0 / 1.0 | **0.0 / 0.0** |
| `video_modality` / `audio_modality` | 3.0 / 3.0 | **1.0 / 1.0** |
| `rescale` | 0.7 | **0.0** |
| forwards per step | 4 | 1 |

`cfg` and `modality` disable at **1.0**; `stg` and `rescale` disable at **0**. Setting
them all to 0 does not turn guidance off -- it enables modality guidance at -1. The node
logs how many forwards per step your settings imply, so check that line if a run is
slower than you expected.

### An empty negative prompt is the *strongest* anti-style setting

Leaving `TT LTX Video Pro`'s negative empty does **not** mean "no negative": the
server substitutes the pipeline's 58-term photoreal default, which includes
"cartoonish rendering, 3D CGI look, unrealistic materials, distorted proportions".
At CFG > 1 that actively pushes the sample away from stylized output. If you are
using a style adapter, supply a short neutral negative (e.g. `blurry, watermark,
text`) or run the distilled profile (CFG 1), where the negative is inert.

### Style adapters: a trigger word alone is not enough

Style adapters work, but only if the prompt already *describes* the style in
plain language. The trigger token by itself does almost nothing.

Look at how these adapters' author writes a prompt in their own published
workflow: the trigger comes first, then a full style sentence, then the scene.

```
f4nt4sy4n1m6, cinematic fantasy anime cel-shaded illustration with hand-drawn
linework, painted shading, vivid magical atmosphere, and stylized fantasy
character design. <the actual scene>
```

Each model card has an "Other Trigger Words That Help" list -- that list is the
style sentence. Use most of it, not just the token.

Measured on the distilled profile, seed 42, identical scene text in every arm:

| prompt | LoRA | result |
|---|---|---|
| scene only | none | photoreal forest, real fox |
| `Pap3rCut0u7` + paper style sentence | none | photoreal forest containing a *cardboard prop* |
| `Pap3rCut0u7` + paper style sentence | papercut @1.0 | **full paper-cutout diorama** -- layered paper trees, cut-paper fox, flat paper sky |
| `P1x4r` + pixar style sentence | none | flat, rubbery cartoon fox |
| `P1x4r` + pixar style sentence | pixar @1.0 | **film-grade stylized character** -- groomed fur, eye caustics, believable stylized anatomy |

The no-LoRA rows are the point: the words alone get you a photoreal scene *of*
paper, or a cheap-looking cartoon. The adapter is what makes the whole frame
render in the material, and what lifts the toon from rubbery to polished.

Earlier testing that gave these adapters only their trigger token concluded they
were inert. That conclusion was wrong -- the prompt was.

The on-device math was verified independently of any of this:
`PCC(W_after - W_before, B@A)` of 0.987-0.9998 on every module kind including
cross-attention Q and the attention gate, zero targets skipped or deferred, and
exact parity with the reference loader's `W + strength * B@A`.

### Geometry

`launch_server.sh --model ltx --height 1088 --width 1920 --frames 121` overrides
the pinned clip shape at launch (multiples of 64; frames-1 divisible by 8).
Geometry cannot change per request.

### Trigger words are not free variables

Most style adapters need a trigger token in the prompt. Some triggers carry meaning of
their own -- `crtanim` reads as "CRT animation", so the base model responds to it even
with no adapter loaded. If you are comparing with-adapter against without-adapter, put
the trigger **only** in the with-adapter prompt, or the comparison is confounded.
Adapters with a deliberately meaningless trigger (for example `P1x4r`) avoid the problem.

Three things the node cannot enforce for you:

- **Exactly one distillation adapter.** Two double-apply and overshoot.
- **A distillation adapter only shows its effect at the settings it was
  calibrated for.** At 30 steps and CFG 3/7 it will look wrong.
- **The adapter must be for LTX-2.3.** An SDXL or Wan file has no key that maps
  onto an LTX module; the server reports it as skipped rather than failing the
  run, so check the node log if an adapter seems to do nothing.

Changing a strength re-binds on device and does **not** reload the adapter, so
tweaking and re-queueing is cheap even for the official distillation adapter,
which is 7.6 GB.

### distilled vs Pro

They are two different checkpoints, so they are two different `model_type`s and
switching between them relaunches the server.

| | `ltx` (distilled) | `ltx_pro` (one-stage) |
|---|---|---|
| Node | `TT_LTXVideo` | `TT_LTXVideoPro` |
| Checkpoint | `ltx-2.3-22b-distilled-1.1` | `ltx-2.3-22b-dev` |
| Steps | 11, fixed in the sigma schedules | `steps` widget, 30 by default |
| Guidance | none | CFG + STG, all exposed |
| Negative prompt | inert | live |
| 241f @ 576x1024 | ~38 s | ~254 s |

`load_tt_ltx_standalone.json` in `user/default/workflows/` has both branches plus
a muted `TT Kill Server`. The Pro branch ships muted (mode 4, the same convention
the SDXL workflows use for `TT_UnloadModel`) so a plain Queue runs only the fast
path — unmute Pro and mute distilled to switch, and expect a server relaunch when
you do.

## Troubleshooting

**`launch_server.sh not found at <path>`** — set `TT_METAL_DIR` to your tt-metal
checkout, or place tt-metal as a sibling of ComfyUI.

**Server never becomes healthy / times out** — first warmup can take up to ~25
min for Wan 2.2 (trace capture). Watch the server log at
`<tt-metal>/<model>_server_comfy.log`. Raise `TT_SERVER_READY_TIMEOUT` if needed.

**`tt-smi` not found** — ensure `tt-smi` is on `PATH` (installed by
`tt-installer`) or set `TT_SMI_BIN` to its console script. Used for the board
reset option of TT Unload Model and for device detection in `launch_server.sh`.

**Port already in use** — set `TT_SERVER_PORT` (and make sure nothing else holds
`:8000`). The ComfyUI web UI port (`--port`, default `8188`) is independent.

**Switching models is slow** — expected: the single-model server is torn down and
relaunched on each `model_type` change.

**LTX clip is the wrong length or resolution** — the node has no geometry
widgets on purpose. LTX-2.3's latent upsampler pins its `GroupNorm` to a fixed
`T*H*W` when the server builds the pipeline, so the shape is chosen at server
launch, not per request; a mismatched request is rejected rather than silently
resized. Relaunch the server with different `--frames` / `--height` / `--width`.

**LTX negative prompt has no effect** — correct, and not a wiring fault. The
distilled pipeline runs without CFG, so there is no unconditional pass for a
negative prompt to push against. The input exists for graph compatibility.

**Cancelling an LTX generation** — the node aborts at its next progress event,
so the graph stops promptly, but the server finishes the generation already in
flight. The next queued request waits for it.

**An LTX LoRA appears to do nothing** — check the server log for
`could not be loaded` or a skipped adapter. The usual causes are an adapter
built for a different model, or a distillation adapter run at the wrong step
count and CFG (it needs 8 steps / CFG 1). Note also that the `ltx` dropdown
lists only what is in `models/loras/ltx/`; if that folder is missing entirely
the node falls back to listing every SDXL and Wan adapter, none of which can
bind here.

**Previews disappear after a restart** — expected. `TT Preview Video` writes to
ComfyUI's temp directory, which is cleared on startup. Unmute `Save Video` for
anything you want to keep, or download it from the player.

**Why not VHS Video Combine?** It is an encoder: its only video input is
`images` (an `IMAGE` frame batch) which it feeds to ffmpeg, so it cannot accept
an already-encoded clip. Bridging with `Get Video Components` works, but it
decodes the MP4 to ~1.7GB of float32 frames and then re-encodes them. The TT
path avoids that end to end -- the server encodes once and the bytes are passed
through untouched, so `TT Preview Video` writes them verbatim and `Save Video`
only remuxes (packet copy) to attach metadata. Keep `format` and `codec` on
`auto` in `Save Video`: forcing a mismatched pair makes it decode and re-encode.

## License

SPDX-License-Identifier: Apache-2.0

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
