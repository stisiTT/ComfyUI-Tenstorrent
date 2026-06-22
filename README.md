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
3. **Model weights** reachable by the tt-metal server (downloaded into its
   `HF_HOME` / ttnn model cache).

## Configuration (environment variables)

All read by `server_manager.py`; every default is overridable:

| Variable            | Default                                  | Purpose |
|---------------------|------------------------------------------|---------|
| `TT_METAL_DIR`      | `../tt-metal` (sibling of ComfyUI)       | tt-metal checkout that holds `launch_server.sh` |
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
| **TT Checkpoint Loader** (`TT_CheckpointLoader`) | Tenstorrent | Stand up a tt-metal model (auto-launch server) and return `MODEL` / `CLIP` / `VAE` handles. Inputs: `model_type` (`sdxl` or `wan22`); optional `board` override and `server_url` (connect to an already-running server instead of auto-standup). |
| **TT LoRA Loader** (`TT_LoraLoader`) | Tenstorrent | Attach a LoRA with separate UNet (`strength_model`) and CLIP (`strength_clip`) scales. Returns `MODEL` / `CLIP`. |
| **TT Wan LoRA Loader** (`TT_WanLoraLoader`) | Tenstorrent/video | Attach per-expert Wan 2.2 LoRA paths (high/low) applied server-side. Returns `MODEL`. |
| **TT KSampler** (`TT_KSampler`) | Tenstorrent/sampling | Run SDXL denoising on the server; returns `LATENT`. |
| **TT VAE Decode** (`TT_VAEDecode`) | Tenstorrent/latent | Decode latents to images using the tt-metal VAE (SDXL and Wan 2.2). |
| **TT VAE Encode** (`TT_VAEEncode`) | Tenstorrent/latent | Encode images to latents using the tt-metal VAE. |
| **TT Wan Sampler** (`TT_WanSampler`) | Tenstorrent/video | Run Wan 2.2 denoising; returns a video `LATENT` for `TT_VAEDecode`. |
| **TT Text To Video** (`TT_TextToVideo`) | Tenstorrent/video | One-shot Wan 2.2 text-to-video; returns image frames. |
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

## License

SPDX-License-Identifier: Apache-2.0

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
