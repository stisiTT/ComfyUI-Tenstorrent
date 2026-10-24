# TT ComfyUI: integration, versions, and how to run

This is the source of truth for how the pieces fit together and which tt-metal
to run. If you only want to *use* the nodes, read `README.md` and `INSTALL.md`;
this file is for keeping the moving parts in sync.

Last updated: 2026-10-24.

## The two repos and how they connect

```
ComfyUI-Tenstorrent (this repo, in ComfyUI/custom_nodes/)        tt-metal (built checkout)
  server_manager.py ── spawns ──▶ media_server/launch_server.sh ──▶ python_env
  nodes.py          ── HTTP  ──▶ media_server/server.py ──▶ runners ──▶ ttnn / models.* ──▶ TT hardware
```

- The nodes **never import tt-metal**. They start the bundled server as a
  subprocess and talk to it over HTTP (`/health`, `/image/generations`,
  `/video/av_generations`, `/latent/denoise`, `/vae/encode`, `/vae/decode`,
  `/video/*`).
- The server ships in this repo under `media_server/` and runs in tt-metal's
  `python_env`, importing `ttnn` and `models.*` from the checkout in
  `TT_METAL_HOME` (default `../tt-metal`, next to ComfyUI).
- Nodes and server change together in one commit, so there is no version
  matrix between them. The only pin is the tt-metal branch.

The server lived in tt-inference-server's `comfyui-media-server` branch (and
before that at the root of a tt-metal branch) until 0.2.0, when it moved here
with its history.

## Which tt-metal to run

| Repo | Branch | Notes |
|------|--------|-------|
| ComfyUI-Tenstorrent | `main` (stisiTT/ComfyUI-Tenstorrent) | These nodes plus `media_server/`, cloned into `custom_nodes/` of stock Comfy-Org/ComfyUI. |
| tt-metal | `stisi/comfyui_20261023_main`, commit pinned in [`TT_METAL_REF`](TT_METAL_REF) | PR #47509's branch, the LTX-2.3 QuietBox 2 / LoRA / progress-event work, and a MiniMax-H3 SP=1 fix, merged with tt-metal `main` of 2026-10-23. Must be **built** (`./build_metal.sh`, `./create_venv.sh`). |

The server calls `fuse_lora(lora_scale, clip_scale)`, the signature PR #47509
settled on after review. A tt-metal without it fails the first SDXL LoRA request
with `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'`.
The HTTP request body still carries `lora_scale_unet` / `lora_scale_clip`.

### The tt-metal PRs the server depends on

The runners import library code that is being upstreamed as separate PRs. The
branch above contains all of them. Status as of 2026-10-24:

| PR | What | Status |
|----|------|--------|
| [#46519](https://github.com/tenstorrent/tt-metal/pull/46519) | Wan 2.2 LoRA (Parshwa's) | merged |
| [#47265](https://github.com/tenstorrent/tt-metal/pull/47265) | Wan 2.2 per-request guidance (`flow_shift`, `boundary_ratio`, `guidance_scale`) | merged |
| [#47715](https://github.com/tenstorrent/tt-metal/pull/47715) | Wan per-step progress events (the ComfyUI progress bar) | merged |
| [#56674](https://github.com/tenstorrent/tt-metal/pull/56674) | LTX-2.3 per-stage transformer state fix | merged |
| [#47509](https://github.com/tenstorrent/tt-metal/pull/47509) | SDXL per-component LoRA (UNet/CLIP scales + text-encoder LoRA) | open, critical path; waiting on a reply to review |
| [#56585](https://github.com/tenstorrent/tt-metal/pull/56585) | Hang loading a tensor over 32 MiB from a cache file | open, needs a rebase |
| [#48616](https://github.com/tenstorrent/tt-metal/pull/48616) | DeviceClass primitive | closed; `media_server/device_specs.py` keeps its own copy |
| [#47510](https://github.com/tenstorrent/tt-metal/pull/47510) | Standalone media server | closed; the server lives in this repo |

Not in any PR yet: LTX-2.3 on a QuietBox 2 (2x2 mesh), LTX progress events,
several LoRAs bound at once, per-modality CFG rescale for LTX, and the MiniMax-H3
prompt padding fix for meshes without sequence parallelism.

## How to run

```bash
export HF_HOME=/path/to/hf_cache      # optional: so weights don't download on first request
cd <ComfyUI>/custom_nodes/ComfyUI-Tenstorrent
./scripts/launch_with_http.sh --port 8188 --listen 127.0.0.1
```

Then pick a model in the **TT Checkpoint Loader** node. It runs
`media_server/launch_server.sh --model <model> --board <board>`, polls
`/health`, and the nodes' requests hit `server.py`. Board defaults: SDXL →
`p150`; Wan 2.2, LTX-2.3 and MiniMax-H3 → `p300x2` (override with
`TT_SDXL_BOARD` / `TT_WAN22_BOARD` / `TT_LTX_BOARD` / `TT_H3_BOARD`). SD3.5
ignores the board and needs an 8-chip Wormhole LoudBox.

`/health` reports the package version the server shipped with. The loader logs
a warning when it differs from the nodes' version, which means the server came
from another checkout (`server_url` or `TT_SERVER_DIR`). See
[`INSTALL.md`](INSTALL.md) for the full setup and first-run times, and
[`media_server/README.md`](media_server/README.md) to run the server by hand.

## What's still missing

Open work is tracked in [`NEXT_STEPS.md`](NEXT_STEPS.md).
