# TT ComfyUI: integration, versions, and how to run

This is the source of truth for how the pieces fit together and which tt-metal
to run. If you only want to *use* the nodes, read `README.md` and `INSTALL.md`;
this file is for keeping the moving parts in sync.

Last updated: 2026-10-23.

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
| tt-metal | `stisi/comfyui_20261023_main` | PR #47509's branch, the LTX-2.3 QuietBox 2 / LoRA / progress-event work, and a MiniMax-H3 SP=1 fix, merged with tt-metal `main` of 2026-10-23. Must be **built** (`./build_metal.sh`, `./create_venv.sh`). |

The server calls `fuse_lora(lora_scale, clip_scale)`, the signature PR #47509
settled on after review. A tt-metal without it fails the first SDXL LoRA request
with `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'`.
The HTTP request body still carries `lora_scale_unet` / `lora_scale_clip`.

### The tt-metal PRs the server depends on

The runners import library code that is being upstreamed as separate PRs. The
branch above contains all of them. Status as last checked:

| PR | What | Status |
|----|------|--------|
| [#46519](https://github.com/tenstorrent/tt-metal/pull/46519) | LoRA (Parshwa's) | merged |
| [#47509](https://github.com/tenstorrent/tt-metal/pull/47509) | SDXL per-component LoRA (UNet/CLIP scales + text-encoder LoRA) | open, critical path |
| [#47265](https://github.com/tenstorrent/tt-metal/pull/47265) | Wan 2.2 per-request guidance (`flow_shift`, `boundary_ratio`, `guidance_scale`) | open |
| [#47715](https://github.com/tenstorrent/tt-metal/pull/47715) | DenoiseStep per-step progress events (the ComfyUI progress bar) | open |
| [#48616](https://github.com/tenstorrent/tt-metal/pull/48616) | DeviceClass primitive | open |
| [#47510](https://github.com/tenstorrent/tt-metal/pull/47510) | Standalone media server | **not going into tt-metal**; the server lives in this repo |

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
`TT_SDXL_BOARD` / `TT_WAN22_BOARD` / `TT_LTX_BOARD` / `TT_H3_BOARD`). See
[`INSTALL.md`](INSTALL.md) for the full setup and first-run times, and
[`media_server/README.md`](media_server/README.md) to run the server by hand.

## What's still missing

1. **tt-metal PRs.** Until #47509 and the LTX/H3 work land on `main`, the
   nodes need the pinned branch above, and it has to be re-merged with `main`
   by hand to pick up new models and fixes.
2. **No HTTP contract version.** Nodes and server now ship together, which
   covers the normal case, but a hand-started server from another checkout
   (`server_url`, `TT_SERVER_DIR`) can still drift. Consider surfacing a
   version in `/health`.
3. **`sd35` is server-supported but not node-exposed.** `server.py` handles
   `sd35`, but `TT_CheckpointLoader`'s `model_type` does not offer it. Add it if
   SD3.5 is meant to be usable from the UI.
4. **Convergence with tt-media-server.** tt-inference-server's OpenAI-style
   `tt-media-server` serves more models on more hardware but has none of the
   staged ops, LoRA stacking or progress streaming the nodes use. Upstreaming
   those, or teaching the nodes to talk to it for plain generation, is the
   longer-term path.
