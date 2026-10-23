# Installing ComfyUI + Tenstorrent

This guide sets up ComfyUI with the Tenstorrent custom nodes, which run **Stable
Diffusion XL** and **SD 3.5** (image), **Wan 2.2**, **LTX-2.3** and **MiniMax-H3**
(video, the last two with audio) on Tenstorrent hardware via a tt-metal
inference server.

However many repos are involved, it is always **two separate Python
environments** talking over a local HTTP socket. That isolation is intentional:
tt-metal's native/NumPy ABI must never mix with ComfyUI's.

```
ComfyUI + this package (ComfyUI venv)         media server (runs in tt-metal's python_env)
  main.py --cpu                                  server.py  (FastAPI media server)
   └ TT_CheckpointLoader                          └ launch_server.sh
        └ spawns ──────────────────Popen────────────▶ launch_server.sh
        └ talks ───────────────HTTP :8000───────────▶ server.py ──▶ Tenstorrent device
```

Where that server code lives is the only thing that differs between the two
stacks in [section 2](#2-pick-a-stack-then-clone): the tt-metal repo root on
Stack 1, or tt-inference-server on Stack 2. It runs under tt-metal's
`python_env` either way.

You pick a model in the **TT Checkpoint Loader** node; it stands up and supervises
the tt-metal server for you. You do not launch the server by hand.

---

## 1. Prerequisites

- **A working Tenstorrent stack.** Drivers (TT-KMD), firmware, hugepages, and
  `tt-smi` must already be installed and `tt-smi` must be on your `PATH`. If you
  have not done this, follow Tenstorrent's
  [hardware setup](https://docs.tenstorrent.com) /
  [TT-Installer](https://github.com/tenstorrent/tt-installer) first, then come back.
  Verify with:
  ```bash
  tt-smi            # should list your device(s)
  ```
- **Supported hardware for this guide:**
  | Model   | Board (default)        | Notes |
  |---------|------------------------|-------|
  | SDXL    | `p150` (1 chip; uses up to 4) | image generation |
  | Wan 2.2 | `p300x2` (QuietBox 2)  | text-to-video |
  | LTX-2.3 | `p300x2` (QuietBox 2)  | text-to-video with audio |
  | MiniMax-H3 | `p300x2` (QuietBox 2) | text- or image-to-video with audio |

  Other boards can be selected with `TT_SDXL_BOARD` / `TT_WAN22_BOARD` /
  `TT_LTX_BOARD` / `TT_H3_BOARD` or the node's `board` input, but the ones
  above are the tested paths.
- **OS / tooling:** Ubuntu 24.04, **Python 3.12**, `git`, and
  [`uv`](https://github.com/astral-sh/uv) (used by tt-metal's venv script):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Disk/RAM:** building tt-metal is heavy (several GB of build artifacts, lots of
  RAM). Model weights add tens of GB on first run.

---

## 2. Pick a stack, then clone

There are two supported layouts. They differ only in **where the media server
lives**; the nodes behave identically on both.

| | server (`TT_METAL_DIR`) | tt-metal (`TT_METAL_HOME`) | `fuse_lora` |
|---|---|---|---|
| **Stack 1** (legacy) | tt-metal repo root | `samt/standalone-media-20260703` @ `ce05994325a` | `(lora_scale_unet, lora_scale_clip)` |
| **Stack 2** (current) | tt-inference-server `comfyui-media-server/` @ `6a4085df0` | `stisi/sdxl-per-component-lora` | `(lora_scale, clip_scale)` |
| **Stack 2 + LTX-2.3 / MiniMax-H3** | tt-inference-server `comfyui-media-server` | `stisi/comfyui_20261023_main` | `(lora_scale, clip_scale)` |

**This guide follows Stack 2**, the relocated server. That is where the work is
headed and what the container path is built from.

> **Do not mix the stacks.** `fuse_lora`'s signature differs between the two
> tt-metal branches, so the server copy and the `TT_METAL_HOME` checkout must
> come from the same row. Pairing the relocated server with
> `samt/standalone-media-20260703` raises `TypeError: fuse_lora() got an
> unexpected keyword argument 'clip_scale'` on the first LoRA request. Either
> way the HTTP contract is unchanged: the request body always carries
> `lora_scale_unet` / `lora_scale_clip`.

> **For LTX-2.3 and MiniMax-H3 use the third row.** Its two branches are
> descendants of Stack 2's, so the `fuse_lora` rule above still holds, but the
> LTX and H3 servers (`--model ltx`, `--model ltx_pro`, `--model h3`) and the
> LTX LoRA node exist only there. LTX adapters go in
> `models/loras/ltx/`; that directory must exist, or the node's dropdown falls
> back to listing adapters for other models, none of which can bind.

Clone stock ComfyUI and the two Tenstorrent repos as siblings, then clone this
package into ComfyUI's `custom_nodes/`:

```bash
cd ~/src                 # or wherever you keep checkouts
git clone https://github.com/Comfy-Org/ComfyUI.git
git clone https://github.com/tenstorrent/tt-metal.git
git clone https://github.com/tenstorrent/tt-inference-server.git
git clone https://github.com/stisiTT/ComfyUI-Tenstorrent.git ComfyUI/custom_nodes/ComfyUI-Tenstorrent
```

Resulting layout:
```
~/src/ComfyUI
~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent
~/src/tt-metal
~/src/tt-inference-server
```

ComfyUI itself is unmodified; everything Tenstorrent-specific is in this
package.

> For **Stack 1**, omit tt-inference-server: `TT_METAL_DIR` defaults to
> `../tt-metal`, whose repo root holds `launch_server.sh`, so the sibling layout
> just works. Either way, set `TT_METAL_DIR` explicitly if you put the server
> somewhere else.

---

## 3. Build tt-metal (pinned branch)

The nodes need per-component LoRA, which is **not on tt-metal's `main`** yet:
it is [PR #47509](https://github.com/tenstorrent/tt-metal/pull/47509), still
open. Check out that branch, init submodules, then build.

```bash
cd ~/src/tt-metal
git checkout stisi/comfyui_20261023_main       # LTX-2.3 + MiniMax-H3; PR #47509's branch merged with main
git submodule update --init --recursive

# System build dependencies (uses sudo; one-time).
./install_dependencies.sh

# Build the library.
./build_metal.sh

# Create the tt-metal Python environment (./python_env, via uv).
./create_venv.sh
```

Then put the server repo on its branch. Nothing to build here, it is Python only:

```bash
cd ~/src/tt-inference-server
git checkout comfyui-media-server              # comfyui-media-server/ with the LTX and H3 servers
```

> **Why a branch and not `main`?** PR #47509 is on the critical path and has not
> merged. Until it does, the server's `fuse_lora(lora_scale, clip_scale)` call
> only works against this branch. Once it lands you can track a release instead.
>
> For **Stack 1** the tt-metal branch is `samt/standalone-media-20260703` and
> there is no tt-inference-server step, since that branch carries its own copy of
> the server at the repo root.

You do **not** need to activate `python_env` yourself or install the server's
`fastapi`/`uvicorn` deps — `launch_server.sh` does that automatically when the
node starts the server.

---

## 4. Set up the ComfyUI environment

The launcher expects the virtualenv at `ComfyUI/venv`.

```bash
cd ~/src/ComfyUI
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r custom_nodes/ComfyUI-Tenstorrent/requirements.txt
```

> **torch note:** compute runs on the Tenstorrent device, not on a GPU, so the
> CPU build of torch is all ComfyUI needs. To avoid pulling multi-GB CUDA wheels:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> (run this before the two `pip install -r` lines).

---

## 5. Launch

```bash
cd ~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent
./scripts/launch_subprocess_mode.sh   # Stack 2; ComfyUI on http://127.0.0.1:8188
# options are passed straight through: --port 8188  --listen 127.0.0.1
```

> `launch_subprocess_mode.sh` exports
> `TT_METAL_DIR=<parent>/tt-inference-server/comfyui-media-server` and
> `TT_METAL_HOME=<parent>/tt-metal`, then execs `launch_with_http.sh`. On
> **Stack 1** run `./scripts/launch_with_http.sh` directly instead: no environment
> variables are needed, since `TT_METAL_DIR` defaults to `../tt-metal`.
>
> Keep `--listen` on `127.0.0.1` unless you trust every host on the network.
> The server accepts a client-supplied `lora_path` and loads it from disk, so
> exposing the UI exposes that too.
>
> There is also a containerized path (`TT_LAUNCH_MODE=docker`) that runs the
> server through tt-inference-server's `run.py`; it needs an image built first,
> so it is opt-in. Both stacks and both launch paths are documented in
> [`INTEGRATION.md`](INTEGRATION.md).

This activates the ComfyUI venv and runs `main.py --cpu` (the device does the
compute; ComfyUI only orchestrates). Open the web UI,
build a graph starting from **TT Checkpoint Loader**, and pick a model:

- `sdxl` → image graph: `TT KSampler` → `TT VAE Decode` → Save Image
- `wan22` → video graph: `TT Wan Sampler` → `TT VAE Decode` → Create Video
- `ltx` / `ltx_pro` → `TT LTX Video` / `TT LTX Video Pro` → `TT Preview Video`
- `h3` → `TT H3 Video` (optionally fed a first and/or last frame) → `TT Preview Video`

Or start from one of the example workflows in the next section. See
[`README.md`](README.md) for the full node reference and the
environment-variable table.

---

## 6. Example workflows

The package ships ready-made graphs in `example_workflows/`. ComfyUI lists them
under **Templates** in the workflow menu, grouped as **ComfyUI-Tenstorrent**.
Opening a template gives you an unsaved copy; save it to keep your edits.

| Workflow | Model | Needs |
|---|---|---|
| `tt_sdxl_photoshop` | SDXL txt2img | nothing extra |
| `tt_sdxl_img2img` | SDXL img2img | an input image, uploaded in the Load Image node |
| `tt_sdxl_img2img_photoshop` | SDXL img2img | the SD-PPP plugin (below) |
| `tt_shutdown_reset` | any | nothing; stops the server and resets the boards |
| `load_tt_wan_standalone` | Wan 2.2 | nothing extra |
| `load_tt_wan_standalone_lora` | Wan 2.2 + Lightning | Wan LoRA files (below) |
| `load_tt_ltx_standalone` | LTX-2.3 | nothing extra for the distilled path |
| `load_tt_ltx_matrix`, `load_tt_ltx_lora_gallery` | LTX-2.3 + style LoRAs | LTX LoRA files (below) |
| `load_tt_h3_standalone` | MiniMax-H3 text to video | nothing extra |
| `load_tt_h3_first_frame` | MiniMax-H3 image to video | a still image, uploaded in the Load Image node (an SDXL render works well) |
| `tt_h3_cat_sunbeam` | MiniMax-H3 text to video | nothing extra; a 6 s showcase prompt in the guide format H3 was trained on (description, soundscape, music) |

Video graphs end in core **Create Video** → **TT Preview Video**, which plays
the clip in the graph without writing a file. To keep the clip, swap in core
**Save Video**. No third-party node packs are needed.

**LoRA files.** Put them under `ComfyUI/models/loras/`. The node dropdowns list
what is there, filtered by folder.

- Wan Lightning, from
  [`lightx2v/Wan2.2-Lightning`](https://huggingface.co/lightx2v/Wan2.2-Lightning):
  ```bash
  huggingface-cli download lightx2v/Wan2.2-Lightning \
      --include "Wan2.2-T2V-A14B-4steps-lora-250928/*.safetensors" \
      --local-dir ComfyUI/models/loras/wan22/lightning
  ```
- LTX-2.3, into `ComfyUI/models/loras/ltx/` (create the folder first; see the
  note in [section 2](#2-pick-a-stack-then-clone)). The example graphs expect
  these file names, so download and rename (or symlink) them:

  | File name in `loras/ltx/` | HuggingFace source |
  |---|---|
  | `ltx-2.3-22b-distilled-lora-384-1.1.safetensors` | `Lightricks/LTX-2.3` (same name) |
  | `pixar_toon_P1x4r.safetensors` | `vrgamedevgirl84/LTX_2.3_Pixar_Toon_Style_LoRa` → `Pixar_Toon.safetensors` |
  | `papercut_Pap3rCut0u7.safetensors` | `vrgamedevgirl84/LTX_2.3_Paper_Cut_Out_Style_LoRa` → `PaperCutOutStyle.safetensors` |
  | `anim90s_9o4n1m.safetensors` | `vrgamedevgirl84/LTX_2.3_90s_Animation_Style_LoRa` → `90sAnimationStyle.safetensors` |

  The suffix in each name is the adapter's trigger word, which the example
  prompts already start with.

**Photoshop.** The `*_photoshop` workflows are driven from Photoshop by the
[SD-PPP](https://sdppp.zombee.tech/) 2.0 plugin (Photoshop 2025 / PS 26.0+).
SD-PPP 2.0 needs no ComfyUI custom node. It lists the workflows saved in
ComfyUI's workflow library (`ComfyUI/user/default/workflows/`), not templates,
so open each Photoshop template and save it first, or copy its JSON from
`example_workflows/` into that folder. In the panel, node titles starting with
`#` are shown as controls and `.` are hidden.

---

## 7. First run: weights + warmup

- **Weights download automatically** from HuggingFace on first use, into
  `HF_HOME` (default `~/.cache/huggingface`):
  - SDXL → `stabilityai/stable-diffusion-xl-base-1.0`
  - Wan 2.2 → `Wan-AI/Wan2.2-T2V-A14B-Diffusers`
  - LTX-2.3 → `Lightricks/LTX-2.3`, plus the Gemma text encoder
    `google/gemma-3-12b-it-qat-q4_0-unquantized` (gated)
  - MiniMax-H3 → `MiniMaxAI/MiniMax-H3` (only the root partitions, ~144 GB of
    the 498 GB repo) plus the Turbo adapter from `lightx2v/Minimax-h3-Turbo`.
    The download is large enough that it is worth doing before the first run:
    ```bash
    huggingface-cli download MiniMaxAI/MiniMax-H3 \
        --include "*.json" "transformer/*" "text_encoder/*" "tokenizer/*" "vae/*" \
                  "audio_vae/*" "processor/*" "scheduler/*" "audio_scheduler/*"
    huggingface-cli download lightx2v/Minimax-h3-Turbo \
        minimax_h3_fl2v_turbo_4step_v1.2_768p_bf16.safetensors
    ```
    H3 is released under the MiniMax H3 Community License, which has
    region-specific terms. Read it on the model page before using the model.

  If a model's HuggingFace page is gated, accept its license there and
  authenticate once so the download can proceed:
  ```bash
  huggingface-cli login
  ```
- **The first standup is slow.** Initial warmup includes trace capture:
  ~5–10 min for SDXL, ~15 min for LTX-2.3, ~15–25 min for Wan 2.2, and
  ~15–20 min for MiniMax-H3, which also writes ~120 GB of converted weights to
  `~/.cache/tt-dit` (later standups take ~2 min). The node
  blocks until the server reports healthy; this is expected, not a hang. Watch
  progress in `<TT_METAL_DIR>/<model>_server_comfy.log`, which on Stack 2 is
  `~/src/tt-inference-server/comfyui-media-server/<model>_server_comfy.log`.
- **A fresh tt-metal checkout compiles its device kernels on first use.** The
  very first standup from a new checkout adds kernel compilation on top of the
  warmup, and the log fills with compiler command lines. Their `-Werror` flags
  and the deprecation notes are not errors. Compiled kernels are cached in
  `~/.cache/tt-metal-cache`, so later standups skip this.
- **One model at a time.** Switching `model_type` tears down the running server
  and starts the new one (another warmup).

---

## 8. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `launch_server.sh not found at <path>` | `TT_METAL_DIR` is wrong for your stack. Stack 2: it must be `<tt-inference-server>/comfyui-media-server` (use `launch_subprocess_mode.sh`). Stack 1: it must be the tt-metal repo root, which the sibling layout gives you by default. |
| `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'` | Mixed stacks. The relocated server is paired with a `samt/standalone-media-20260703` checkout; see the table in section 2. |
| Server never becomes healthy | First Wan 2.2 warmup can take ~25 min; check `<TT_METAL_DIR>/<model>_server_comfy.log`; raise `TT_SERVER_READY_TIMEOUT` (seconds). |
| `tt-smi not found` | Ensure `tt-smi` is on `PATH` (installed by tt-installer) or set `TT_SMI_BIN`. |
| `Error: --board is required when --model ltx` | Only when launching the server by hand -- the node always passes it. Add `--board p300x2` (or your board) to `launch_server.sh`. |
| LTX clip is the wrong size or length | Geometry is pinned when the pipeline is built, so it cannot change per request. Relaunch with `--height 1088 --width 1920 --frames 121`: height and width must be multiples of 64, and `frames - 1` divisible by 8. |
| Port `:8000` already in use | Set `TT_SERVER_PORT`. The web UI port (`--port`, default 8188) is separate. |
| Wan workflow reports `VHS_VideoCombine` missing | It is an older copy of the graph. Use the one from `example_workflows/`, which uses core Create Video instead, or install ComfyUI-VideoHelperSuite. |
| LoRA dropdown is empty | The files are not under `ComfyUI/models/loras/` in the expected subfolder (`wan22/…`, `ltx/`); see section 6. |
| Device wedged after a crash | `tt-smi -r` to reset, or use the **TT Unload Model** node's board-reset option. |

For the full list of configuration variables and per-node details, see the node
[README](README.md).
