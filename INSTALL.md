# Installing ComfyUI + Tenstorrent

This guide sets up ComfyUI with the Tenstorrent custom nodes, which run **Stable
Diffusion XL** and **SD 3.5** (image), **Wan 2.2**, **LTX-2.3** and **MiniMax-H3**
(video, the last two with audio) on Tenstorrent hardware via a tt-metal
inference server.

It is always **two separate Python environments** talking over a local HTTP
socket. That isolation is intentional:
tt-metal's native/NumPy ABI must never mix with ComfyUI's.

```
ComfyUI + this package (ComfyUI venv)         media server (runs in tt-metal's python_env)
  main.py --cpu                                  server.py  (FastAPI media server)
   └ TT_CheckpointLoader                          └ launch_server.sh
        └ spawns ──────────────────Popen────────────▶ launch_server.sh
        └ talks ───────────────HTTP :8000───────────▶ server.py ──▶ Tenstorrent device
```

The media server ships inside this package (`media_server/`) and runs under
the `python_env` of a tt-metal checkout you build once.

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
  | SD3.5 Large | Wormhole LoudBox (8 chips) | image generation; not testable on a QuietBox |
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

## 2. Clone

Three checkouts: stock ComfyUI, this package inside ComfyUI's `custom_nodes/`,
and tt-metal next to ComfyUI (section 3 clones it for you).

```bash
cd ~/src                 # or wherever you keep checkouts
git clone https://github.com/Comfy-Org/ComfyUI.git
git clone https://github.com/stisiTT/ComfyUI-Tenstorrent.git ComfyUI/custom_nodes/ComfyUI-Tenstorrent
```

Resulting layout:
```
~/src/ComfyUI
~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent              <- nodes
~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent/media_server <- the server they launch
~/src/tt-metal                                              <- built in section 3
```

ComfyUI itself is unmodified; everything Tenstorrent-specific is in this
package. The nodes look for tt-metal next to ComfyUI; if yours is elsewhere,
set `TT_METAL_HOME` to it.

> LTX adapters go in `ComfyUI/models/loras/ltx/`; that directory must exist, or
> the LTX LoRA node's dropdown falls back to listing adapters for other models,
> none of which can bind.

---

## 3. Build tt-metal (pinned branch)

The media server needs tt-metal changes that are **not on `main`** yet:
per-component LoRA ([PR #47509](https://github.com/tenstorrent/tt-metal/pull/47509),
still open), the LTX-2.3 QuietBox 2 and LoRA work, and a MiniMax-H3 fix. The
branch and commit that carry them on top of a recent `main` are pinned in
[`TT_METAL_REF`](TT_METAL_REF).

One script does the whole section:

```bash
~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent/scripts/setup_tt_metal.sh
```

It clones tt-metal into `~/src/tt-metal` (or reuses a checkout already there),
checks out the pinned commit, then asks before installing system packages with
sudo and before the build. It then creates `python_env` and installs the media
server's dependencies (`fastapi`, `uvicorn`) into it. Declining the build on an
already-built checkout skips straight to those steps. A checkout that
already contains the pinned commit is left on its branch; one with local
changes is never touched. `--yes` answers every prompt, `--skip-deps` skips the
sudo step, and `--metal-home DIR` uses another location.

The same steps by hand:

```bash
cd ~/src
git clone https://github.com/tenstorrent/tt-metal.git
cd tt-metal
git checkout stisi/comfyui_20261023_main
git submodule update --init --recursive

# System build dependencies (uses sudo; one-time).
./install_dependencies.sh

# Build the library.
./build_metal.sh

# Create the tt-metal Python environment (./python_env, via uv).
./create_venv.sh

# The media server's own dependencies.
uv pip install --python python_env/bin/python \
    -r ~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent/media_server/requirements-server.txt
```

You do **not** need to activate `python_env` yourself; `launch_server.sh` does
that when the node starts the server. If the server dependencies are missing,
the launch stops and says to run the setup script.

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
./scripts/launch_with_http.sh         # ComfyUI on http://127.0.0.1:8188
# options are passed straight through: --port 8188  --listen 127.0.0.1
```

No environment variables are needed with the layout above: the nodes launch
`media_server/` from this package against `../tt-metal`. Set `TT_METAL_HOME`
if tt-metal lives elsewhere.

> Keep `--listen` on `127.0.0.1` unless you trust every host on the network.
> The server accepts a client-supplied `lora_path` and loads it from disk, so
> exposing the UI exposes that too.

This activates the ComfyUI venv and runs `main.py --cpu` (the device does the
compute; ComfyUI only orchestrates). Open the web UI,
build a graph starting from **TT Checkpoint Loader**, and pick a model:

- `sdxl` → image graph: `TT KSampler` → `TT VAE Decode` → Save Image
- `sd35` → the same image graph, txt2img only (LoudBox only)
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
| `tt_sdxl_img2img` | SDXL img2img | nothing extra; starts from a bundled example image (swap in your own in Load Image) |
| `tt_sdxl_img2img_photoshop` | SDXL img2img | the SD-PPP plugin (below) |
| `tt_sd35_txt2img` | SD3.5 Large txt2img | an 8-chip Wormhole LoudBox |
| `tt_shutdown_reset` | any | nothing; stops the server and resets the boards |
| `load_tt_wan_standalone` | Wan 2.2 | nothing extra |
| `load_tt_wan_standalone_lora` | Wan 2.2 + Lightning | Wan LoRA files (below) |
| `load_tt_ltx_standalone` | LTX-2.3 | nothing extra for the distilled path |
| `load_tt_ltx_matrix`, `load_tt_ltx_lora_gallery` | LTX-2.3 + style LoRAs | LTX LoRA files (below) |
| `load_tt_h3_standalone` | MiniMax-H3 text to video | nothing extra |
| `load_tt_h3_first_frame` | MiniMax-H3 image to video | nothing extra; animates the same bundled image (an SDXL render of your own works well) |
| `tt_h3_cat_sunbeam` | MiniMax-H3 text to video | nothing extra; a 6 s showcase prompt in the guide format H3 was trained on (description, soundscape, music) |
| `tt_h3_dragon_castle` | MiniMax-H3 text to video | nothing extra; a 6 s showcase with a landing, a roar and an orchestral score |

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
  note in [section 2](#2-clone)). The example graphs expect
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
  progress in
  `~/src/ComfyUI/custom_nodes/ComfyUI-Tenstorrent/media_server/<model>_server_comfy.log`.
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
| `launch_server.sh not found at <path>` | `TT_SERVER_DIR` (or the old `TT_METAL_DIR`) is set in your environment and points away from the bundled server. Unset both to use `media_server/` in this package. |
| `No built tt-metal at <path>` | Build tt-metal next to ComfyUI (section 3), or set `TT_METAL_HOME` to your built checkout. |
| `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'` | The tt-metal checkout is on the wrong branch. Check out the branch in section 3 and rebuild. |
| Server never becomes healthy | First Wan 2.2 warmup can take ~25 min; check `media_server/<model>_server_comfy.log`; raise `TT_SERVER_READY_TIMEOUT` (seconds). |
| `tt-smi not found` | Ensure `tt-smi` is on `PATH` (installed by tt-installer) or set `TT_SMI_BIN`. |
| `Error: --board is required when --model ltx` | Only when launching the server by hand -- the node always passes it. Add `--board p300x2` (or your board) to `launch_server.sh`. |
| LTX clip is the wrong size or length | Geometry is pinned when the pipeline is built, so it cannot change per request. Relaunch with `--height 1088 --width 1920 --frames 121`: height and width must be multiples of 64, and `frames - 1` divisible by 8. |
| Port `:8000` already in use | Set `TT_SERVER_PORT`. The web UI port (`--port`, default 8188) is separate. |
| Wan workflow reports `VHS_VideoCombine` missing | It is an older copy of the graph. Use the one from `example_workflows/`, which uses core Create Video instead, or install ComfyUI-VideoHelperSuite. |
| LoRA dropdown is empty | The files are not under `ComfyUI/models/loras/` in the expected subfolder (`wan22/…`, `ltx/`); see section 6. |
| Device wedged after a crash | `tt-smi -r` to reset, or use the **TT Unload Model** node's board-reset option. |

For the full list of configuration variables and per-node details, see the node
[README](README.md).
