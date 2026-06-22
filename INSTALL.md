# Installing ComfyUI + Tenstorrent

This guide sets up ComfyUI with the Tenstorrent custom nodes, which run **Stable
Diffusion XL** (image) and **Wan 2.2** (text-to-video) on Tenstorrent hardware
via a tt-metal inference server.

It is two repositories with two separate Python environments that talk over a
local HTTP socket — the isolation is intentional (tt-metal's native/NumPy ABI
must never mix with ComfyUI's):

```
ComfyUI  (this repo, ComfyUI venv)            tt-metal  (built from source, python_env)
  main.py --tenstorrent                          server.py  (FastAPI media server)
   └ TT_CheckpointLoader                          └ launch_server.sh
        └ spawns ──────────────────Popen────────────▶ launch_server.sh
        └ talks ───────────────HTTP :8000───────────▶ server.py ──▶ Tenstorrent device
```

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

  Other boards can be selected with `TT_SDXL_BOARD` / `TT_WAN22_BOARD` or the
  node's `board` input, but the two above are the tested paths.
- **OS / tooling:** Ubuntu 22.04, **Python 3.10**, `git`, and
  [`uv`](https://github.com/astral-sh/uv) (used by tt-metal's venv script):
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
- **Disk/RAM:** building tt-metal is heavy (several GB of build artifacts, lots of
  RAM). Model weights add tens of GB on first run.

---

## 2. Clone both repos as siblings

The nodes default to finding tt-metal as a **sibling** of the ComfyUI checkout
(`../tt-metal`). Keep this layout and there is nothing to configure:

```bash
cd ~/src                 # or wherever you keep checkouts
git clone https://github.com/stisiTT/ComfyUI.git
git clone https://github.com/tenstorrent/tt-metal.git
```

Resulting layout:
```
~/src/ComfyUI
~/src/tt-metal
```

> If you put tt-metal elsewhere, set `TT_METAL_DIR=/path/to/tt-metal` in the
> environment before launching ComfyUI.

---

## 3. Build tt-metal (pinned commit)

The standalone media server lives on a specific commit that is **not on
tt-metal's `main`**. Check it out, init submodules, then build.

```bash
cd ~/src/tt-metal
git checkout -b comfyui-tt 594eb46335d    # local branch at the pinned commit
git submodule update --init --recursive

# System build dependencies (uses sudo; one-time).
./install_dependencies.sh

# Build the library.
./build_metal.sh

# Create the tt-metal Python environment (./python_env, via uv).
./create_venv.sh
```

> **Why a pinned commit?** This is the known-good build the nodes were tested
> against. The work is being upstreamed; once it merges you can track a release
> instead. To update, re-`git checkout` a newer commit and rebuild.

You do **not** need to activate `python_env` yourself or install the server's
`fastapi`/`uvicorn` deps — `launch_server.sh` does that automatically when the
node starts the server.

---

## 4. Set up the ComfyUI environment

The launcher expects the virtualenv at `ComfyUI/venv`.

```bash
cd ~/src/ComfyUI
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **torch note:** compute runs on the Tenstorrent device, not on a GPU, so the
> CPU build of torch is all ComfyUI needs. To avoid pulling multi-GB CUDA wheels:
> ```bash
> pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
> ```
> (run this before `pip install -r requirements.txt`).

---

## 5. Launch

```bash
cd ~/src/ComfyUI
./launch_with_http.sh                 # ComfyUI on http://127.0.0.1:8188
# options: --port 8188  --listen 0.0.0.0
```

This activates the ComfyUI venv and runs `main.py --tenstorrent`. Open the web UI,
build a graph starting from **TT Checkpoint Loader**, and pick a model:

- `sdxl` → image graph: `TT KSampler` → `TT VAE Decode` → Save Image
- `wan22` → video graph: `TT Wan Sampler` → `TT VAE Decode`

See [`custom_nodes/tenstorrent_nodes/README.md`](custom_nodes/tenstorrent_nodes/README.md)
for the full node reference and the environment-variable table.

---

## 6. First run: weights + warmup

- **Weights download automatically** from HuggingFace on first use, into
  `HF_HOME` (default `~/.cache/huggingface`):
  - SDXL → `stabilityai/stable-diffusion-xl-base-1.0`
  - Wan 2.2 → `Wan-AI/Wan2.2-T2V-A14B-Diffusers`

  If a model's HuggingFace page is gated, accept its license there and
  authenticate once so the download can proceed:
  ```bash
  huggingface-cli login
  ```
- **The first standup is slow.** Initial warmup includes trace capture:
  ~5–10 min for SDXL, ~15–25 min for Wan 2.2. The node blocks until the server
  reports healthy — this is expected, not a hang. Watch progress in
  `~/src/tt-metal/<model>_server_comfy.log`.
- **One model at a time.** Switching `model_type` tears down the running server
  and starts the new one (another warmup).

---

## 7. Troubleshooting

| Symptom | Fix |
|---------|-----|
| `launch_server.sh not found at <path>` | Use the sibling layout, or set `TT_METAL_DIR` to your tt-metal checkout. |
| Server never becomes healthy | First Wan 2.2 warmup can take ~25 min; check `<tt-metal>/<model>_server_comfy.log`; raise `TT_SERVER_READY_TIMEOUT` (seconds). |
| `tt-smi not found` | Ensure `tt-smi` is on `PATH` (installed by tt-installer) or set `TT_SMI_BIN`. |
| Port `:8000` already in use | Set `TT_SERVER_PORT`. The web UI port (`--port`, default 8188) is separate. |
| Device wedged after a crash | `tt-smi -r` to reset, or use the **TT Unload Model** node's board-reset option. |

For the full list of configuration variables and per-node details, see the node
[README](custom_nodes/tenstorrent_nodes/README.md).
