# Installing ComfyUI + Tenstorrent

This guide sets up ComfyUI with the Tenstorrent custom nodes, which run **Stable
Diffusion XL** (image) and **Wan 2.2** (text-to-video) on Tenstorrent hardware
via a tt-metal inference server.

However many repos are involved, it is always **two separate Python
environments** talking over a local HTTP socket. That isolation is intentional:
tt-metal's native/NumPy ABI must never mix with ComfyUI's.

```
ComfyUI  (this repo, ComfyUI venv)            media server (runs in tt-metal's python_env)
  main.py --tenstorrent                          server.py  (FastAPI media server)
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

  Other boards can be selected with `TT_SDXL_BOARD` / `TT_WAN22_BOARD` or the
  node's `board` input, but the two above are the tested paths.
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

**This guide follows Stack 2**, the relocated server. That is where the work is
headed and what the container path is built from.

> **Do not mix the stacks.** `fuse_lora`'s signature differs between the two
> tt-metal branches, so the server copy and the `TT_METAL_HOME` checkout must
> come from the same row. Pairing the relocated server with
> `samt/standalone-media-20260703` raises `TypeError: fuse_lora() got an
> unexpected keyword argument 'clip_scale'` on the first LoRA request. Either
> way the HTTP contract is unchanged: the request body always carries
> `lora_scale_unet` / `lora_scale_clip`.

Clone all three as siblings:

```bash
cd ~/src                 # or wherever you keep checkouts
git clone https://github.com/stisiTT/ComfyUI.git
git clone https://github.com/tenstorrent/tt-metal.git
git clone https://github.com/tenstorrent/tt-inference-server.git
```

Resulting layout:
```
~/src/ComfyUI
~/src/tt-metal
~/src/tt-inference-server
```

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
git checkout stisi/sdxl-per-component-lora     # PR #47509: collapsed fuse_lora signature
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
git checkout samt/comfyui-media-server         # provides comfyui-media-server/
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
./launch_subprocess_mode.sh           # Stack 2; ComfyUI on http://127.0.0.1:8188
# options are passed straight through: --port 8188  --listen 127.0.0.1
```

> `launch_subprocess_mode.sh` exports
> `TT_METAL_DIR=<parent>/tt-inference-server/comfyui-media-server` and
> `TT_METAL_HOME=<parent>/tt-metal`, then execs `launch_with_http.sh`. On
> **Stack 1** run `./launch_with_http.sh` directly instead: no environment
> variables are needed, since `TT_METAL_DIR` defaults to `../tt-metal`.
>
> Keep `--listen` on `127.0.0.1` unless you trust every host on the network.
> The server accepts a client-supplied `lora_path` and loads it from disk, so
> exposing the UI exposes that too.
>
> There is also a containerized path (`TT_LAUNCH_MODE=docker`) that runs the
> server through tt-inference-server's `run.py`; it needs an image built first,
> so it is opt-in. Both stacks and both launch paths are documented in
> [`custom_nodes/tenstorrent_nodes/INTEGRATION.md`](custom_nodes/tenstorrent_nodes/INTEGRATION.md).

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
| `launch_server.sh not found at <path>` | `TT_METAL_DIR` is wrong for your stack. Stack 2: it must be `<tt-inference-server>/comfyui-media-server` (use `launch_subprocess_mode.sh`). Stack 1: it must be the tt-metal repo root, which the sibling layout gives you by default. |
| `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'` | Mixed stacks. The relocated server is paired with a `samt/standalone-media-20260703` checkout; see the table in section 2. |
| Server never becomes healthy | First Wan 2.2 warmup can take ~25 min; check `<tt-metal>/<model>_server_comfy.log`; raise `TT_SERVER_READY_TIMEOUT` (seconds). |
| `tt-smi not found` | Ensure `tt-smi` is on `PATH` (installed by tt-installer) or set `TT_SMI_BIN`. |
| Port `:8000` already in use | Set `TT_SERVER_PORT`. The web UI port (`--port`, default 8188) is separate. |
| Device wedged after a crash | `tt-smi -r` to reset, or use the **TT Unload Model** node's board-reset option. |

For the full list of configuration variables and per-node details, see the node
[README](custom_nodes/tenstorrent_nodes/README.md).
