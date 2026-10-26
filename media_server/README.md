# media_server

The FastAPI inference server behind the ComfyUI-Tenstorrent nodes. It serves
image (SDXL, SD3.5) and video (Wan 2.2, LTX-2.3, MiniMax-H3) generation on
Tenstorrent hardware, and exposes the staged and streaming contract the nodes
drive, which tt-inference-server's OpenAI-style `tt-media-server/` does not:

| | this server | `tt-media-server/` |
|---|---|---|
| Consumer | ComfyUI-Tenstorrent nodes | OpenAI-style API clients |
| One-shot | `/image/generations`, `/video/generations`, `/video/av_generations` | `/v1/images/generations`, `/v1/videos/generations` |
| Staged ops | `/latent/denoise`, `/vae/encode`, `/vae/decode`, `/video/denoise`, `/video/vae_decode` | none |
| Streaming | NDJSON progress on the `*_stream` routes | LLM / audio only |

It used to live in tt-inference-server's `comfyui-media-server` branch and
moved here, with its history, so the nodes and the server ship together.

## How it runs

You do not normally start it yourself. `server_manager.py` in the package root
runs `./launch_server.sh --model <model> --board <board>` from this directory
when a TT Checkpoint Loader node asks for a model, and stops it on a model
switch or when ComfyUI exits.

`launch_server.sh` needs `TT_METAL_HOME`, a built tt-metal checkout. It
activates that checkout's `python_env` and runs `server.py` with both
tt-metal and this directory on `PYTHONPATH`. `requirements-server.txt` must
already be installed in that `python_env`; `scripts/setup_tt_metal.sh` does
it, and the launcher stops with a pointer to that script if it is missing. The nodes default `TT_METAL_HOME` to the `tt-metal` checkout next
to ComfyUI.

## Running it by hand

Useful when iterating on a runner:

```bash
cd custom_nodes/ComfyUI-Tenstorrent/media_server
export TT_METAL_HOME=~/src/tt-metal              # your built tt-metal
./launch_server.sh --model sdxl --board p150     # or wan22 / ltx / ltx_pro / h3 with --board p300x2
# add --dev for a shorter warmup while iterating
```

The server comes up on `http://127.0.0.1:8000`; `./stop_server.sh` shuts it
down cleanly. A server started this way is picked up by the nodes as long as it
serves the model they ask for, or set `server_url` on the TT Checkpoint Loader
to point at it explicitly.

## Layout

```
server.py            FastAPI app: /health, /metrics, generation, staged and streaming routes
worker.py            multiprocessing worker; all ttnn imports happen in the child
*_runner.py          tt-metal-coupled runners, the only files that import ttnn / models.*
*_config.py          per-model config dataclasses
device_specs.py      board and (model, board) deployment matrix
utils/               cache / image / logger / validation helpers (no ttnn)
image_test.py        HTTP client used for smoke and accuracy testing
launch_server.sh     launcher (needs TT_METAL_HOME)
stop_server.sh       graceful shutdown (SIGTERM the root process; workers close the device)
requirements-server.txt  fastapi + uvicorn, installed into tt-metal's python_env
```

Imports are flat (`from device_specs import ...`), so the server runs with this
directory as its working directory, separate from ComfyUI's process.

The runners need the tt-metal branch named in the package's `INSTALL.md`; it
carries the per-component LoRA, LTX and MiniMax-H3 changes they call.
