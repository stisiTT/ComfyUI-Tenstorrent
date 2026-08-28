# TT ComfyUI — integration, versions, and how to run

This is the source-of-truth for how the three repos fit together, which
commits/branches to run, and the two ways to launch the server. If you only want
to *use* the nodes, read `README.md`; this file is for keeping the moving parts
in sync.

Last updated: 2026-08-28.

## The three repos and how they connect

```
ComfyUI (this repo)                 tt-metal                         tt-inference-server
  custom_nodes/tenstorrent_nodes      standalone media server          comfyui-media-server/
    server_manager.py  ── spawns ──▶  launch_server.sh                   (relocated copy of the
    nodes.py           ── HTTP  ──▶   server.py  ──▶ TT hardware          same server, for the
                                      (runners import ttnn/models.*)      Docker/run.py path)
```

- The nodes **never import tt-metal**. They stand up the server as a subprocess
  and talk to it over HTTP (`/health`, `/image/generations`, `/latent/denoise`,
  `/vae/encode`, `/vae/decode`, `/video/*`).
- The server is **the same code in two places right now**: it lives at the root
  of a tt-metal branch (where it runs today) and has been copied into
  tt-inference-server (where it's headed). Keep that in mind — see *Two ways to
  run* and *What's still missing*.

## Which versions to run (pin these)

| Repo | Branch | Commit | Pushed? | Notes |
|------|--------|--------|---------|-------|
| tt-metal | `stisi/sdxl-per-component-lora` | `eee0f689b4c` | yes (origin) | **`TT_METAL_HOME` for the relocated stack.** Carries PR #47509's `fuse_lora(lora_scale, clip_scale)` — the per-component LoRA the demo shows. Must be **built** (`./build_metal.sh`, `./create_venv.sh`). |
| tt-inference-server | `samt/comfyui-media-server` | `6a4085df0` | yes (origin) | **`TT_METAL_DIR` for the relocated stack.** The server relocated into `comfyui-media-server/`, plus the two fixes the relocation needed (`TT_METAL_HOME` clobbering, collapsed `fuse_lora` signature). |
| ComfyUI | `master` (stisiTT/ComfyUI) | `b5607d01`+ | yes (origin) | These custom nodes. Fork of comfyanonymous/ComfyUI; the nodes live entirely in `custom_nodes/tenstorrent_nodes/`. |
| tt-metal | `samt/standalone-media-20260703` | `ce05994325a` | yes (origin) | *Pre-relocation stack only.* Amalgamation of the PRs below **plus** its own copy of the server at the repo root. Self-consistent, but its `fuse_lora` is the older `(lora_scale_unet, lora_scale_clip)`. |

### The tt-metal PRs the server depends on

The server's runners import library code that is being upstreamed as separate
PRs. The amalgamation branch above already contains all of them; this is their
status as standalone PRs (so you know what "merged to main" will eventually look
like):

| PR | What | Status |
|----|------|--------|
| [#46519](https://github.com/tenstorrent/tt-metal/pull/46519) | LoRA (Parshwa's) | merged |
| [#47509](https://github.com/tenstorrent/tt-metal/pull/47509) | SDXL per-component LoRA (UNet/CLIP scales + text-encoder LoRA) | open — critical path |
| [#47265](https://github.com/tenstorrent/tt-metal/pull/47265) | Wan 2.2 per-request guidance (`flow_shift`, `boundary_ratio`, `guidance_scale`) | open — nearly done |
| [#47715](https://github.com/tenstorrent/tt-metal/pull/47715) | DenoiseStep per-step progress events (the ComfyUI progress bar) | open — ready, needs reviewer |
| [#48616](https://github.com/tenstorrent/tt-metal/pull/48616) | DeviceClass primitive | open — mergeable |
| [#47510](https://github.com/tenstorrent/tt-metal/pull/47510) | Standalone media server | **not going into tt-metal** — relocating to tt-inference-server |

> Note: the old `comfyui_tt_update.md` in the ComfyUI root references
> #47508 / #47511 — those are **closed** (superseded by #46519). Use the table
> above, not that note.

## Two ways to run

### Path A — subprocess (works today) ✅

The nodes spawn `launch_server.sh` on the host, no container — so there is no
image to build, though tt-metal itself must already be built. This is the path
the 2026-08-16/17 wan22 + LoRA runs used, and the only one that works right now
(see Path B's stale-image blocker).

Since the relocation, the server itself lives in **tt-inference-server**
(`comfyui-media-server/`), while `TT_METAL_HOME` still points at the built
tt-metal checkout that supplies `ttnn` / `models.*`:

```bash
export TT_LAUNCH_MODE=subprocess                   # docker is the default now
export TT_METAL_DIR=<tt-inference-server>/comfyui-media-server   # holds launch_server.sh
export TT_METAL_HOME=<tt-metal>                    # built tt-metal on stisi/sdxl-per-component-lora
export HF_HOME=/path/to/hf_cache                   # so weights don't download on first request
cd <ComfyUI>
./launch_with_http.sh --port 8188 --listen 127.0.0.1
```

`launch_subprocess_mode.sh` in the ComfyUI root wraps exactly this.

> **The two stacks are not interchangeable — do not mix them.** `fuse_lora`'s
> signature differs between the tt-metal branches, so the server copy and the
> `TT_METAL_HOME` checkout have to come from the same stack:
>
> | | server (`TT_METAL_DIR`) | tt-metal (`TT_METAL_HOME`) | `fuse_lora` |
> |---|---|---|---|
> | **Relocated** (this demo) | tt-inference-server `comfyui-media-server/` @ `6a4085df0` | `stisi/sdxl-per-component-lora` @ `eee0f689b4c` | `(lora_scale, clip_scale)` |
> | **Pre-relocation** | tt-metal repo root | `samt/standalone-media-20260703` @ `ce05994325a` | `(lora_scale_unet, lora_scale_clip)` |
>
> Pairing the relocated server with `samt/standalone-media-20260703` raises
> `TypeError: fuse_lora() got an unexpected keyword argument 'clip_scale'` on the
> first LoRA request.

Then pick a model in the **TT Checkpoint Loader** node. It spawns the server,
polls `/health`, and the staged ops hit `server.py`. First warmup is slow
(~5–10 min SDXL, ~15–25 min Wan 2.2 — trace capture). Board defaults:
SDXL → `p150`, Wan 2.2 → `p300x2` (override with `TT_SDXL_BOARD` /
`TT_WAN22_BOARD`).

### Path B — `run.py --docker-server` (target, NOT ready yet) 🚧

The intended end state: the nodes launch the server through tt-inference-server's
standard entrypoint, in dev mode, pointed at a locally-built image:

```bash
python run.py --model <sdxl|wan> --workflow server \
    --tt-device p300x2 --docker-server --dev-mode \
    --override-docker-image <locally-built comfyui-media-server image>
```

**Now wired — the node drives this by default** (`TT_LAUNCH_MODE=docker`).
`server_manager.py` builds and runs the `run.py … --docker-server --dev-mode
--override-docker-image comfyui-media-server:dev` command, discovers the
container, polls `/health`, and stops the container on teardown. The container
entrypoint (`docker-entrypoint.sh`) maps `run.py`'s `MODEL`/`DEVICE` env onto the
launcher.

**Blocker: the local image is stale.** The `comfyui-media-server:dev` image on
this box was built 2026-07-03, which predates both fixes now on
`samt/comfyui-media-server`. Running it crashes wan22 on p300x2 at warmup:

```
Custom mesh graph descriptor file not found:
  .../comfyui-media-server/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
```

That is the `TT_METAL_HOME` bug fixed in `469f987fa`; the image has to be rebuilt
to pick it up. Until then docker mode does not work, despite being the default.

```bash
cd <tt-inference-server>/comfyui-media-server
./build_image.sh comfyui-media-server:dev      # long: builds tt-metal from source
```

Fall back to the working path meanwhile with `TT_LAUNCH_MODE=subprocess` (see
Path A), which is what `launch_subprocess_mode.sh` in the ComfyUI root sets up.

See `tt-inference-server/comfyui-media-server/LOCAL_TESTING.md` for details.

## What's still missing / decide before this leaves dev

Plain-English list of the gaps, in rough priority order:

1. **Rebuild the Docker image.** `comfyui-media-server:dev` predates the fixes on
   `samt/comfyui-media-server`, so docker mode — which is now the *default* —
   crashes wan22 at warmup. Either rebuild the image or flip the default back to
   `subprocess`; shipping a broken default is the most likely way a new user
   bounces off this.
2. **Two copies of the server will drift.** The server exists both at the
   tt-metal branch root and in `comfyui-media-server/`. The two fixes in
   `6a4085df0` went into the tt-inference-server copy only — the tt-metal-root
   copy still has the old `fuse_lora` call and the `TT_METAL_HOME` clobber.
   Decide which is source-of-truth and how they stay in sync.
3. **SDXL/Wan name collision in tt-inference-server.** `sdxl`/`wan` already
   resolve to the OpenAI-API `tt-media-server` image (no ComfyUI staged ops).
   Path B's `--override-docker-image` dodges this in dev; a permanent path needs a
   distinct `inference_engine` (e.g. `comfyui`) or separate model-spec entries.
4. **No HTTP contract version.** The nodes call fixed endpoints
   (`/latent/denoise`, `/vae/*`). Nothing pins the node ↔ server API version, so a
   server change can silently break the nodes. Consider surfacing a version in
   `/health`.
5. **`sd35` is server-supported but not node-exposed.** `server.py` handles
   `sd35`, but `TT_CheckpointLoader`'s `model_type` only offers `sdxl` / `wan22`
   (`nodes.py:197`). Add it if SD3.5 is meant to be usable from the UI.
6. **Weights aren't spelled out.** Document the exact HF repos and LoRA files the
   server expects (SDXL base, Wan2.2-T2V-A14B, SD3.5-large, any test LoRAs), plus
   disk/RAM needs, so `HF_HOME` is set up correctly the first time.
7. **The transition itself.** Once the PRs merge to tt-metal `main` and #47510 is
   retired, the server no longer lives in tt-metal — it lives in
   tt-inference-server. On that day `TT_METAL_DIR` + Path A stop being the story
   and Path B has to be ready. Plan the cutover.
