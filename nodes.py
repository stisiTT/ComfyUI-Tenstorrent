"""
Custom ComfyUI nodes for Tenstorrent hardware.

Provides nodes for loading models and running staged inference (denoise /
VAE encode / VAE decode) on Tenstorrent accelerators via the tt-metal HTTP
inference server (see ``comfy/backends/tt_http_client.py``).

Architecture (all-custom nodes, server owns the denoise loop):

    TT_CheckpointLoader -> (MODEL, CLIP, VAE)
    [TT_LoraLoader]     -> MODEL (with lora attached)
    TT_KSampler         -> LATENT      (/latent/denoise, falls back to /image/generations)
    TT_VAEDecode        -> IMAGE       (/vae/decode)
    TT_VAEEncode        -> LATENT      (/vae/encode)
"""

import logging
import os
import sys
from typing import Optional, Tuple

# Add comfy to path for imports
comfy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "comfy")
if comfy_path not in sys.path:
    sys.path.insert(0, comfy_path)

try:
    from comfy.backends.tt_http_client import get_client, StagedOpNotAvailable
    print("✓ Successfully imported tt_http_client (HTTP transport)")
except ImportError as e:
    print(f"❌ Failed to import tt_http_client: {e}")
    import traceback
    traceback.print_exc()
    get_client = None

    class StagedOpNotAvailable(RuntimeError):
        pass

from . import server_manager
from .wrappers import TTModelWrapper, TTCLIPWrapper, TTVAEWrapper
from .utils import get_model_config, format_bytes

try:
    import folder_paths
except ImportError:
    folder_paths = None

logger = logging.getLogger(__name__)

# Sentinel dropdown entry meaning "no adapter".
LORA_NONE = "None"


def _lora_choices(prefix: Optional[str] = None) -> list:
    """Return LoRA dropdown options from ComfyUI's ``models/loras`` folder.

    The first entry is always ``LORA_NONE`` (no adapter). When ``prefix`` is given
    (e.g. ``"sdxl"`` or ``"wan22"``), only adapters under that subfolder are shown,
    falling back to the full list if the prefix matches nothing. Wrapped so a
    missing folder never raises from ``INPUT_TYPES`` (which would drop the node
    from ``/object_info`` and break it in the UI).
    """
    try:
        names = folder_paths.get_filename_list("loras") if folder_paths is not None else []
    except Exception:
        names = []
    if prefix:
        filtered = [n for n in names if n.replace("\\", "/").startswith(f"{prefix}/")]
        names = filtered or names
    return [LORA_NONE] + list(names)


def _resolve_lora_path(name: Optional[str]) -> Optional[str]:
    """Map a dropdown selection to a server-readable absolute path.

    Returns ``None`` for the no-adapter sentinel / empty selection. Co-located
    ComfyUI and tt-metal server share the filesystem, so the absolute path under
    ``models/loras`` is valid for the server to load.
    """
    name = (name or "").strip()
    if not name or name == LORA_NONE:
        return None
    if folder_paths is not None:
        resolved = folder_paths.get_full_path("loras", name)
        if resolved:
            return resolved
    return name


def _default_server_url() -> str:
    """Resolve the default server URL from ComfyUI cli args / env."""
    try:
        from comfy.cli_args import args
        if getattr(args, "tt_server_url", None):
            return args.tt_server_url
    except Exception:
        pass
    return os.getenv("TT_SERVER_URL", "http://127.0.0.1:8000")


def _extract_prompt_text(conditioning) -> Optional[str]:
    """Best-effort extraction of prompt text from a ComfyUI CONDITIONING.

    The TT CLIP wrapper stores the original prompt string in the conditioning
    metadata under the ``prompt`` key (see wrappers.TTCLIPWrapper).
    """
    if isinstance(conditioning, list) and len(conditioning) > 0:
        cond_data = conditioning[0]
        if isinstance(cond_data, (list, tuple)) and len(cond_data) >= 2:
            metadata = cond_data[1]
            if isinstance(metadata, dict):
                return metadata.get("prompt")
    return None


def _attach_wan_lora_params(params: dict, model) -> None:
    """Copy wan22 LoRA fields (attached via TT_WanLoraLoader) onto a request dict.

    The model wrapper's ``lora`` dict carries per-expert paths and a shared scale.
    None values are dropped by the HTTP client, so unset experts are simply
    omitted from the request.
    """
    lora = getattr(model, "lora", None)
    if not lora:
        return
    high = lora.get("high_lora_path")
    low = lora.get("low_lora_path")
    if not high and not low:
        return
    params["high_lora_path"] = high
    params["low_lora_path"] = low
    params["lora_scale"] = lora.get("lora_scale")


def build_denoise_progress_callback(num_steps, unique_id, section_labels=None):
    """Build a progress callback that drives a native ComfyUI ProgressBar.

    Shared by TT_KSampler (SDXL) and TT_WanSampler (wan22). Returns None if the
    progress plumbing is unavailable, in which case the client falls back to a
    non-streaming request. ``section_labels`` maps server ``section_start`` event
    names to human-readable phase text (SDXL emits no section events).
    """
    try:
        import comfy.utils
    except Exception:
        logger.warning("comfy.utils unavailable; progress bar disabled")
        return None

    try:
        from server import PromptServer
    except Exception:
        PromptServer = None

    section_labels = section_labels or {}
    pbar = comfy.utils.ProgressBar(max(int(num_steps), 1), node_id=unique_id)

    def on_progress(event):
        etype = event.get("type")
        if etype == "denoise_step":
            total = int(event.get("total") or num_steps)
            step = int(event.get("step") or 0)
            pbar.update_absolute(step, total)
        elif etype == "section_start" and unique_id is not None and PromptServer is not None:
            label = section_labels.get(event.get("name"), event.get("name") or "")
            try:
                PromptServer.instance.send_progress_text(label, unique_id)
            except Exception:
                pass

    return on_progress


class TT_CheckpointLoader:
    """
    Stand up a Tenstorrent tt-metal model and return handles.

    Picking the model here "opens" it: the node spawns and supervises the
    tt-metal server (via launch_server.sh) for the selected model on the default
    board (sdxl -> p150, wan22 -> p300x2), then connects over HTTP. The server
    owns the device and the denoise trace.

    Switching the model restarts the server (multi-minute warmup). MODEL/CLIP/VAE
    drive the staged graph for both paths: sdxl -> CLIP Text Encode -> TT_KSampler
    -> TT_VAEDecode; wan22 -> CLIP Text Encode -> TT_WanSampler -> TT_VAEDecode
    (or the monolithic TT_TextToVideo).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_type": (["sdxl", "wan22"], {
                    "default": "sdxl",
                    "tooltip": "Model to stand up. sdxl -> image graph; wan22 -> video graph (TT_WanSampler or TT_TextToVideo)"
                }),
            },
            "optional": {
                "board": ("STRING", {
                    "default": "",
                    "tooltip": "Advanced: override board (default sdxl=p150, wan22=p300x2)"
                }),
                "server_url": ("STRING", {
                    "default": "",
                    "tooltip": "Advanced: connect to an already-running server URL instead of auto-standup"
                }),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("model", "clip", "vae")
    OUTPUT_TOOLTIPS = (
        "Diffusion/video model handle (routes to the tt-metal server)",
        "CLIP text encoder — feed into CLIP Text Encode (sdxl and wan22)",
        "VAE decoder — feed into TT_VAEDecode (sdxl and wan22)",
    )
    FUNCTION = "load_checkpoint"
    CATEGORY = "Tenstorrent"
    DESCRIPTION = "Stand up a tt-metal model (auto-launch server) and return MODEL/CLIP/VAE handles"

    def load_checkpoint(self, model_type: str, board: str = "", server_url: str = "") -> Tuple:
        if get_client is None:
            raise RuntimeError("Tenstorrent HTTP client not available (import failed).")

        server_url = (server_url or "").strip()
        board = (board or "").strip() or None

        if server_url:
            # Advanced: connect to an externally-managed server (no standup).
            base_url = server_url
            logger.info(f"TT_CheckpointLoader: connecting to external server {base_url} (no standup)")
        else:
            # Auto-standup: spawn/supervise the tt-metal server for this model.
            logger.info(f"TT_CheckpointLoader: ensuring tt-metal server for '{model_type}' (board={board or 'default'})")
            logger.info("First run / model switch can take several minutes (model load + trace capture)...")
            base_url = server_manager.ensure_server(model_type, board=board)

        client = get_client(base_url)
        try:
            health = client.get_health()
        except Exception as e:
            raise RuntimeError(f"Could not reach tt-metal server at {base_url}: {e}")

        status = health.get("status")
        served = health.get("model", "unknown")
        logger.info(f"Server healthy={status}, serving model='{served}', workers={health.get('workers_alive')}")
        if status != "healthy":
            raise RuntimeError(f"tt-metal server at {base_url} is not healthy: {health}")

        model = TTModelWrapper(client, model_type, server_info=health)
        clip = TTCLIPWrapper(client, model_type, server_info=health)
        vae = TTVAEWrapper(client, model_type, server_info=health)
        logger.info("Created MODEL/CLIP/VAE handles")
        return (model, clip, vae)


class TT_LoraLoader:
    """
    Attach a LoRA adapter to a Tenstorrent MODEL handle.

    The LoRA is resolved/downloaded and applied server-side (mirrors the
    tt-media-server ``lora_path`` / ``lora_scale`` request fields). This node
    only records the request on the MODEL handle; the values are forwarded to
    the server by TT_KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from TT_CheckpointLoader"}),
                "lora_name": (_lora_choices("sdxl"), {
                    "tooltip": "LoRA adapter from ComfyUI/models/loras (select None to disable)"
                }),
                "lora_scale": ("FLOAT", {
                    "default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA adapter scale"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Model with LoRA request attached",)
    FUNCTION = "apply_lora"
    CATEGORY = "Tenstorrent"
    DESCRIPTION = "Attach a LoRA (lora_path + lora_scale) to be applied server-side"

    def apply_lora(self, model, lora_name: str, lora_scale: float) -> Tuple:
        if not hasattr(model, "with_lora"):
            raise RuntimeError("TT_LoraLoader requires a Tenstorrent MODEL from TT_CheckpointLoader.")
        lora_path = _resolve_lora_path(lora_name)
        if not lora_path:
            logger.info("TT_LoraLoader: no lora selected, passing model through unchanged")
            return (model,)
        logger.info(f"TT_LoraLoader: attaching lora_path='{lora_path}', lora_scale={lora_scale}")
        return (model.with_lora({"lora_path": lora_path, "lora_scale": lora_scale}),)


class TT_WanLoraLoader:
    """
    Attach a LoRA adapter to a Tenstorrent wan22 MODEL handle.

    Wan2.2 is a two-expert (MoE) model: the high-noise expert handles early
    layout steps and the low-noise expert handles late detail steps. LoRA
    adapters are therefore specified per expert. Either or both paths may be
    set; ``lora_scale`` applies to both. The values are recorded on the MODEL
    handle and forwarded to the server (which loads/binds them on device) by
    TT_WanSampler / TT_TextToVideo.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "wan22 MODEL from TT_CheckpointLoader"}),
                "high_lora_name": (_lora_choices("wan22"), {
                    "tooltip": "High-noise expert adapter from ComfyUI/models/loras (select None to skip)"
                }),
                "low_lora_name": (_lora_choices("wan22"), {
                    "tooltip": "Low-noise expert adapter from ComfyUI/models/loras (select None to skip)"
                }),
                "lora_scale": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "LoRA adapter scale (applied to both experts)"
                }),
            }
        }

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Model with wan22 LoRA request attached",)
    FUNCTION = "apply_lora"
    CATEGORY = "Tenstorrent/video"
    DESCRIPTION = "Attach per-expert wan22 LoRA paths (high/low) to be applied server-side"

    def apply_lora(self, model, high_lora_name: str, low_lora_name: str, lora_scale: float) -> Tuple:
        if not hasattr(model, "with_lora"):
            raise RuntimeError("TT_WanLoraLoader requires a Tenstorrent MODEL from TT_CheckpointLoader.")
        if getattr(model, "model_type", None) != "wan22":
            raise RuntimeError(
                f"TT_WanLoraLoader requires a wan22 model (got '{getattr(model, 'model_type', None)}'). "
                "For SDXL, use TT_LoraLoader."
            )
        high = _resolve_lora_path(high_lora_name)
        low = _resolve_lora_path(low_lora_name)
        if not high and not low:
            logger.info("TT_WanLoraLoader: no lora paths, passing model through unchanged")
            return (model,)
        logger.info(
            f"TT_WanLoraLoader: attaching high_lora_path='{high}', low_lora_path='{low}', lora_scale={lora_scale}"
        )
        return (model.with_lora({
            "high_lora_path": high,
            "low_lora_path": low,
            "lora_scale": lora_scale,
        }),)


class TT_KSampler:
    """
    Run the denoise loop on the tt-metal server and return latents.

    Prefers the staged ``/latent/denoise`` endpoint (Milestone 2). If the server
    only exposes the full pipeline (``/image/generations``), it falls back to a
    single full-pipeline call and returns the decoded image as a passthrough
    LATENT (consumed by TT_VAEDecode). Supports txt2img and img2img.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model from TT_CheckpointLoader"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning (from CLIPTextEncode)"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (from CLIPTextEncode)"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1, "tooltip": "Denoising steps"}),
                "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "CFG scale"}),
                "guidance_rescale": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Guidance rescale (0 = off)"}),
            },
            "optional": {
                "latent_image": ("LATENT", {"tooltip": "Input latents for img2img (optional)"}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise strength (<1 = img2img)"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    OUTPUT_TOOLTIPS = ("Denoised latent tensors",)
    FUNCTION = "sample"
    CATEGORY = "Tenstorrent/sampling"
    DESCRIPTION = "Run denoising on the tt-metal server; returns LATENT"

    def sample(self, model, positive, negative, seed, steps, cfg, guidance_rescale,
               latent_image=None, denoise=1.0, unique_id=None) -> Tuple:
        if not hasattr(model, "client"):
            raise RuntimeError("TT_KSampler requires a Tenstorrent MODEL from TT_CheckpointLoader.")
        if getattr(model, "model_type", None) != "sdxl":
            raise RuntimeError(
                f"TT_KSampler supports the SDXL image path only (got '{getattr(model, 'model_type', None)}'). "
                "For wan22, use the TT_TextToVideo node."
            )

        client = model.client
        positive_text = _extract_prompt_text(positive) or "a beautiful landscape"
        negative_text = _extract_prompt_text(negative) or ""

        logger.info(f"TT_KSampler: steps={steps}, cfg={cfg}, rescale={guidance_rescale}, seed={seed}")
        logger.info(f"  positive: {positive_text[:80]!r}")

        params = {
            "prompt": positive_text,
            "negative_prompt": negative_text,
            "num_inference_steps": int(steps),
            "guidance_scale": float(cfg),
            "seed": int(seed),
        }
        if guidance_rescale and guidance_rescale > 0.0:
            params["guidance_rescale"] = float(guidance_rescale)
        if model.model_type == "sdxl":
            params["prompt_2"] = positive_text
            params["negative_prompt_2"] = negative_text
        # LoRA passthrough (attached via TT_LoraLoader)
        if getattr(model, "lora", None):
            params["lora_path"] = model.lora.get("lora_path")
            params["lora_scale"] = model.lora.get("lora_scale")

        # img2img: not yet supported by the staged SDXL path (the base
        # TtSDXLPipeline standup does not accept an input-latent / denoising-start
        # contract). Fail loudly instead of silently ignoring the input.
        if latent_image is not None and float(denoise) < 1.0:
            raise RuntimeError(
                "TT_KSampler: staged img2img (latent_image with denoise < 1.0) is not "
                "supported in this POC. Use denoise=1.0 (txt2img). img2img is a follow-up."
            )

        progress_callback = build_denoise_progress_callback(int(steps), unique_id)

        try:
            latents = client.denoise(progress_callback=progress_callback, **params)
            logger.info(f"TT_KSampler: received latents {tuple(latents.shape)} via /latent/denoise")
            lora_status = getattr(client, "last_lora_status", None)
            if lora_status and not lora_status.get("applied"):
                logger.warning(
                    f"TT_KSampler: requested LoRA {lora_status.get('requested')!r} was not applied "
                    f"(reason={lora_status.get('skipped_reason')}); output uses base weights."
                )
            elif lora_status and lora_status.get("skipped_reason"):
                logger.warning(
                    f"TT_KSampler: LoRA {lora_status.get('requested')!r} only partially applied "
                    f"(reason={lora_status.get('skipped_reason')})."
                )
            return ({"samples": latents},)
        except StagedOpNotAvailable:
            logger.warning(
                "Staged /latent/denoise not available — falling back to full-pipeline "
                "/image/generations (TT_VAEDecode will pass the image through)."
            )
            # Drop staged-only fields the full endpoint doesn't accept.
            params.pop("latent_image", None)
            params.pop("denoise_strength", None)
            params.pop("lora_path", None)
            params.pop("lora_scale", None)
            images = client.generate_image(**params)  # [B, H, W, C] in [0, 1]
            return ({"samples": images, "tt_already_decoded": True},)


class TT_VAEDecode:
    """
    Decode latents to images/frames via the tt-metal server.

    SDXL latents go to ``/vae/decode`` (image batch [B, H, W, C]); wan22 video
    latents go to ``/video/vae_decode`` (frame batch [T, H, W, C]). If the LATENT
    was produced by the full-pipeline fallback (carries ``tt_already_decoded``),
    the image is passed through unchanged.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "samples": ("LATENT", {"tooltip": "Latents from TT_KSampler (sdxl) or TT_WanSampler (wan22)"}),
                "vae": ("VAE", {"tooltip": "VAE from TT_CheckpointLoader"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_TOOLTIPS = ("Decoded images in [0, 1]",)
    FUNCTION = "decode"
    CATEGORY = "Tenstorrent/latent"
    DESCRIPTION = "Decode latents to images using the tt-metal VAE"

    def decode(self, samples, vae) -> Tuple:
        if not isinstance(samples, dict) or "samples" not in samples:
            raise RuntimeError(f"TT_VAEDecode requires LATENT format. Got: {type(samples)}")

        # Full-pipeline fallback already produced images.
        if samples.get("tt_already_decoded"):
            logger.info("TT_VAEDecode: passthrough (image already decoded by full-pipeline fallback)")
            return (samples["samples"],)

        if not hasattr(vae, "client"):
            raise RuntimeError("TT_VAEDecode requires a Tenstorrent VAE from TT_CheckpointLoader.")
        model_type = getattr(vae, "model_type", None)
        if model_type not in ("sdxl", "wan22"):
            raise RuntimeError(
                f"TT_VAEDecode supports the SDXL image path and wan22 video path (got '{model_type}')."
            )

        latents = samples["samples"]
        if model_type == "wan22":
            logger.info(f"TT_VAEDecode: decoding wan22 latents {tuple(latents.shape)} via /video/vae_decode")
            images = vae.client.vae_decode_video(latents)  # [T, H, W, C] in [0, 1]
        else:
            logger.info(f"TT_VAEDecode: decoding latents {tuple(latents.shape)} via /vae/decode")
            images = vae.client.vae_decode(latents)  # [B, H, W, C] in [0, 1]
        images = images.clamp(0.0, 1.0)
        logger.info(f"TT_VAEDecode: received image {tuple(images.shape)}")
        return (images,)


class TT_VAEEncode:
    """
    Encode images to latents via the tt-metal server (``/vae/encode``).

    Useful for img2img: feed the resulting LATENT into TT_KSampler.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pixels": ("IMAGE", {"tooltip": "Input images from LoadImage or other source"}),
                "vae": ("VAE", {"tooltip": "VAE from TT_CheckpointLoader"}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    OUTPUT_TOOLTIPS = ("Encoded latent tensors",)
    FUNCTION = "encode"
    CATEGORY = "Tenstorrent/latent"
    DESCRIPTION = "Encode images to latents using the tt-metal VAE"

    def encode(self, pixels, vae) -> Tuple:
        import torch

        if not hasattr(vae, "client"):
            raise RuntimeError("TT_VAEEncode requires a Tenstorrent VAE from TT_CheckpointLoader.")
        if not isinstance(pixels, torch.Tensor) or pixels.ndim != 4:
            raise RuntimeError(f"TT_VAEEncode expects IMAGE tensor [B, H, W, C]. Got: {type(pixels)}")

        logger.info(f"TT_VAEEncode: encoding image {tuple(pixels.shape)} via /vae/encode")
        latents = vae.client.vae_encode(pixels)  # [B, C, H, W]
        logger.info(f"TT_VAEEncode: received latents {tuple(latents.shape)}")
        return ({"samples": latents},)


class TT_WanSampler:
    """
    Staged wan22 sampler: run the denoise loop on the tt-metal server and return
    video LATENT (``/video/denoise``).

    This is the staged counterpart to the monolithic TT_TextToVideo: pair it with
    CLIP Text Encode (positive/negative) and TT_VAEDecode to get the standard
    Load -> Encode -> Sample -> Decode graph. Geometry (width/height/num_frames)
    is fixed at server start, so it is not exposed here.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "wan22 MODEL from TT_CheckpointLoader"}),
                "positive": ("CONDITIONING", {"tooltip": "Positive conditioning (from CLIP Text Encode)"}),
                "negative": ("CONDITIONING", {"tooltip": "Negative conditioning (from CLIP Text Encode)"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1, "tooltip": "Denoising steps"}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "CFG for the high-noise expert (layout). Wan2.2 recommended ~4.0; must be > 1."}),
                "guidance_scale_2": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "CFG for the low-noise expert (detail). Recommended ~3.0; must be > 1."}),
                "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Scheduler flow shift. ~12.0 for 480p, ~5.0 for 720p."}),
                "boundary_ratio": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Fraction of steps handled by the high-noise expert (default 0.875)."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
            },
            "hidden": {
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("samples",)
    OUTPUT_TOOLTIPS = ("Denoised video latents [B, z_dim, F, H, W] (feed into TT_VAEDecode)",)
    FUNCTION = "sample"
    CATEGORY = "Tenstorrent/video"
    DESCRIPTION = "Run wan22 denoising on the tt-metal server; returns video LATENT for TT_VAEDecode"

    # Human-readable labels for the pipeline section events emitted by the server.
    _SECTION_LABELS = {
        "encoder": "Encoding prompt",
        "prepare_latents": "Preparing latents",
        "denoising": "Denoising",
        "vae": "VAE decoding",
    }

    def sample(self, model, positive, negative, num_inference_steps, guidance_scale,
               guidance_scale_2, flow_shift, boundary_ratio, seed, unique_id=None) -> Tuple:
        if not hasattr(model, "client"):
            raise RuntimeError("TT_WanSampler requires a Tenstorrent MODEL from TT_CheckpointLoader.")
        if getattr(model, "model_type", None) != "wan22":
            raise RuntimeError(
                f"TT_WanSampler requires a wan22 model (got '{getattr(model, 'model_type', None)}'). "
                "Select 'wan22' in TT_CheckpointLoader."
            )

        positive_text = _extract_prompt_text(positive) or "a cinematic shot of a city at night"
        negative_text = _extract_prompt_text(negative) or ""

        logger.info(
            f"TT_WanSampler: steps={num_inference_steps}, guidance={guidance_scale}/{guidance_scale_2}, "
            f"flow_shift={flow_shift}, boundary_ratio={boundary_ratio}, seed={seed}"
        )
        logger.info(f"  positive: {positive_text[:80]!r}")

        progress_callback = build_denoise_progress_callback(
            int(num_inference_steps), unique_id, self._SECTION_LABELS
        )

        video_params = {
            "prompt": positive_text,
            "negative_prompt": negative_text,
            "num_inference_steps": int(num_inference_steps),
            "guidance_scale": float(guidance_scale),
            "guidance_scale_2": float(guidance_scale_2),
            "flow_shift": float(flow_shift),
            "boundary_ratio": float(boundary_ratio),
            "seed": int(seed),
        }
        # LoRA passthrough (attached via TT_WanLoraLoader). Per-expert paths +
        # scale are applied on device by the server's WanRunner.
        _attach_wan_lora_params(video_params, model)

        latents = model.client.denoise_video(progress_callback=progress_callback, **video_params)
        logger.info(f"TT_WanSampler: received latents {tuple(latents.shape)} via /video/denoise")
        return ({"samples": latents},)


class TT_TextToVideo:
    """
    Generate a video (frames) from text on the tt-metal server (wan22).

    Monolithic one-call path (text-encode + sample + VAE-decode in a single
    ``/video/generations`` request). For a staged graph (CLIP Text Encode ->
    TT_WanSampler -> TT_VAEDecode), use those nodes instead. Output is an IMAGE
    batch of T frames [T, H, W, C] in [0, 1], which can be saved (Save Image) or
    fed to a video-combine node.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "wan22 MODEL from TT_CheckpointLoader"}),
                "prompt": ("STRING", {"multiline": True, "default": "a cinematic shot of a city at night", "tooltip": "Positive prompt"}),
                "negative_prompt": ("STRING", {"multiline": True, "default": "", "tooltip": "Negative prompt"}),
                "num_inference_steps": ("INT", {"default": 30, "min": 1, "max": 200, "step": 1, "tooltip": "Denoising steps"}),
                "num_frames": ("INT", {"default": 81, "min": 5, "max": 257, "step": 4, "tooltip": "Number of frames"}),
                "width": ("INT", {"default": 1280, "min": 64, "max": 2048, "step": 16, "tooltip": "Frame width"}),
                "height": ("INT", {"default": 720, "min": 64, "max": 2048, "step": 16, "tooltip": "Frame height"}),
                "guidance_scale": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "CFG for the high-noise expert (early/layout steps). Higher = follows the prompt more strongly but can over-saturate. Wan2.2 recommended ~4.0; must be > 1."}),
                "guidance_scale_2": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "CFG for the low-noise expert (late/detail steps). Recommended ~3.0; must be > 1. Only used in two-stage mode (boundary_ratio > 0)."}),
                "flow_shift": ("FLOAT", {"default": 12.0, "min": 0.0, "max": 30.0, "step": 0.1, "tooltip": "Scheduler flow shift - skews the noise schedule. Use ~12.0 for 480p and ~5.0 for 720p. Higher shifts more denoising into early steps."}),
                "boundary_ratio": ("FLOAT", {"default": 0.875, "min": 0.0, "max": 1.0, "step": 0.005, "tooltip": "Split between the two MoE experts (fraction of timesteps handled by the high-noise expert). Default 0.875. Lower hands more steps to the low-noise/detail expert."}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "tooltip": "Random seed"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("frames",)
    OUTPUT_TOOLTIPS = ("Video frames [T, H, W, C] in [0, 1]",)
    FUNCTION = "generate"
    CATEGORY = "Tenstorrent/video"
    DESCRIPTION = (
        "Generate video frames from text on the tt-metal server (wan22).\n\n"
        "Tunable knobs (hover each input for details):\n"
        "- guidance_scale / guidance_scale_2: CFG for the high-noise (layout) and "
        "low-noise (detail) MoE experts. Recommended ~4.0 / ~3.0; must be > 1.\n"
        "- flow_shift: scheduler noise-schedule shift. ~12.0 for 480p, ~5.0 for 720p.\n"
        "- boundary_ratio: fraction of steps handled by the high-noise expert (default 0.875)."
    )

    def generate(self, model, prompt, negative_prompt, num_inference_steps, num_frames,
                 width, height, guidance_scale, guidance_scale_2, flow_shift, boundary_ratio, seed) -> Tuple:
        if not hasattr(model, "client"):
            raise RuntimeError("TT_TextToVideo requires a Tenstorrent MODEL from TT_CheckpointLoader.")
        if getattr(model, "model_type", None) != "wan22":
            raise RuntimeError(
                f"TT_TextToVideo requires a wan22 model (got '{getattr(model, 'model_type', None)}'). "
                "Select 'wan22' in TT_CheckpointLoader."
            )

        logger.info(f"TT_TextToVideo: {num_frames} frames @ {width}x{height}, steps={num_inference_steps}, seed={seed}")
        video_params = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "num_inference_steps": int(num_inference_steps),
            "num_frames": int(num_frames),
            "width": int(width),
            "height": int(height),
            "guidance_scale": float(guidance_scale),
            "guidance_scale_2": float(guidance_scale_2),
            "flow_shift": float(flow_shift),
            "boundary_ratio": float(boundary_ratio),
            "seed": int(seed),
        }
        # LoRA passthrough (attached via TT_WanLoraLoader).
        _attach_wan_lora_params(video_params, model)

        frames = model.client.generate_video(**video_params)
        logger.info(f"TT_TextToVideo: received frames {tuple(frames.shape)}")
        return (frames,)


class TT_ModelInfo:
    """Display information about a connected Tenstorrent model handle."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"model": ("MODEL", {"tooltip": "Model to inspect"})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_TOOLTIPS = ("Model information as text",)
    FUNCTION = "get_info"
    CATEGORY = "Tenstorrent/utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Display information about a Tenstorrent model handle"

    def get_info(self, model) -> Tuple[str]:
        try:
            if hasattr(model, "client"):
                lines = [
                    "=== Tenstorrent Model Info ===",
                    f"Server model: {model.model_id}",
                    f"Model type:   {model.model_type}",
                    f"Server URL:   {getattr(model.client, 'base_url', 'N/A')}",
                ]
                if model.server_info:
                    lines.append(f"Workers:      {model.server_info.get('workers_alive')}/{model.server_info.get('workers_total')}")
                if getattr(model, "lora", None):
                    lines.append(f"LoRA:         {model.lora.get('lora_path')} (scale={model.lora.get('lora_scale')})")
                if hasattr(model, "config"):
                    cfg = model.config
                    lines += [
                        "",
                        "Configuration:",
                        f"  Latent channels: {cfg.get('latent_channels', 'N/A')}",
                        f"  VAE scale factor: {cfg.get('vae_scale_factor', 'N/A')}",
                    ]
                info = "\n".join(lines)
            else:
                info = f"Not a Tenstorrent model\nType: {type(model).__name__}"
            logger.info(f"Model info:\n{info}")
            return (info,)
        except Exception as e:
            err = f"Error getting model info: {e}"
            logger.error(err)
            return (err,)


class TT_UnloadModel:
    """
    Stop the tt-metal inference server (and optionally reset the boards).

    The tt-metal HTTP server owns the model for its full lifetime, so "unloading"
    means stopping the server process. This stops the server this ComfyUI process
    started AND reaps any orphaned server recorded in the PID lock file.

    When ``reset_board`` is enabled, an all-device tt-smi board reset is run AFTER
    the server is stopped (the device must be released first). The reset affects
    ALL Tenstorrent boards on the host.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model to release"}),
            },
            "optional": {
                "reset_board": ("BOOLEAN", {
                    "default": False,
                    "tooltip": "Danger: after stopping the server, run a tt-smi reset of ALL Tenstorrent boards on this host",
                }),
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "unload"
    CATEGORY = "Tenstorrent/utils"
    OUTPUT_NODE = True
    DESCRIPTION = "Stop the tt-metal server; optionally reset all Tenstorrent boards"

    def unload(self, model, reset_board: bool = False) -> Tuple:
        logger.info("TT_UnloadModel: stopping tt-metal server (managed + PID-file fallback)...")
        server_manager.get_manager().stop_any()
        logger.info("TT_UnloadModel: server stopped.")

        if reset_board:
            logger.warning("TT_UnloadModel: reset_board=True -> resetting ALL Tenstorrent boards via tt-smi")
            server_manager.reset_all_boards()
            logger.info("TT_UnloadModel: board reset completed.")
        return ()
