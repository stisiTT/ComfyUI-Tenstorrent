# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

"""
HTTP client for the Tenstorrent tt-metal inference server.

This replaces the legacy Unix-socket bridge (``tenstorrent_backend.py``) with a
plain HTTP/REST client that talks to the maintained FastAPI server stood up by
``tt-metal/launch_server.sh`` (``server.py``).

Transport notes:
- Tensors (latents / images) cross as base64-encoded NumPy ``.npy`` payloads in
  JSON. This is POC-simple and language-agnostic.
- ``bfloat16`` has no NumPy equivalent, so tensors are downcast to ``float32``
  before serialization (mirrors the old ``TensorBridge._parse_dtype`` behaviour).
"""

import base64
import io
import json
import logging
import threading
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import requests
import torch

logger = logging.getLogger("comfy.backends.tt_http")


def _log_lora_status(lora: Optional[Dict[str, Any]]) -> None:
    """Emit a visible warning when the server reports a LoRA was skipped/partial.

    ``lora`` is the optional status object returned by the SDXL denoise endpoints
    (None when no adapter was requested). It carries ``applied`` plus per-component
    flags and an optional ``skipped_reason``.
    """
    if not lora:
        return
    requested = lora.get("requested")
    if not lora.get("applied"):
        logger.warning(
            f"LoRA {requested!r} was NOT applied (reason={lora.get('skipped_reason')}). "
            "Output uses base weights."
        )
    elif lora.get("skipped_reason"):
        logger.warning(
            f"LoRA {requested!r} only partially applied "
            f"(unet={lora.get('unet')}, text_encoder={lora.get('text_encoder')}, "
            f"scale_unet={lora.get('scale_unet')}, scale_clip={lora.get('scale_clip')}, "
            f"reason={lora.get('skipped_reason')})."
        )


class StagedOpNotAvailable(RuntimeError):
    """Raised when the server does not expose a staged endpoint (older server).

    Lets nodes gracefully fall back to the full-pipeline ``/image/generations``
    endpoint (Milestone 1 behaviour) when ``/latent/denoise`` and friends are not
    deployed.
    """


# ---------------------------------------------------------------------------
# Tensor <-> base64 NumPy helpers
# ---------------------------------------------------------------------------


def tensor_to_b64npy(tensor: torch.Tensor) -> Dict[str, Any]:
    """Serialize a torch tensor to a base64 ``.npy`` payload.

    Returns a dict carrying the base64 blob plus shape/dtype for logging/debug.
    """
    if tensor.is_cuda:
        tensor = tensor.cpu()
    # NumPy has no bfloat16 — downcast so the blob is round-trippable.
    if tensor.dtype == torch.bfloat16:
        tensor = tensor.float()
    arr = tensor.detach().contiguous().numpy()
    return ndarray_to_b64npy(arr)


def ndarray_to_b64npy(arr: np.ndarray) -> Dict[str, Any]:
    buf = io.BytesIO()
    np.save(buf, np.ascontiguousarray(arr), allow_pickle=False)
    return {
        "b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
    }


def b64npy_to_tensor(payload: Dict[str, Any]) -> torch.Tensor:
    """Deserialize a base64 ``.npy`` payload back into a torch tensor."""
    arr = b64npy_to_ndarray(payload)
    return torch.from_numpy(arr.copy())


def b64npy_to_ndarray(payload: Dict[str, Any]) -> np.ndarray:
    raw = base64.b64decode(payload["b64"])
    return np.load(io.BytesIO(raw), allow_pickle=False)


def _b64jpeg_to_tensor(b64_str: str) -> torch.Tensor:
    """Decode a base64 JPEG/PNG into a ComfyUI IMAGE tensor [H, W, C] in [0, 1]."""
    from PIL import Image

    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr)


# ---------------------------------------------------------------------------
# HTTP client
# ---------------------------------------------------------------------------


class TTHttpClient:
    """Thread-safe HTTP client for the tt-metal inference server."""

    def __init__(self, base_url: Optional[str] = None, timeout: float = 600.0):
        import os

        self.base_url = (base_url or os.getenv("TT_SERVER_URL", "http://127.0.0.1:8000")).rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        self._lock = threading.RLock()
        # LoRA status from the most recent denoise call (None when no adapter was
        # requested). Read by nodes to surface skipped/partial adapters in the UI.
        self.last_lora_status: Optional[Dict[str, Any]] = None
        logger.info(f"TTHttpClient initialized for {self.base_url}")

    # -- core --------------------------------------------------------------

    def _post(self, path: str, payload: Dict[str, Any], timeout: Optional[float] = None) -> Dict[str, Any]:
        url = f"{self.base_url}{path}"
        with self._lock:
            resp = self._session.post(url, json=payload, timeout=timeout or self.timeout)
        if resp.status_code == 404:
            raise StagedOpNotAvailable(f"Server endpoint {path} not available (404)")
        if resp.status_code >= 400:
            raise RuntimeError(f"Server error {resp.status_code} on {path}: {resp.text[:500]}")
        return resp.json()

    def get_health(self) -> Dict[str, Any]:
        url = f"{self.base_url}/health"
        with self._lock:
            resp = self._session.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.json()

    # -- full pipeline (Milestone 1, zero tt-metal change) -----------------

    def generate_image(self, **params) -> torch.Tensor:
        """Call the full-pipeline ``/image/generations`` endpoint.

        Returns a ComfyUI IMAGE tensor [B, H, W, C] in [0, 1].
        """
        body = {k: v for k, v in params.items() if v is not None}
        data = self._post("/image/generations", body)
        images = data.get("images", [])
        if not images:
            raise RuntimeError("Server returned no images")
        tensors = [_b64jpeg_to_tensor(b) for b in images]
        return torch.stack(tensors, dim=0)

    # -- video (wan22, monolithic) -----------------------------------------

    def generate_video(self, **params) -> torch.Tensor:
        """Call ``/video/generations`` (wan22).

        Returns frames as a ComfyUI IMAGE batch [T, H, W, C] in [0, 1].
        Uses a longer timeout since video generation is slow.
        """
        body = {k: v for k, v in params.items() if v is not None}
        data = self._post("/video/generations", body, timeout=max(self.timeout, 3600.0))
        frames = data.get("frames", [])
        if not frames:
            raise RuntimeError("Server returned no video frames")
        tensors = [_b64jpeg_to_tensor(b) for b in frames]
        return torch.stack(tensors, dim=0)

    # -- video staged ops (wan22) ------------------------------------------

    def denoise_video(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **params,
    ) -> torch.Tensor:
        """Call ``/video/denoise`` (wan22). Returns latents [B, z_dim, F, H, W].

        Uses the long video timeout since denoising runs the full sampling loop.

        If ``progress_callback`` is provided, the streaming ``/video/denoise_stream``
        endpoint is used: each progress event (``section_start`` / ``section_end`` /
        ``denoise_step``) is passed to the callback as it arrives, and the final
        latent tensor is returned. Falls back to the blocking endpoint when the
        server does not expose the streaming variant (older servers).
        """
        body = {k: v for k, v in params.items() if v is not None}
        if progress_callback is not None:
            try:
                return self._denoise_stream("/video/denoise_stream", body, progress_callback)
            except StagedOpNotAvailable:
                logger.info(
                    "Streaming /video/denoise_stream not available; "
                    "falling back to blocking /video/denoise"
                )
        data = self._post("/video/denoise", body, timeout=max(self.timeout, 3600.0))
        return b64npy_to_tensor(data["latent"])

    def _denoise_stream(
        self,
        path: str,
        body: Dict[str, Any],
        progress_callback: Callable[[Dict[str, Any]], None],
    ) -> torch.Tensor:
        """Stream an NDJSON denoise endpoint, forwarding progress events.

        Shared by ``denoise`` (SDXL) and ``denoise_video`` (wan22). Raises
        ``StagedOpNotAvailable`` on 404 so the caller can fall back to the
        blocking endpoint.
        """
        url = f"{self.base_url}{path}"
        timeout = max(self.timeout, 3600.0)
        with self._lock:
            resp = self._session.post(url, json=body, timeout=timeout, stream=True)
            if resp.status_code == 404:
                raise StagedOpNotAvailable(f"Server endpoint {path} not available (404)")
            if resp.status_code >= 400:
                raise RuntimeError(
                    f"Server error {resp.status_code} on {path}: {resp.text[:500]}"
                )

            latent: Optional[torch.Tensor] = None
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    continue
                event = json.loads(line)
                etype = event.get("type")
                if etype == "result":
                    latent = b64npy_to_tensor(event["latent"])
                    self.last_lora_status = event.get("lora")
                    _log_lora_status(self.last_lora_status)
                elif etype == "error":
                    raise RuntimeError(f"Server error on {path}: {event.get('detail')}")
                else:
                    try:
                        progress_callback(event)
                    except Exception:
                        logger.exception("progress_callback raised; continuing stream")

        if latent is None:
            raise RuntimeError("Streaming denoise ended without a result event")
        return latent

    def vae_decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Call ``/video/vae_decode`` (wan22). Returns frames [T, H, W, C] in [0, 1]."""
        body = {"latent": tensor_to_b64npy(latents)}
        data = self._post("/video/vae_decode", body, timeout=max(self.timeout, 3600.0))
        return b64npy_to_tensor(data["image"])

    # -- staged ops (Milestone 2) ------------------------------------------

    def denoise(
        self,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        **params,
    ) -> torch.Tensor:
        """Call ``/latent/denoise`` (SDXL). Returns latents [B, C, H, W].

        If ``progress_callback`` is provided, the streaming
        ``/latent/denoise_stream`` endpoint is used: each ``denoise_step`` event
        is passed to the callback as it arrives, and the final latent tensor is
        returned. Falls back to the blocking endpoint when the server does not
        expose the streaming variant (older servers).

        Raises ``StagedOpNotAvailable`` if the server lacks the blocking endpoint.
        """
        body = {k: v for k, v in params.items() if v is not None}
        if progress_callback is not None:
            try:
                return self._denoise_stream("/latent/denoise_stream", body, progress_callback)
            except StagedOpNotAvailable:
                logger.info(
                    "Streaming /latent/denoise_stream not available; "
                    "falling back to blocking /latent/denoise"
                )
        data = self._post("/latent/denoise", body)
        self.last_lora_status = data.get("lora")
        _log_lora_status(self.last_lora_status)
        return b64npy_to_tensor(data["latent"])

    def vae_decode(self, latents: torch.Tensor) -> torch.Tensor:
        """Call ``/vae/decode``. Returns an IMAGE tensor [B, H, W, C] in [0, 1]."""
        body = {"latent": tensor_to_b64npy(latents)}
        data = self._post("/vae/decode", body)
        return b64npy_to_tensor(data["image"])

    def vae_encode(self, images: torch.Tensor) -> torch.Tensor:
        """Call ``/vae/encode``. Returns latents as a torch tensor [B, C, H, W]."""
        body = {"image": tensor_to_b64npy(images)}
        data = self._post("/vae/encode", body)
        return b64npy_to_tensor(data["latent"])


# ---------------------------------------------------------------------------
# Singleton (keyed by base_url)
# ---------------------------------------------------------------------------

_clients: Dict[str, TTHttpClient] = {}
_clients_lock = threading.Lock()


def get_client(base_url: Optional[str] = None) -> TTHttpClient:
    """Get or create a cached HTTP client for ``base_url``."""
    import os

    resolved = (base_url or os.getenv("TT_SERVER_URL", "http://127.0.0.1:8000")).rstrip("/")
    with _clients_lock:
        if resolved not in _clients:
            _clients[resolved] = TTHttpClient(resolved)
        return _clients[resolved]
