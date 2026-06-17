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
import logging
import threading
from typing import Any, Dict, List, Optional

import numpy as np
import requests
import torch

logger = logging.getLogger("comfy.backends.tt_http")


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

    def denoise_video(self, **params) -> torch.Tensor:
        """Call ``/video/denoise`` (wan22). Returns latents [B, z_dim, F, H, W].

        Uses the long video timeout since denoising runs the full sampling loop.
        """
        body = {k: v for k, v in params.items() if v is not None}
        data = self._post("/video/denoise", body, timeout=max(self.timeout, 3600.0))
        return b64npy_to_tensor(data["latent"])

    def vae_decode_video(self, latents: torch.Tensor) -> torch.Tensor:
        """Call ``/video/vae_decode`` (wan22). Returns frames [T, H, W, C] in [0, 1]."""
        body = {"latent": tensor_to_b64npy(latents)}
        data = self._post("/video/vae_decode", body, timeout=max(self.timeout, 3600.0))
        return b64npy_to_tensor(data["image"])

    # -- staged ops (Milestone 2) ------------------------------------------

    def denoise(self, **params) -> torch.Tensor:
        """Call ``/latent/denoise``. Returns latents as a torch tensor [B, C, H, W].

        Raises ``StagedOpNotAvailable`` if the server lacks the endpoint.
        """
        body = {k: v for k, v in params.items() if v is not None}
        data = self._post("/latent/denoise", body)
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
