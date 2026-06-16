"""
Utility functions for Tenstorrent ComfyUI nodes.

Includes model configuration and helper functions.
"""

import torch
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def get_model_config(model_type: str) -> Dict[str, Any]:
    """
    Get model configuration for a given model type.

    Args:
        model_type: One of 'sdxl', 'sd35', 'sd14'

    Returns:
        Configuration dictionary
    """
    configs = {
        "sdxl": {
            "latent_channels": 4,
            "unet_in_channels": 4,
            "clip_dim": 1280,
            "vae_scale_factor": 8,
            "vae_latent_channels": 4,
            "text_encoder_hidden_size": 768,
            "text_encoder_2_hidden_size": 1280,
            "model_size_gb": 6.94,
        },
        "sd35": {
            "latent_channels": 16,
            "unet_in_channels": 16,
            "clip_dim": 4096,
            "vae_scale_factor": 8,
            "vae_latent_channels": 16,
            "text_encoder_hidden_size": 4096,
            "model_size_gb": 11.9,
        },
        "sd14": {
            "latent_channels": 4,
            "unet_in_channels": 4,
            "clip_dim": 768,
            "vae_scale_factor": 8,
            "vae_latent_channels": 4,
            "text_encoder_hidden_size": 768,
            "model_size_gb": 3.4,
        },
        # Wan2.2 T2V is a text->video model handled via a dedicated node; the
        # CLIP/VAE handles are unused. This entry just lets the wrappers be
        # constructed without raising.
        "wan22": {
            "latent_channels": 16,
            "unet_in_channels": 16,
            "clip_dim": 4096,
            "vae_scale_factor": 8,
            "vae_latent_channels": 16,
            "text_encoder_hidden_size": 4096,
            "model_size_gb": 28.0,
        },
    }

    if model_type not in configs:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(configs.keys())}")

    return configs[model_type]


def format_bytes(bytes_val: int) -> str:
    """
    Format bytes as human-readable string.

    Args:
        bytes_val: Number of bytes

    Returns:
        Formatted string (e.g., "1.5 GB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"


def validate_latent_shape(latent: torch.Tensor, model_type: str) -> None:
    """
    Validate latent tensor shape for a given model type.

    Args:
        latent: Latent tensor [batch, channels, height, width]
        model_type: Model type (sdxl, sd35, sd14)

    Raises:
        ValueError: If latent shape is invalid
    """
    config = get_model_config(model_type)
    expected_channels = config["latent_channels"]

    if not isinstance(latent, torch.Tensor):
        raise TypeError(f"Latent must be a torch.Tensor, got {type(latent)}")

    if latent.ndim != 4:
        raise ValueError(f"Latent must have 4 dimensions, got {latent.ndim}")

    if latent.shape[1] != expected_channels:
        raise ValueError(
            f"Latent tensor for {model_type} must have {expected_channels} channels, "
            f"got {latent.shape[1]}"
        )
