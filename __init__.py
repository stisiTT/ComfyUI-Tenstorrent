"""
Tenstorrent ComfyUI Nodes

Provides integration with Tenstorrent hardware via the tt-metal HTTP inference
server. Implements all-custom nodes for staged SDXL inference (checkpoint /
LoRA / KSampler / VAE encode / VAE decode) on Tenstorrent accelerators.
"""

from .nodes import (
    TT_CheckpointLoader,
    TT_LoraLoader,
    TT_KSampler,
    TT_VAEDecode,
    TT_VAEEncode,
    TT_WanSampler,
    TT_TextToVideo,
    TT_ModelInfo,
    TT_UnloadModel,
)

NODE_CLASS_MAPPINGS = {
    "TT_CheckpointLoader": TT_CheckpointLoader,
    "TT_LoraLoader": TT_LoraLoader,
    "TT_KSampler": TT_KSampler,
    "TT_VAEDecode": TT_VAEDecode,
    "TT_VAEEncode": TT_VAEEncode,
    "TT_WanSampler": TT_WanSampler,
    "TT_TextToVideo": TT_TextToVideo,
    "TT_ModelInfo": TT_ModelInfo,
    "TT_UnloadModel": TT_UnloadModel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TT_CheckpointLoader": "TT Checkpoint Loader",
    "TT_LoraLoader": "TT LoRA Loader",
    "TT_KSampler": "TT KSampler",
    "TT_VAEDecode": "TT VAE Decode",
    "TT_VAEEncode": "TT VAE Encode",
    "TT_WanSampler": "TT Wan Sampler",
    "TT_TextToVideo": "TT Text To Video",
    "TT_ModelInfo": "TT Model Info",
    "TT_UnloadModel": "TT Kill Server",
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
