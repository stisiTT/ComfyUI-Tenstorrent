# Tenstorrent Custom Nodes - Implementation Summary

**Date**: 2025-12-12
**Status**: Complete
**Location**: /home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/

## Overview

Implemented complete custom node integration for Tenstorrent hardware in ComfyUI-tt_standalone. The nodes provide a bridge between ComfyUI's frontend and the Tenstorrent backend for SDXL inference.

## Files Implemented

### 1. __init__.py (29 lines)
**Purpose**: Node registration and initialization

**Key Components**:
- NODE_CLASS_MAPPINGS: Maps node IDs to class implementations
- NODE_DISPLAY_NAME_MAPPINGS: User-friendly display names
- Module exports for ComfyUI

### 2. nodes.py (406 lines)
**Purpose**: Core node implementations

**Nodes Implemented**:

#### TT_CheckpointLoader
- **Function**: Initialize SDXL model on Tenstorrent hardware
- **Inputs**: model_type (sdxl/sd35/sd14), device_id (0-31)
- **Outputs**: MODEL, CLIP, VAE wrappers
- **Pattern**: Adapted from ComfyUI-tt lines 36-126
- **Key Logic**:
  - Connects to backend singleton
  - Calls backend.init_model() with config
  - Returns lightweight wrappers (no weight storage)

#### TT_FullDenoise
- **Function**: Complete text-to-image generation on bridge server
- **Inputs**: model, positive, negative, seed, steps, cfg, width, height
- **Outputs**: IMAGE tensor [B, H, W, C] in [0, 1]
- **Pattern**: Adapted from proven SSIM 0.998+ pattern (lines 251-433)
- **Key Logic**:
  - Prepares inference parameters (prompts, seed, cfg, etc.)
  - Calls backend.full_denoise() for complete inference
  - Deserializes images from shared memory
  - Ensures correct format for ComfyUI (permute if needed)

#### TT_ModelInfo
- **Function**: Display loaded model information
- **Inputs**: model
- **Outputs**: STRING (formatted info)
- **Use Case**: Debugging, monitoring model state

#### TT_UnloadModel
- **Function**: Explicitly unload model from device
- **Inputs**: model
- **Outputs**: None (output node)
- **Use Case**: Memory management

### 3. wrappers.py (107 lines)
**Purpose**: Lightweight model wrappers

**Classes**:
- **TTModelWrapper**: Stores model_id, backend ref, config
- **TTCLIPWrapper**: Stores model_id, backend ref, config
- **TTVAEWrapper**: Stores model_id, backend ref, config

**Design**: Minimal metadata storage (no weights, no full ModelPatcher interface)

### 4. utils.py (102 lines)
**Purpose**: Helper functions

**Functions**:
- get_model_config(model_type): Returns config dict (channels, dims, sizes)
- format_bytes(bytes_val): Human-readable byte formatting
- validate_latent_shape(latent, model_type): Shape validation

### 5. README.md (300 lines)
**Purpose**: Complete documentation

**Sections**:
- Installation instructions
- Prerequisites (hardware, bridge server, models)
- Usage guide with workflow examples
- Node reference (all inputs/outputs documented)
- Architecture diagram
- Troubleshooting guide
- Performance tips
- Limitations and roadmap

## Key Design Decisions

### 1. Simplified vs Full Pattern
**Decision**: Use simplified wrappers for standalone version

**Rationale**:
- ComfyUI-tt uses full ModelPatcher interface (1000+ lines in wrappers.py)
- Standalone version doesn't need step-by-step KSampler integration
- TT_FullDenoise handles complete inference on bridge server
- Lightweight wrappers (107 lines vs 1492 lines) are sufficient

### 2. Text Input vs Conditioning Tensors
**Decision**: TT_FullDenoise accepts text strings directly

**Rationale**:
- Standalone backend.full_denoise() expects text prompts
- CLIP encoding happens on bridge server
- No need for ComfyUI CLIP nodes in workflow
- Simpler workflow: CheckpointLoader → FullDenoise → SaveImage

### 3. Shared Memory for Images
**Decision**: Use backend.tensor_bridge for image transfer

**Rationale**:
- Zero-copy performance for large tensors
- Consistent with backend implementation
- Automatic cleanup via TensorBridge.tensor_from_shm()

### 4. No Intermediate Nodes
**Decision**: Don't implement CLIPTextEncode, KSampler, VAEDecode nodes

**Rationale**:
- Bridge server handles complete inference pipeline
- Intermediate nodes would require step-by-step bridge API (not implemented)
- TT_FullDenoise is sufficient for text-to-image generation
- Can add later if bridge server supports step-by-step mode

## Integration Points

### Backend Connection
```python
from comfy.backends.tenstorrent_backend import get_backend
backend = get_backend()
```

### Model Initialization
```python
model_id = backend.init_model(model_type, {"device_id": str(device_id)})
```

### Full Inference
```python
response = backend.full_denoise(
    model_id=model_id,
    prompt=positive,
    negative_prompt=negative,
    num_inference_steps=steps,
    guidance_scale=cfg,
    width=width,
    height=height,
    seed=seed
)
images = backend.tensor_bridge.tensor_from_shm(response["images_shm"])
```

### Model Unloading
```python
backend.unload_model(model_id)
```

## Validation Results

### Import Test
```bash
python3 -c "from custom_nodes.tenstorrent_nodes import NODE_CLASS_MAPPINGS; print(list(NODE_CLASS_MAPPINGS.keys()))"
```
**Result**: ✓ All 4 nodes loaded successfully

### Node Structure Test
```bash
# Verified all nodes have:
# - INPUT_TYPES classmethod
# - FUNCTION attribute
# - RETURN_TYPES tuple
# - CATEGORY string
```
**Result**: ✓ All nodes properly structured

### INPUT_TYPES Validation
- TT_CheckpointLoader: model_type, device_id ✓
- TT_FullDenoise: model, positive, negative, seed, steps, cfg, width, height ✓
- TT_ModelInfo: model ✓
- TT_UnloadModel: model ✓

## Comparison to Reference Pattern

### From ComfyUI-tt (Full Version)
- **TT_CheckpointLoader**: Lines 36-126 (91 lines)
  - Our version: Similar structure, simplified error handling
- **TT_FullDenoise**: Lines 251-433 (183 lines)
  - Our version: Simplified conditioning extraction (no tensor decomposition)
  - Uses text strings instead of conditioning tensors
- **Wrappers**: Lines 29-1492 (1463 lines)
  - Our version: 107 lines (87% reduction)
  - Removed: Full ModelPatcher interface, apply_model, hooks, patches
  - Kept: Model metadata, backend reference, config

## Testing Recommendations

### 1. Node Loading Test
```bash
cd /home/tt-admin/ComfyUI-tt_standalone
python3 main.py --cpu --listen  # Start ComfyUI
# Check: Tenstorrent nodes appear in node menu
```

### 2. Backend Connection Test
```python
from comfy.backends.tenstorrent_backend import get_backend
backend = get_backend()
response = backend.ping()
print(response)  # Should succeed if bridge server running
```

### 3. Full Workflow Test
1. Start bridge server
2. Load ComfyUI
3. Create workflow: TT_CheckpointLoader → TT_FullDenoise → SaveImage
4. Execute workflow
5. Verify image generated

## Next Steps (Phase 3)

1. **Bridge Server Implementation**
   - Implement full_denoise operation handler
   - Handle CLIP encoding, UNet denoising, VAE decode
   - Return images via shared memory

2. **Integration Testing**
   - Test complete workflow end-to-end
   - Verify image quality (SSIM > 0.998 target)
   - Performance benchmarking

3. **Advanced Features**
   - Img2Img support (requires encode_vae operation)
   - ControlNet integration
   - LoRA support
   - Multi-image batch generation

## File Locations

```
/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/
├── __init__.py              # Node registration
├── nodes.py                 # Node implementations
├── wrappers.py              # Model wrappers
├── utils.py                 # Helper functions
├── README.md                # User documentation
└── IMPLEMENTATION_SUMMARY.md # This file
```

## Dependencies

- **ComfyUI**: Base framework
- **TenstorrentBackend**: /home/tt-admin/ComfyUI-tt_standalone/comfy/backends/tenstorrent_backend.py
- **Bridge Server**: /home/tt-admin/tt-metal/models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_server.py (Phase 3)

## Success Criteria

- [x] 4 nodes implemented
- [x] Node registration complete
- [x] Import test passes
- [x] Structure validation passes
- [x] README documentation complete
- [x] Integration points identified
- [ ] Bridge server implementation (Phase 3)
- [ ] End-to-end workflow test (Phase 3)
- [ ] SSIM validation (Phase 3)

## Notes

- **CRITICAL**: TT_FullDenoise expects text strings, not conditioning tensors
- **IMPORTANT**: Bridge server must implement full_denoise operation
- **REMINDER**: First model load may take 5+ minutes (compilation)
- **TIP**: Use TT_UnloadModel for memory management between generations
