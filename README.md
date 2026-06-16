# Tenstorrent Custom Nodes for ComfyUI

Custom nodes for running Stable Diffusion models on Tenstorrent hardware (N150, N300, T3K, Galaxy) via the TT-Comfy bridge server.

## Features

- **TT Checkpoint Loader**: Load SDXL/SD3.5/SD1.4 models on Tenstorrent hardware
- **TT Full Denoise**: Complete text-to-image generation (CLIP + UNet + VAE) on TT hardware
- **TT Model Info**: Display information about loaded models
- **TT Unload Model**: Explicitly unload models to free device memory

## Installation

The custom nodes are already installed in this ComfyUI-tt_standalone instance:

```
/home/tt-admin/ComfyUI-tt_standalone/custom_nodes/tenstorrent_nodes/
├── __init__.py          # Node registration
├── nodes.py             # Node implementations
├── wrappers.py          # Model wrappers
├── utils.py             # Helper functions
└── README.md            # This file
```

## Prerequisites

1. **Tenstorrent Hardware**: N150, N300, T3K, or Galaxy accelerator
2. **Bridge Server**: TT-Comfy bridge server must be running
3. **Models**: SDXL checkpoint weights accessible to the bridge server

## Usage

### Starting the Bridge Server

Before using these nodes, start the bridge server:

```bash
# From tt-metal directory
cd /home/tt-admin/tt-metal
python models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_server.py
```

The server will listen on `/tmp/tt-comfy.sock` (Unix domain socket).

### Basic Workflow

Create a simple text-to-image workflow in ComfyUI:

```
[TT Checkpoint Loader] → [TT Full Denoise] → [SaveImage]
         ↓
   (model output)
```

**Node Setup:**

1. **TT Checkpoint Loader**
   - `model_type`: "sdxl" (or "sd35", "sd14")
   - `device_id`: 0 (Tenstorrent device ID)
   - **Outputs**: model, clip, vae (lightweight wrappers)

2. **TT Full Denoise**
   - `model`: Connect from TT Checkpoint Loader
   - `positive`: "a beautiful landscape with mountains"
   - `negative`: "blurry, low quality"
   - `steps`: 20
   - `cfg`: 7.0
   - `width`: 1024
   - `height`: 1024
   - `seed`: 42
   - **Output**: images (ready for SaveImage)

3. **SaveImage**
   - `images`: Connect from TT Full Denoise
   - Saves generated images to ComfyUI output folder

### Example Workflow JSON

```json
{
  "1": {
    "class_type": "TT_CheckpointLoader",
    "inputs": {
      "model_type": "sdxl",
      "device_id": 0
    }
  },
  "2": {
    "class_type": "TT_FullDenoise",
    "inputs": {
      "model": ["1", 0],
      "positive": "a beautiful landscape with mountains and a lake",
      "negative": "blurry, low quality, distorted",
      "seed": 42,
      "steps": 20,
      "cfg": 7.0,
      "width": 1024,
      "height": 1024
    }
  },
  "3": {
    "class_type": "SaveImage",
    "inputs": {
      "images": ["2", 0],
      "filename_prefix": "TT_"
    }
  }
}
```

## Node Reference

### TT_CheckpointLoader

Load a Stable Diffusion model on Tenstorrent hardware.

**Inputs:**
- `model_type`: Model type ("sdxl", "sd35", "sd14")
- `device_id`: Tenstorrent device ID (0-31)

**Outputs:**
- `model`: Model wrapper for inference
- `clip`: CLIP wrapper (metadata only, encoding happens on bridge)
- `vae`: VAE wrapper (metadata only, decoding happens on bridge)

**Notes:**
- First load may take 2-5 minutes (compilation + weight loading)
- Subsequent loads are faster (cached)

### TT_FullDenoise

Run complete text-to-image generation on Tenstorrent hardware.

**Inputs:**
- `model`: Model from TT_CheckpointLoader
- `positive`: Positive prompt (text)
- `negative`: Negative prompt (text)
- `seed`: Random seed for reproducibility
- `steps`: Number of denoising steps (1-100)
- `cfg`: Classifier-Free Guidance scale (0.0-20.0)
- `width`: Image width in pixels (256-2048, multiple of 64)
- `height`: Image height in pixels (256-2048, multiple of 64)

**Outputs:**
- `images`: Generated images [B, H, W, C] in range [0, 1]

**Notes:**
- This node performs the entire inference pipeline on the bridge server
- Text encoding (CLIP), denoising (UNet), and VAE decode all happen on TT hardware
- Returns final pixel-space images ready for saving

### TT_ModelInfo

Display information about a loaded model.

**Inputs:**
- `model`: Model to inspect

**Outputs:**
- `info`: Model information as text

**Use Case:** Debugging, monitoring model state

### TT_UnloadModel

Explicitly unload a model from Tenstorrent device.

**Inputs:**
- `model`: Model to unload

**Outputs:** None (output node)

**Use Case:** Free device memory when switching models or done with generation

## Architecture

```
ComfyUI (Frontend)
    ↓
Tenstorrent Custom Nodes
    ↓
TenstorrentBackend (comfy/backends/tenstorrent_backend.py)
    ↓
Unix Domain Socket (/tmp/tt-comfy.sock)
    ↓
Bridge Server (tt_sdxl_server.py)
    ↓
tt-metal SDXL Implementation
    ↓
Tenstorrent Hardware (N150/N300/T3K/Galaxy)
```

### Key Design Principles

1. **Lightweight Wrappers**: Node wrappers store metadata only, not model weights
2. **Bridge-Owned Inference**: All inference happens on the bridge server
3. **Shared Memory**: Tensors transferred via shared memory for zero-copy performance
4. **Unix Sockets**: Low-latency IPC between ComfyUI and bridge server

## Troubleshooting

### "Tenstorrent backend not available"

**Cause**: Bridge server not running or socket not accessible

**Solution**:
```bash
# Check if socket exists
ls -la /tmp/tt-comfy.sock

# If not, start bridge server
cd /home/tt-admin/tt-metal
python models/experimental/stable_diffusion_xl_base/tt/tt_sdxl_server.py
```

### "Failed to connect to server"

**Cause**: Socket path incorrect or permissions issue

**Solution**:
```bash
# Check socket permissions
ls -la /tmp/tt-comfy.sock

# If needed, set environment variable
export TT_COMFY_SOCKET=/tmp/tt-comfy.sock
```

### "Model initialization timeout"

**Cause**: First-time compilation taking longer than expected

**Solution**:
- Be patient, first load can take 5+ minutes
- Check bridge server logs for compilation progress
- Subsequent loads will be much faster (cached)

### Images are black or corrupted

**Cause**: Bridge server crashed or returned invalid data

**Solution**:
- Check bridge server logs for errors
- Restart bridge server
- Verify model weights are accessible

### Memory errors

**Cause**: Not enough device memory for model + activations

**Solution**:
- Use TT_UnloadModel to unload unused models
- Reduce batch size (generate one image at a time)
- Restart bridge server to clear memory

## Performance Tips

1. **First Run**: Expect 5+ minutes for compilation (one-time cost)
2. **Cached Runs**: Subsequent generations should be faster
3. **Memory Management**: Unload models when switching between model types
4. **Batch Size**: Currently optimized for single-image generation
5. **Resolution**: 1024x1024 is optimal for SDXL

## Limitations

- **Model Support**: Currently SDXL only (SD3.5 and SD1.4 coming soon)
- **Img2Img**: Not yet implemented (text-to-image only)
- **ControlNet**: Not yet supported
- **LoRA**: Not yet supported
- **Batch Generation**: Single image per inference call

## Development

### File Structure

- `__init__.py`: Node registration for ComfyUI
- `nodes.py`: Node implementations (TT_CheckpointLoader, TT_FullDenoise, etc.)
- `wrappers.py`: Lightweight model wrappers (TTModelWrapper, TTCLIPWrapper, TTVAEWrapper)
- `utils.py`: Helper functions (get_model_config, format_bytes, etc.)

### Adding New Nodes

1. Create node class in `nodes.py` with INPUT_TYPES and execution function
2. Register in `__init__.py` NODE_CLASS_MAPPINGS
3. Add display name in NODE_DISPLAY_NAME_MAPPINGS
4. Restart ComfyUI to load new nodes

## License

SPDX-License-Identifier: Apache-2.0

SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

## Support

For issues or questions:
- Check bridge server logs
- Verify socket connection
- Ensure hardware is properly initialized
- Review this README for troubleshooting steps
