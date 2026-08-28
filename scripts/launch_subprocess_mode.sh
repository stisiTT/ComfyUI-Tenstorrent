#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# ComfyUI launcher — Tenstorrent nodes in SUBPROCESS mode
#
# The node's docker mode (TT_LAUNCH_MODE=docker, the default) drives a prebuilt
# comfyui-media-server image via tt-inference-server's run.py. Until that image
# is rebuilt past the TT_METAL_HOME fix it crashes wan22 on p300x2 with:
#   Custom mesh graph descriptor file not found:
#   .../comfyui-media-server/tt_metal/fabric/mesh_graph_descriptors/p300_x2_mesh_graph_descriptor.textproto
#
# This script instead selects the known-good host path: spawn
# tt-inference-server/comfyui-media-server/launch_server.sh directly against an
# already-built tt-metal checkout (no Docker, no image).
#
# The two paths below must come from the SAME stack — the relocated server calls
# fuse_lora(lora_scale, clip_scale), which only the per-component-LoRA tt-metal
# branch accepts. See custom_nodes/tenstorrent_nodes/INTEGRATION.md.
#
# Usage:
#   ./launch_subprocess_mode.sh [--port 8188] [--listen 127.0.0.1]
#   (any args are passed straight through to launch_with_http.sh)
#
# Layout assumed (override either path via the environment):
#   <parent>/ComfyUI              <- this repo
#   <parent>/tt-inference-server  <- holds comfyui-media-server/
#   <parent>/tt-metal             <- built tt-metal (supplies ttnn / models.*)
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PARENT_DIR="$(dirname "${SCRIPT_DIR}")"

export TT_LAUNCH_MODE=subprocess
# Directory holding launch_server.sh (the relocated server).
export TT_METAL_DIR="${TT_METAL_DIR:-${PARENT_DIR}/tt-inference-server/comfyui-media-server}"
# Built tt-metal checkout the server imports ttnn / models.* from.
export TT_METAL_HOME="${TT_METAL_HOME:-${PARENT_DIR}/tt-metal}"

if [[ ! -x "${TT_METAL_DIR}/launch_server.sh" ]]; then
    echo "error: ${TT_METAL_DIR}/launch_server.sh not found or not executable." >&2
    echo "       Set TT_METAL_DIR to your tt-inference-server/comfyui-media-server dir." >&2
    exit 1
fi
if [[ ! -d "${TT_METAL_HOME}/tt_metal" ]]; then
    echo "error: ${TT_METAL_HOME} does not look like a tt-metal checkout (no tt_metal/)." >&2
    echo "       Set TT_METAL_HOME to your built tt-metal checkout." >&2
    exit 1
fi

echo "TT_LAUNCH_MODE=subprocess"
echo "TT_METAL_DIR=${TT_METAL_DIR}"
echo "TT_METAL_HOME=${TT_METAL_HOME}"

exec "${SCRIPT_DIR}/launch_with_http.sh" "$@"
