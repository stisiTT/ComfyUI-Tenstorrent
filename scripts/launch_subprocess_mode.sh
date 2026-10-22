#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# ComfyUI launcher — Tenstorrent nodes in SUBPROCESS mode
#
# Subprocess mode is already the node's default, but its default TT_METAL_DIR is
# ../tt-metal — the two-repo stack, where the server sits at the tt-metal repo
# root. This script selects the OTHER subprocess stack: the relocated server in
# tt-inference-server/comfyui-media-server, run against a built tt-metal checkout
# (still no Docker, no image).
#
# Use this only if you are on the relocated stack. For the two-repo default,
# ./launch_with_http.sh on its own is enough.
#
# The two paths below must come from the SAME stack — the relocated server calls
# fuse_lora(lora_scale, clip_scale), which only the per-component-LoRA tt-metal
# branch accepts. See INTEGRATION.md.
#
# Usage:
#   ./launch_subprocess_mode.sh [--port 8188] [--listen 127.0.0.1]
#   (any args are passed straight through to launch_with_http.sh)
#
# Layout assumed (override either path via the environment):
#   <parent>/ComfyUI              <- stock ComfyUI, this repo in custom_nodes/
#   <parent>/tt-inference-server  <- holds comfyui-media-server/
#   <parent>/tt-metal             <- built tt-metal (supplies ttnn / models.*)
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# <parent>/ComfyUI/custom_nodes/ComfyUI-Tenstorrent/scripts -> <parent>
PARENT_DIR="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

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
