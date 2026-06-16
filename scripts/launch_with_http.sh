#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# ComfyUI launcher for the Tenstorrent HTTP nodes
#
# ComfyUI is launched ALONE — no model/board args and no server start here.
# The TT_CheckpointLoader node owns the tt-metal server lifecycle: selecting a
# model in the node spawns and supervises launch_server.sh (sdxl -> p150,
# wan22 -> p300x2) and talks to it over HTTP.
#
# Architecture:
#   ComfyUI (TT_CheckpointLoader -> server_manager.py) --Popen--> tt-metal server
#                                                       --HTTP-->  tt-metal server
#
# This script only:
#   1. Validates the ComfyUI venv
#   2. Launches ComfyUI in the ComfyUI venv
#   3. On exit, as a backstop, kills any tt-metal server the node left running
#      (read from the server_manager PID/lock file)
#
# Usage:
#   ./launch_with_http.sh [--port 8188] [--listen 127.0.0.1]
#
# Options:
#   -h, --help        Show this help and exit
#   --port PORT       ComfyUI listen port (default: 8188)
#   --listen IP       ComfyUI listen address (default: 127.0.0.1)
################################################################################

set -euo pipefail

#===============================================================================
# Configuration
#===============================================================================
readonly COMFYUI_DIR="${COMFYUI_DIR:-/home/stisi/ComfyUI}"
readonly COMFYUI_VENV="${COMFYUI_DIR}/venv"
readonly SERVER_PID_FILE="${TT_SERVER_PID_FILE:-/tmp/tt_comfy_server.pid}"

COMFYUI_PORT="8188"
COMFYUI_LISTEN="127.0.0.1"

#===============================================================================
# Logging helpers
#===============================================================================
info()    { echo -e "\033[0;36m[INFO]\033[0m $*"; }
success() { echo -e "\033[0;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[0;33m[WARN]\033[0m $*"; }
error()   { echo -e "\033[0;31m[ERROR]\033[0m $*" >&2; }
status()  { echo -e "\033[1;34m===> $*\033[0m"; }

#===============================================================================
# Cleanup — backstop kill of a server the node left running
#===============================================================================
cleanup() {
    local exit_code=$?
    if [[ -f "${SERVER_PID_FILE}" ]]; then
        local pgid pid
        pgid=$(grep -o '"pgid"[[:space:]]*:[[:space:]]*[0-9]*' "${SERVER_PID_FILE}" 2>/dev/null | grep -o '[0-9]*' || true)
        pid=$(grep -o '"pid"[[:space:]]*:[[:space:]]*[0-9]*' "${SERVER_PID_FILE}" 2>/dev/null | grep -o '[0-9]*' || true)
        if [[ -n "${pgid}" ]] && kill -0 "-${pgid}" 2>/dev/null; then
            warn "Backstop: terminating tt-metal server process group ${pgid}"
            kill -TERM "-${pgid}" 2>/dev/null || true
            sleep 3
            kill -KILL "-${pgid}" 2>/dev/null || true
        elif [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
            warn "Backstop: terminating tt-metal server PID ${pid}"
            kill -TERM "${pid}" 2>/dev/null || true
        fi
        rm -f "${SERVER_PID_FILE}" 2>/dev/null || true
    fi
    exit "${exit_code}"
}
trap cleanup EXIT INT TERM

#===============================================================================
# Help
#===============================================================================
show_help() { sed -n '7,33p' "$0" | sed 's/^# \{0,1\}//'; }

#===============================================================================
# Argument parsing
#===============================================================================
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        --port) COMFYUI_PORT="$2"; shift 2 ;;
        --listen) COMFYUI_LISTEN="$2"; shift 2 ;;
        *) error "Unknown option: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

#===============================================================================
# Prerequisite validation
#===============================================================================
validate_prerequisites() {
    status "Validating prerequisites"
    local errors=0
    [[ -d "${COMFYUI_DIR}" ]] || { error "ComfyUI dir not found: ${COMFYUI_DIR}"; ((errors++)); }
    [[ -d "${COMFYUI_VENV}" ]] || { error "ComfyUI venv not found: ${COMFYUI_VENV}"; ((errors++)); }
    if [[ ${errors} -gt 0 ]]; then error "Validation failed with ${errors} error(s)"; exit 1; fi
    success "All prerequisites validated"
}

#===============================================================================
# Launch ComfyUI (foreground)
#===============================================================================
launch_comfyui() {
    status "Launching ComfyUI"
    # shellcheck disable=SC1091
    source "${COMFYUI_VENV}/bin/activate"
    cd "${COMFYUI_DIR}"
    echo ""
    success "ComfyUI will be available at: http://${COMFYUI_LISTEN}:${COMFYUI_PORT}"
    info "Pick a model in the TT Checkpoint Loader node to stand up the tt-metal server."
    info "Press Ctrl+C to stop ComfyUI (and any tt-metal server it started)."
    echo ""
    exec python3 main.py --port "${COMFYUI_PORT}" --listen "${COMFYUI_LISTEN}" --tenstorrent
}

#===============================================================================
# Main
#===============================================================================
main() {
    echo ""
    status "ComfyUI + Tenstorrent (node-driven server standup)"
    echo ""
    validate_prerequisites
    launch_comfyui
}

main "$@"
