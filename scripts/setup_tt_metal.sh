#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

################################################################################
# Set up the tt-metal the media server runs against
#
# Clones tt-metal next to ComfyUI (or reuses the checkout already there), checks
# out the commit pinned in TT_METAL_REF, installs system packages, builds,
# creates tt-metal's python_env, and installs the media server's dependencies
# into it. Asks before each slow or sudo step. Declining the build on an
# already-built checkout still installs the server dependencies.
#
# An existing checkout is never moved off a commit that already contains the
# pin, and never touched if it has local changes.
#
# Usage:
#   ./setup_tt_metal.sh [--yes] [--skip-deps] [--metal-home DIR]
#
# Options:
#   -h, --help        Show this help and exit
#   --yes             Answer yes to every prompt
#   --skip-deps       Do not run install_dependencies.sh (needs sudo)
#   --metal-home DIR  tt-metal checkout to use (default: TT_METAL_HOME, else
#                     tt-metal next to ComfyUI)
################################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PACKAGE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly COMFYUI_DIR="${COMFYUI_DIR:-$(cd "${PACKAGE_DIR}/../.." && pwd)}"
readonly REF_FILE="${PACKAGE_DIR}/TT_METAL_REF"
readonly METAL_URL="https://github.com/tenstorrent/tt-metal.git"

METAL_HOME="${TT_METAL_HOME:-$(cd "${COMFYUI_DIR}/.." && pwd)/tt-metal}"
ASSUME_YES=false
SKIP_DEPS=false

info()    { echo -e "\033[0;36m[INFO]\033[0m $*"; }
success() { echo -e "\033[0;32m[SUCCESS]\033[0m $*"; }
warn()    { echo -e "\033[0;33m[WARN]\033[0m $*"; }
error()   { echo -e "\033[0;31m[ERROR]\033[0m $*" >&2; }
status()  { echo -e "\033[1;34m===> $*\033[0m"; }

show_help() { sed -n '8,27p' "$0" | sed 's/^# \{0,1\}//'; }

# Never moves an existing local branch: one by the pinned name is only reused
# when it already contains the pin, otherwise the pin is checked out detached.
checkout_pin() {
    if ! git -C "${METAL_HOME}" show-ref --verify --quiet "refs/heads/${PIN_BRANCH}"; then
        git -C "${METAL_HOME}" checkout -b "${PIN_BRANCH}" "${PIN_COMMIT}"
    elif git -C "${METAL_HOME}" merge-base --is-ancestor "${PIN_COMMIT}" "${PIN_BRANCH}" \
         && git -C "${METAL_HOME}" checkout "${PIN_BRANCH}" 2>/dev/null; then
        :
    else
        warn "Cannot use local branch ${PIN_BRANCH}; checking out ${PIN_COMMIT:0:11} detached instead."
        git -C "${METAL_HOME}" checkout --detach "${PIN_COMMIT}"
    fi
}

confirm() {
    if [[ "${ASSUME_YES}" == "true" ]]; then return 0; fi
    local reply
    read -r -p "$1 [y/N] " reply
    [[ "${reply}" =~ ^[Yy]$ ]]
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help) show_help; exit 0 ;;
        --yes) ASSUME_YES=true; shift ;;
        --skip-deps) SKIP_DEPS=true; shift ;;
        --metal-home) METAL_HOME="$2"; shift 2 ;;
        *) error "Unknown option: $1"; echo "Run with --help for usage."; exit 1 ;;
    esac
done

PIN_BRANCH="$(sed -n 's/^branch=//p' "${REF_FILE}")"
PIN_COMMIT="$(sed -n 's/^commit=//p' "${REF_FILE}")"
[[ -n "${PIN_COMMIT}" ]] || { error "No commit= line in ${REF_FILE}"; exit 1; }

status "tt-metal at ${METAL_HOME}, pinned to ${PIN_BRANCH} (${PIN_COMMIT:0:11})"

#===============================================================================
# Checkout
#===============================================================================
if [[ ! -e "${METAL_HOME}" ]]; then
    confirm "Clone tt-metal into ${METAL_HOME}? (a few GB)" || { info "Stopped before cloning."; exit 0; }
    git clone "${METAL_URL}" "${METAL_HOME}"
    checkout_pin
elif [[ ! -d "${METAL_HOME}/.git" && ! -f "${METAL_HOME}/.git" ]]; then
    error "${METAL_HOME} exists but is not a git checkout"
    exit 1
else
    git -C "${METAL_HOME}" cat-file -e "${PIN_COMMIT}^{commit}" 2>/dev/null \
        || git -C "${METAL_HOME}" fetch origin "${PIN_BRANCH}"
    if git -C "${METAL_HOME}" merge-base --is-ancestor "${PIN_COMMIT}" HEAD; then
        info "Existing checkout already contains the pinned commit; leaving it where it is."
    elif [[ -n "$(git -C "${METAL_HOME}" status --porcelain --untracked-files=no)" ]]; then
        error "${METAL_HOME} has local changes and does not contain the pinned commit."
        error "Commit or stash them, or point --metal-home at another checkout."
        exit 1
    else
        warn "${METAL_HOME} is on $(git -C "${METAL_HOME}" rev-parse --short HEAD), which lacks the pinned commit."
        confirm "Check out ${PIN_BRANCH} at ${PIN_COMMIT:0:11}?" || { info "Stopped; checkout unchanged."; exit 0; }
        checkout_pin
    fi
fi
git -C "${METAL_HOME}" submodule update --init --recursive

#===============================================================================
# Build
#===============================================================================
cd "${METAL_HOME}"
if [[ "${SKIP_DEPS}" == "true" ]]; then
    info "Skipping system packages (--skip-deps)."
elif confirm "Install tt-metal's system packages with sudo (install_dependencies.sh)?"; then
    sudo ./install_dependencies.sh
else
    info "Skipping system packages; the build fails if any are missing."
fi

if confirm "Build tt-metal now? (10 minutes or more)"; then
    ./build_metal.sh
elif [[ -f python_env/bin/activate ]]; then
    info "Skipping the build; using the existing build and python_env."
else
    info "Stopped before building."
    exit 0
fi

if [[ -f python_env/bin/activate ]]; then
    info "python_env already exists; keeping it."
else
    ./create_venv.sh
fi

#===============================================================================
# Server dependencies
#===============================================================================
# python_env is created by uv and has no pip, so install with uv. Fall back to
# bootstrapping pip via ensurepip only if uv is unavailable.
VENV_PY="${METAL_HOME}/python_env/bin/python"
REQ_FILE="${PACKAGE_DIR}/media_server/requirements-server.txt"
status "Installing the media server's dependencies into python_env"
UV_BIN="$(command -v uv || echo "${HOME}/.local/bin/uv")"
if [[ -x "${UV_BIN}" ]]; then
    "${UV_BIN}" pip install --python "${VENV_PY}" -r "${REQ_FILE}"
else
    "${VENV_PY}" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "${VENV_PY}" -m pip install -r "${REQ_FILE}"
fi

success "tt-metal is ready at ${METAL_HOME}"
if [[ "${METAL_HOME}" != "$(cd "${COMFYUI_DIR}/.." && pwd)/tt-metal" ]]; then
    info "It is not next to ComfyUI, so export TT_METAL_HOME=${METAL_HOME} before launching."
fi
info "Launch ComfyUI with: ${SCRIPT_DIR}/launch_with_http.sh"
