#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
#
# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC

# Kept so existing commands keep working. The media server now ships in this
# package (media_server/), so there is no separate server checkout to point at
# and launch_with_http.sh needs no environment. Arguments pass straight through.
echo "launch_subprocess_mode.sh is no longer needed; use ./launch_with_http.sh" >&2
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/launch_with_http.sh" "$@"
