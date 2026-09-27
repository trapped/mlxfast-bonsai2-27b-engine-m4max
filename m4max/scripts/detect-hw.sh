#!/usr/bin/env bash
# Print this Mac's Apple Silicon generation and whether M5-only NAX kernels apply.
# MLX gates NAX itself (device.cpp is_nax_available: GPU arch gen >= 17 and
# macOS >= 26.2); this script only reports, it never forces NAX on.
set -euo pipefail
brand=$(sysctl -n machdep.cpu.brand_string)
mem_gb=$(( $(sysctl -n hw.memsize) / 1073741824 ))
gen=$(echo "$brand" | grep -oE 'M[0-9]+' | head -1 | tr -d M)
nax=no; [[ "${gen:-0}" -ge 5 ]] && nax=yes
echo "chip=\"$brand\" generation=M${gen} memory_gb=${mem_gb} macos=$(sw_vers -productVersion) nax_capable=${nax}"
