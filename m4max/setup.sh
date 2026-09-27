#!/usr/bin/env bash
# One-shot setup for this fork on an M4 Max: toolchain, upstream's own setup (build +
# pinned artifact downloads), the weight transform, bonsai-serve, and the Python shim env.
set -euo pipefail
M="$(cd "$(dirname "$0")" && pwd)"; ROOT="$(dirname "$M")"; cd "$ROOT"
HW="$("$M/scripts/detect-hw.sh")"; echo "$HW"
if [[ "$HW" == *"nax_capable=yes"* ]]; then
  echo "WARNING: this is an M5-class GPU. m4max/config.env disables M5 tensor routes and the"
  echo "         verify-row padding tuned for NAX; use upstream main instead, or edit config.env." >&2
fi
if ! xcrun metal --version >/dev/null 2>&1; then
  xcodebuild -runFirstLaunch
  xcodebuild -downloadComponent MetalToolchain
fi
./setup.sh   # upstream: builds bench-worker + mlx.metallib, downloads/verifies target, MTP head, DFlash 2
REV="$(git rev-parse HEAD)"
# The transform's output format changes between upstream revisions: redo it whenever HEAD moves.
if [[ ! -f weights/config.json || "$(cat weights/.m4max-rev 2>/dev/null)" != "$REV" ]]; then
  rm -rf weights
  .build/release/mlxfast-swift transform --reference reference_weights/Ternary-Bonsai-2-27B-mlx-2bit --output weights
  echo "$REV" > weights/.m4max-rev
fi
swift build -c release --force-resolved-versions --scratch-path .build-worker --product bonsai-serve
(cd "$M" && uv sync --frozen)
echo "setup complete. start with: m4max/run-server.sh  (or m4max/scripts/launchd.sh install)"
