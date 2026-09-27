#!/usr/bin/env bash
# Start the OpenAI-compatible server on 127.0.0.1:${BONSAI_PORT:-8000}.
# Tunables (env): BONSAI_SPEC=dflash|mtp|serial  BONSAI_DEPTH  BONSAI_CONTEXT
#                 BONSAI_MAX_OUTPUT  BONSAI_KV_BYTES  BONSAI_PORT
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"; cd "$ROOT"
# config.env supplies machine defaults; variables already set in the environment win.
if [[ -f config.env ]]; then
  while IFS='=' read -r k v; do
    [[ -z "$k" || "$k" == \#* ]] && continue
    [[ -z "${!k+x}" ]] && export "$k=$v"
  done < config.env
fi
exec uv run --frozen python server/bonsai_openai.py "$@"
