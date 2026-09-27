#!/usr/bin/env bash
# Cooled A/B of decode step time (p50 over 400 tokens) between two env configs. Stop the server first.
# usage: ab.sh "ENV_A" "ENV_B" rounds  -> p50 decode step ms per config, cooled between runs
cd "$(dirname "$0")/.."; set -a; source config.env; set +a
for i in $(seq 1 ${3:-2}); do for cfg in "$1" "$2"; do
  sleep 90
  ms=$( (echo '{"prompt_tokens":[248045,846,198,9707,248046,198,248045,74455,198],"max_tokens":400,"stop_tokens":[]}') | env $cfg CBV2_STEP_PROFILE=1 ../.build-worker/release/bonsai-serve --weights ../weights 2>&1 >/dev/null | grep "v2.step.wall" | awk -F'|' '{print $6}')
  echo "[$cfg] p50 step ms: $ms"
done; done
