#!/usr/bin/env bash
# Cooled A/B of prefill (4k-token prompt) between two env configs. Stop the server first: scripts/launchd.sh uninstall
cd "$(dirname "$0")/.."; set -a; source config.env; set +a
for i in $(seq 1 ${3:-2}); do for cfg in "$1" "$2"; do
  sleep 60
  r=$(env $cfg ../.build-worker/release/bonsai-serve --weights ../weights --prefill-chunk 512 < bench/req4k.jsonl 2>/dev/null | tail -1)
  echo "[$cfg] $(echo $r | python3 -c 'import json,sys;d=json.load(sys.stdin);print("ttft_s",round(d["ttft_ms"]/1000,2),"prefill tok/s",round(d["prompt_tokens"]/(d["ttft_ms"]/1000),1))')"
done; done
