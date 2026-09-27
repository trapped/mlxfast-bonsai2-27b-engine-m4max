#!/usr/bin/env bash
# Add the local Bonsai 2 provider to Pi (~/.pi/agent/models.json) without touching
# other providers or the default model. Backs up both config files first.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
BK="$ROOT/pi/backup/$(date +%Y%m%d-%H%M%S)"; mkdir -p "$BK"
cp -p "$AGENT/models.json" "$AGENT/settings.json" "$BK/" 2>/dev/null || true
# The first backup is the pre-Bonsai state; pi/revert.sh restores it.
[[ -e "$ROOT/pi/backup/original" ]] || ln -sfn "$BK" "$ROOT/pi/backup/original"
python3 - "$AGENT/models.json" "$ROOT/pi/bonsai-provider.json" <<'PY'
import json, os, sys
path, prov = sys.argv[1], json.load(open(sys.argv[2]))
cfg = json.load(open(path)) if os.path.exists(path) else {}
cfg.setdefault("providers", {})["bonsai"] = prov
json.dump(cfg, open(path, "w"), indent=2); open(path, "a").write("\n")
PY
echo "backup: $BK"; echo "added provider 'bonsai' to $AGENT/models.json"
