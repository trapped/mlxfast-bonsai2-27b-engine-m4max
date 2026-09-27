#!/usr/bin/env bash
# Restore Pi's models.json/settings.json to the state before pi/install.sh first ran.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
AGENT="${PI_CODING_AGENT_DIR:-$HOME/.pi/agent}"
BK="${1:-$ROOT/pi/backup/original}"
cp -p "$BK/models.json" "$BK/settings.json" "$AGENT/"
echo "restored Pi config from $(readlink "$BK" || echo "$BK")"
