#!/usr/bin/env bash
# Rebase this fork's M4 Max commits onto the latest upstream main, rebuild, and re-check.
#   m4max/sync-upstream.sh            # rebase + rebuild + correctness gate
#   git push --force-with-lease       # publish once you are happy with the numbers
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"; cd "$ROOT"
git remote get-url upstream >/dev/null 2>&1 || git remote add upstream https://github.com/Layr-Labs/mlxfast-bonsai2-27b-engine.git
git fetch upstream
echo "new upstream commits:"; git log --oneline HEAD..upstream/main | cat
git rebase upstream/main
m4max/setup.sh
# Upstream's own gate against the shipped golden, with the M4 knobs on.
set -a; source m4max/config.env; set +a
./tools/fetch-benchd.sh >/dev/null
./benchmark.sh --local-iterate || true
grep -E '"passed_correctness"|"correctness_checked_steps"' score.local-iterate.json
echo "Now compare speed before pushing, e.g.: (cd m4max && uv run python bench/spec-ab.py dflash 3 && uv run python bench/quote-ab.py dflash 3)"
