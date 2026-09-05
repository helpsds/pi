#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${BASE_MODEL:?Set BASE_MODEL to the local pi05_base snapshot}"
FT="${FT_MODEL:?Set FT_MODEL to the fine-tuned pretrained_model directory}"
OUT="$ROOT/outputs/pi05_sorting_base_vs_ft_10seeds"
WORKER="$ROOT/scripts/eval_pi05_sorting_multiseed_worker.sh"
mkdir -p "$OUT"

bash "$WORKER" base "$BASE" 0 2 6200 "$OUT/base" >"$OUT/base.log" 2>&1 &
base_pid=$!
bash "$WORKER" ft_030000 "$FT" 1 3 6300 "$OUT/ft_030000" >"$OUT/ft_030000.log" 2>&1 &
ft_pid=$!

status=0
wait "$base_pid" || status=1
wait "$ft_pid" || status=1
[[ "$status" -eq 0 ]] || { echo "one or both workers failed" >&2; exit 1; }

python - "$OUT" <<'PY'
import json, sys
from pathlib import Path
root = Path(sys.argv[1])
base = json.loads((root / "base/summary.json").read_text())
ft = json.loads((root / "ft_030000/summary.json").read_text())
payload = {
    "protocol": {"tasks": 8, "seeds_per_task": 10, "episodes_per_model": 80,
                 "environment_seeds": list(range(12000, 12010)), "n_action_steps": 10},
    "base": {k: v for k, v in base.items() if k != "rows"},
    "ft_030000": {k: v for k, v in ft.items() if k != "rows"},
    "success_rate_delta": ft["success_rate"] - base["success_rate"],
}
(root / "summary.json").write_text(json.dumps(payload, indent=2))
print(json.dumps(payload, indent=2))
PY
