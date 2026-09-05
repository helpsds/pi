#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LABEL="$1"
MODEL="$2"
MODEL_GPU="$3"
EGL_GPU="$4"
PORT_BASE="$5"
OUT="$6"
DATA="${DATASET_ROOT:-$ROOT/data/pi05_sorting_mimicgen_200eps}"
TOKENIZER_PATH="${PI05_TOKENIZER_PATH:?Set PI05_TOKENIZER_PATH to a local PaliGemma tokenizer snapshot}"
SIM_CONDA_ENV="${SIM_CONDA_ENV:-robocasa}"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROOT/lerobot}"
OBJECTS=(red_cube red_cube green_cube green_cube cylinder cylinder small_box small_box)
BINS=(red_bin blue_bin red_bin blue_bin red_bin blue_bin red_bin blue_bin)
PIDS=()

[[ -f "$MODEL/model.safetensors" ]] || { echo "missing model: $MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/tokenizer.json" ]] || { echo "missing local tokenizer" >&2; exit 1; }
mkdir -p "$OUT"

cleanup() {
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# Ten new environment seeds; Base and FT receive exactly the same list.
for seed in $(seq 12000 12009); do
    batch="$OUT/seed_$seed"
    completed="$( { rg -l 'SORTING_RESULT=' "$batch" -g server.log 2>/dev/null || true; } | wc -l)"
    if [[ "$completed" -eq 8 ]]; then
        echo "[$LABEL] skipping completed seed $seed"
        continue
    fi
    [[ ! -e "$batch" ]] || { echo "[$LABEL] incomplete batch exists: $batch" >&2; exit 1; }
    PIDS=()
    ports=()
    policy_seeds=()
    for index in $(seq 0 7); do
        port=$((PORT_BASE + index))
        # Make policy sampling deterministic but distinct across task and environment seed.
        policy_seed=$((seed * 10 + index))
        task_dir="$batch/task_${index}_${OBJECTS[$index]}_to_${BINS[$index]}"
        mkdir -p "$task_dir"
        (
            source "$CONDA_SH"
            conda activate "$SIM_CONDA_ENV"
            export CUDA_VISIBLE_DEVICES=0,1,2,3
            export PYTHONNOUSERSITE=1 PYTHONPATH="$ROOT/third_party/robosuite:$ROOT"
            export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="$EGL_GPU"
            exec python -u "$ROOT/scripts/sorting_eval_server.py" \
                --target-object "${OBJECTS[$index]}" --target-bin "${BINS[$index]}" \
                --seed "$seed" --horizon 300 --port "$port" --video "$task_dir/rollout.mp4"
        ) >"$task_dir/server.log" 2>&1 &
        PIDS+=("$!")
        ports+=("$port")
        policy_seeds+=("$policy_seed")
    done
    ready=0
    for _ in $(seq 1 180); do
        ready=0
        for index in $(seq 0 7); do
            task_dir="$batch/task_${index}_${OBJECTS[$index]}_to_${BINS[$index]}"
            grep -qi 'waiting for policy client' "$task_dir/server.log" 2>/dev/null && ready=$((ready + 1))
        done
        [[ "$ready" -eq 8 ]] && break
        sleep 1
    done
    [[ "$ready" -eq 8 ]] || { echo "[$LABEL] only $ready/8 servers ready for seed $seed" >&2; exit 1; }
    (
        export CUDA_VISIBLE_DEVICES="$MODEL_GPU" TOKENIZERS_PARALLELISM=false PYTHONNOUSERSITE=1
        export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
        export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
        export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PI05_TOKENIZER_PATH="$TOKENIZER_PATH"
        export PYTHONPATH="$LEROBOT_ROOT/src:$ROOT/scripts:$ROOT"
        exec "${POLICY_PYTHON:-$ROOT/.venv/bin/python}" -u "$ROOT/scripts/pi05_sorting_multi_client.py" \
            --model "$MODEL" --dataset-root "$DATA" --repo-id local/pi05_sorting_mimicgen_200eps \
            --ports "${ports[@]}" --policy-seeds "${policy_seeds[@]}" --n-action-steps 10
    ) >"$batch/client.log" 2>&1
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    PIDS=()
done

python - "$LABEL" "$OUT" <<'PY'
import json, re, sys
from pathlib import Path
label, root = sys.argv[1], Path(sys.argv[2])
rows = []
for log in sorted(root.glob("seed_*/task_*/server.log")):
    matches = re.findall(r"SORTING_RESULT=(\{.*\})", log.read_text())
    if len(matches) != 1:
        raise RuntimeError(f"{log}: expected one result, got {len(matches)}")
    rows.append(json.loads(matches[0]))
successes = sum(bool(row["success"]) for row in rows)
grasped = sum(bool(row["grasp_success"]) for row in rows)
payload = {
    "label": label,
    "n_action_steps": 10,
    "successes": successes,
    "episodes": len(rows),
    "success_rate": successes / len(rows),
    "grasp_successes": grasped,
    "grasp_rate": grasped / len(rows),
    "rows": rows,
}
(root / "summary.json").write_text(json.dumps(payload, indent=2))
print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
PY
