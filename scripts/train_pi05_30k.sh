#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LEROBOT_ROOT="${LEROBOT_ROOT:-$ROOT/lerobot}"
BASE_MODEL="${BASE_MODEL:?Set BASE_MODEL to the local pi05_base snapshot}"
DATASET_ROOT="${DATASET_ROOT:-$ROOT/data/pi05_sorting_mimicgen_200eps}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/outputs/pi05_sorting_30k}"
TOKENIZER_PATH="${PI05_TOKENIZER_PATH:?Set PI05_TOKENIZER_PATH to the local PaliGemma tokenizer snapshot}"

[[ -f "$DATASET_ROOT/meta/info.json" ]] || { echo "missing dataset: $DATASET_ROOT" >&2; exit 1; }
[[ -f "$BASE_MODEL/model.safetensors" ]] || { echo "missing Base weights: $BASE_MODEL" >&2; exit 1; }
[[ -f "$TOKENIZER_PATH/tokenizer.json" ]] || { echo "missing tokenizer: $TOKENIZER_PATH" >&2; exit 1; }
[[ ! -e "$OUTPUT_DIR" ]] || { echo "refusing to overwrite: $OUTPUT_DIR" >&2; exit 1; }

export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES:-0,1,2,3}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PI05_TOKENIZER_PATH="$TOKENIZER_PATH" PYTHONPATH="$LEROBOT_ROOT/src:$ROOT"

exec torchrun --standalone --nnodes=1 --nproc-per-node=4 \
  "$(command -v lerobot-train)" \
  --dataset.repo_id=local/pi05_sorting_mimicgen_200eps --dataset.root="$DATASET_ROOT" \
  --dataset.return_uint8=true --dataset.eval_split=0 \
  --policy.type=pi05 --policy.pretrained_path="$BASE_MODEL" --policy.device=cuda \
  --policy.dtype=bfloat16 --policy.push_to_hub=false \
  --policy.normalization_mapping='{"ACTION":"QUANTILES","STATE":"QUANTILES","VISUAL":"IDENTITY"}' \
  --policy.n_action_steps=10 --policy.empty_cameras=1 --policy.use_relative_actions=false \
  --policy.gradient_checkpointing=true --policy.freeze_vision_encoder=false \
  --policy.train_expert_only=true --policy.train_vision_with_expert=true \
  --output_dir="$OUTPUT_DIR" --job_name=pi05_sorting_30k \
  --seed=1000 --num_workers=4 --batch_size=1 \
  --accelerator.gradient_accumulation.steps=16 \
  --steps=30000 --env_eval_freq=0 --eval_steps=0 --log_freq=10 \
  --save_checkpoint=true --save_freq=10000 --ema.enable=false --wandb.enable=false
