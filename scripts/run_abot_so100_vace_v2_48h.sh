#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ABOT_ROOT="${ABOT_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)/abot-physworld}"
OPEN_ROOT="${OPEN_ROOT:-$(cd "$PROJECT_ROOT/.." && pwd)/data_challenge}"
DIT_CHECKPOINT="${DIT_CHECKPOINT:-$PROJECT_ROOT/models/Abot-PhysWorld/abotpw_i2v_480p.safetensors}"
OUTPUT_PATH="${OUTPUT_PATH:-$PROJECT_ROOT/outputs/abot_so100_vace_v2_48h}"
PYTHON_BIN="${PYTHON_BIN:-python}"

test -f "$DIT_CHECKPOINT" || {
  echo "Missing official ABot robot SFT DiT: $DIT_CHECKPOINT" >&2
  exit 2
}
test -f "$PROJECT_ROOT/outputs/abot_so100_train/metadata.jsonl"
test -f "$PROJECT_ROOT/artifacts/cosmos_predict25/action_robust_stats_fold17.json"

cd "$PROJECT_ROOT"
export PYTHONPATH="src:.${PYTHONPATH:+:$PYTHONPATH}"

exec "$PYTHON_BIN" scripts/train_abot_so100_vace_v2.py \
  --abot_root "$ABOT_ROOT" \
  --action_stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --dataset_base_path "$OPEN_ROOT/data/train" \
  --dataset_metadata_path outputs/abot_so100_train/metadata.jsonl \
  --data_file_keys video \
  --height 480 --width 640 --num_frames 17 \
  --model_id_with_origin_paths 'Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
  --dit_checkpoint "$DIT_CHECKPOINT" \
  --learning_rate 5e-6 --weight_decay 0.01 \
  --num_epochs 100000 --max_train_steps 0 --max_train_seconds 172800 \
  --save_steps 1000 --keep_last_checkpoints 4 --keep_every_n_steps 5000 \
  --output_path "$OUTPUT_PATH" \
  --extra_inputs input_image \
  --chunk_num_frames 17 --min_stride 1 --max_stride 1 \
  --dataset_video_resize_mode stretch \
  --use_gradient_checkpointing_offload \
  --dataset_num_workers 1 \
  "$@"
