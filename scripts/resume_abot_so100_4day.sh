#!/usr/bin/env bash
set -euo pipefail

PROJECT="/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge"
PYTHON_BIN="/home/video_generation/inha_challenge/ChoiSeongYong/.miniforge3/envs/inha_sy/bin/python"
ABOT_ROOT="/home/video_generation/inha_challenge/ChoiSeongYong/abot-physworld"
OUTPUT="$PROJECT/outputs/abot_so100_vace_4day"

cd "$PROJECT"
test -s "$OUTPUT/step-7900.safetensors"

if pgrep -f '[t]rain_abot_so100_vace.py' >/dev/null; then
    echo "An ABot training process is already running; refusing duplicate launch." >&2
    exit 2
fi

exec env CUDA_VISIBLE_DEVICES=1 PYTHONPATH=src:. \
    "$PYTHON_BIN" scripts/train_abot_so100_vace.py \
    --abot_root "$ABOT_ROOT" \
    --action_stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
    --dataset_base_path /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge/data/train \
    --dataset_metadata_path outputs/abot_so100_train/metadata.jsonl \
    --data_file_keys video \
    --height 480 --width 640 --num_frames 17 \
    --model_id_with_origin_paths 'Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
    --learning_rate 5e-6 \
    --num_epochs 100000 \
    --max_train_steps 0 \
    --max_train_seconds 345600 \
    --save_steps 100 \
    --output_path "$OUTPUT" \
    --resume_from_step 7900 \
    --remove_prefix_in_ckpt pipe.vace. \
    --trainable_models vace \
    --extra_inputs input_image \
    --disable_text_condition true \
    --init_vace_from_dit true \
    --init_vace_from_dit_vace_in_dim 96 \
    --chunk_num_frames 17 \
    --min_stride 1 --max_stride 1 \
    --dataset_video_resize_mode stretch \
    --use_gradient_checkpointing_offload \
    --dataset_num_workers 1
