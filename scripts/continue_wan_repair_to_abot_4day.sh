#!/usr/bin/env bash
set -u

PROJECT="/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge"
PYTHON_BIN="/home/video_generation/inha_challenge/ChoiSeongYong/.miniforge3/envs/inha_sy/bin/python"
ABOT_ROOT="/home/video_generation/inha_challenge/ChoiSeongYong/abot-physworld"
OUTPUT="$PROJECT/outputs/abot_so100_vace_4day"
REPAIR_LOG="$OUTPUT/wan_prefix_repair.log"
CHAIN_LOG="$OUTPUT/continue_repair_to_train.log"
TRAIN_LOG="$OUTPUT/train.log"
SUCCESS_MARKER="all seven Wan DiT shards passed official SHA-256 and safetensors validation"

mkdir -p "$OUTPUT"
cd "$PROJECT" || exit 1

log() {
    echo "$(date '+%F %T') $*" >> "$CHAIN_LOG"
}

log "waiting for verified Wan prefix repair"
while ! grep -Fq "$SUCCESS_MARKER" "$REPAIR_LOG" 2>/dev/null; do
    if ! pgrep -f '[r]epair_wan_shard_prefixes.py' >/dev/null; then
        log "repair process ended without the success marker; refusing to start training"
        exit 1
    fi
    sleep 30
done
log "repair validation completed"

if pgrep -f '[t]rain_abot_so100_vace.py' >/dev/null; then
    log "an ABot training process already exists; refusing duplicate launch"
    exit 1
fi

# Do not collide with an inference or another compute process that the user may
# have started while the repair was running. Xorg does not appear in this
# compute-app query.
while true; do
    gpu_query=$(nvidia-smi --id=1 --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null)
    gpu_rc=$?
    if [ "$gpu_rc" -ne 0 ]; then
        log "nvidia-smi unavailable; retrying GPU preflight in 60s"
        sleep 60
        continue
    fi
    if [ -z "$gpu_query" ]; then
        break
    fi
    log "GPU 1 is occupied by compute PID(s): $gpu_query; waiting 60s"
    sleep 60
done

log "GPU 1 is free; starting VACE-only training with a 345600-second budget"
echo "" >> "$TRAIN_LOG"
echo "===== automatic verified start $(date '+%F %T') =====" >> "$TRAIN_LOG"

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
    --output_path outputs/abot_so100_vace_4day \
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
    --dataset_num_workers 1 \
    >> "$TRAIN_LOG" 2>&1
