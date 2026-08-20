#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="$ROOT/models/Wan-AI/Wan2.1-I2V-14B-480P"
SHARED_DIR="$ROOT/models/Wan-AI/Wan2.1-T2V-1.3B"
LOG_DIR="$ROOT/outputs/abot_so100_vace_1800/download_logs"
mkdir -p "$MODEL_DIR" "$SHARED_DIR" "$LOG_DIR"

download_one() {
  local repo="$1"
  local dir="$2"
  local name="$3"
  local target="$dir/$name"
  local part="$target.part"
  local url="https://huggingface.co/$repo/resolve/main/$name?download=true"

  if [[ -s "$target" ]]; then
    echo "[$(date -Is)] complete $repo/$name"
    return 0
  fi
  if [[ -f "$dir/$name.incomplete" && ! -f "$part" ]]; then
    mv "$dir/$name.incomplete" "$part"
  fi
  echo "[$(date -Is)] start $repo/$name"
  curl --fail --location --retry 20 --retry-delay 5 --retry-all-errors \
    --continue-at - --output "$part" "$url"
  mv "$part" "$target"
  echo "[$(date -Is)] complete $repo/$name"
}

export -f download_one
export MODEL_DIR SHARED_DIR LOG_DIR

for i in 1 2 3 4 5 6 7; do
  printf -v shard 'diffusion_pytorch_model-%05d-of-00007.safetensors' "$i"
  download_one Wan-AI/Wan2.1-I2V-14B-480P "$MODEL_DIR" "$shard" \
    >"$LOG_DIR/$shard.log" 2>&1 &
done
wait

download_one Wan-AI/Wan2.1-I2V-14B-480P "$MODEL_DIR" \
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
  >"$LOG_DIR/clip.log" 2>&1
download_one Wan-AI/Wan2.1-T2V-1.3B "$SHARED_DIR" \
  models_t5_umt5-xxl-enc-bf16.pth >"$LOG_DIR/t5.log" 2>&1
download_one Wan-AI/Wan2.1-T2V-1.3B "$SHARED_DIR" \
  Wan2.1_VAE.pth >"$LOG_DIR/vae.log" 2>&1

echo "[$(date -Is)] all Wan HF direct downloads complete"
