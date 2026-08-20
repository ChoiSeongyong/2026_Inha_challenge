#!/usr/bin/env bash
set -euo pipefail

PROJECT=/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge
UPSTREAM=/home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5
PYTHON="$UPSTREAM/.venv/bin/python"
OUTPUT="$PROJECT/outputs/cosmos_predict25_so100_until_wan_ready"
CKPT="$PROJECT/outputs/cosmos_predict25_so100_continue_14k_to_32k/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/iter_000032000"
WAN="$PROJECT/models/Wan-AI"

cd "$PROJECT"
mkdir -p "$OUTPUT"

wan_ready() {
  WAN_ROOT="$WAN" python3 - <<'PY'
from pathlib import Path
import os

root = Path(os.environ["WAN_ROOT"])
i2v = root / "Wan2.1-I2V-14B-480P"
t2v = root / "Wan2.1-T2V-1.3B"
expected = {
    "diffusion_pytorch_model-00001-of-00007.safetensors": 9847223624,
    "diffusion_pytorch_model-00002-of-00007.safetensors": 9797021016,
    "diffusion_pytorch_model-00003-of-00007.safetensors": 9797041744,
    "diffusion_pytorch_model-00004-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00005-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00006-of-00007.safetensors": 9692142864,
    "diffusion_pytorch_model-00007-of-00007.safetensors": 7061933536,
    "models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth": 4772359047,
}
if any(not (i2v / name).is_file() or (i2v / name).stat().st_size != size for name, size in expected.items()):
    raise SystemExit(1)
for name in ("models_t5_umt5-xxl-enc-bf16.pth", "Wan2.1_VAE.pth"):
    if not (t2v / name).is_file() or (t2v / name).stat().st_size <= 0:
        raise SystemExit(1)
PY
}

echo "[cosmos-watch] starting resume from $CKPT at $(date -Is)"
setsid env CUDA_VISIBLE_DEVICES=1 PYTHONPATH="$PROJECT:$UPSTREAM" \
  "$PYTHON" scripts/run_cosmos_predict25_so100.py \
  --upstream-root "$UPSTREAM" \
  --open-root /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge \
  --output-root "$OUTPUT" \
  --max-iter 1000000 \
  --grad-accum-iter 8 \
  --master-port 12346 \
  --override model.config.net.atten_backend=transformer_engine \
  --override checkpoint.load_path="$CKPT" \
  --override checkpoint.load_training_state=true \
  --override checkpoint.strict_resume=false \
  --override checkpoint.save_iter=1000 \
  --execute \
  > "$OUTPUT/launcher.log" 2>&1 &
COSMOS_PID=$!
echo "$COSMOS_PID" > "$OUTPUT/cosmos_watch_pid"
echo "[cosmos-watch] process_group=$COSMOS_PID"

while kill -0 "$COSMOS_PID" 2>/dev/null; do
  if wan_ready; then
    echo "[cosmos-watch] all Wan files validated at $(date -Is); stopping Cosmos"
    kill -INT -- "-$COSMOS_PID" 2>/dev/null || true
    sleep 90
    kill -TERM -- "-$COSMOS_PID" 2>/dev/null || true
    wait "$COSMOS_PID" || true
    find "$OUTPUT" -type d -path '*/checkpoints/iter_*' -printf '%T@ %p\n' 2>/dev/null | sort -n | tail -1 \
      | tee "$OUTPUT/last_checkpoint_at_stop.txt"
    exit 0
  fi
  sleep 60
done

echo "[cosmos-watch] Cosmos exited before Wan download completed at $(date -Is)" >&2
exit 1
