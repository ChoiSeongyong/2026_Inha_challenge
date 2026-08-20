# Wan2.1-I2V-14B MacBook 다운로드 및 서버 전송

MacBook에서 Wan 14B를 다운로드하고 학습 서버로 전송하는 절차입니다.

## 1. MacBook 준비

Terminal에서 Python 가상환경과 ModelScope를 준비합니다.

```bash
python3 --version
python3 -m venv ~/wan_download_env
source ~/wan_download_env/bin/activate
python -m pip install -U modelscope
```

## 2. MacBook에서 다운로드

```bash
mkdir -p "$HOME/wan_models/Wan2.1-I2V-14B-480P"
mkdir -p "$HOME/wan_models/Wan2.1-T2V-1.3B"
```

Wan 14B:

```bash
modelscope download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir "$HOME/wan_models/Wan2.1-I2V-14B-480P" \
  --max-workers 8
```

T5/VAE:

```bash
modelscope download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir "$HOME/wan_models/Wan2.1-T2V-1.3B" \
  --max-workers 4
```

중단되면 같은 명령을 다시 실행해 재개합니다. `.incomplete` 파일만 남아 있으면 완료된 것이 아닙니다.

```bash
du -sh "$HOME/wan_models/Wan2.1-I2V-14B-480P"
find "$HOME/wan_models/Wan2.1-I2V-14B-480P" -maxdepth 1 -name '*.incomplete'
```

## 3. MacBook 다운로드 결과

다음 파일이 있어야 합니다.

```text
~/wan_models/Wan2.1-I2V-14B-480P/
  diffusion_pytorch_model-00001-of-00007.safetensors
  diffusion_pytorch_model-00002-of-00007.safetensors
  diffusion_pytorch_model-00003-of-00007.safetensors
  diffusion_pytorch_model-00004-of-00007.safetensors
  diffusion_pytorch_model-00005-of-00007.safetensors
  diffusion_pytorch_model-00006-of-00007.safetensors
  diffusion_pytorch_model-00007-of-00007.safetensors
  models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth

~/wan_models/Wan2.1-T2V-1.3B/
  models_t5_umt5-xxl-enc-bf16.pth
  Wan2.1_VAE.pth
```

## 4. 서버의 기존 다운로드 중지

서버에서 실행합니다.

```bash
pkill -f 'modelscope download'
cd /home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge
find models/Wan-AI/Wan2.1-I2V-14B-480P -maxdepth 1 -name '*.incomplete' -delete
```

## 5. MacBook에서 서버로 전송

macOS에는 보통 `rsync`가 포함되어 있습니다.

```bash
rsync --version
```

Wan 14B 전송:

```bash
rsync -avP --info=progress2 \
  "$HOME/wan_models/Wan2.1-I2V-14B-480P/" \
  video_generation@aigpu0618:/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge/models/Wan-AI/Wan2.1-I2V-14B-480P/
```

T5/VAE 전송:

```bash
rsync -avP --info=progress2 \
  "$HOME/wan_models/Wan2.1-T2V-1.3B/" \
  video_generation@aigpu0618:/home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge/models/Wan-AI/Wan2.1-T2V-1.3B/
```

처음 접속할 때 fingerprint가 나오면 서버가 맞는지 확인한 뒤 `yes`를 입력합니다. 전송이 중단되면 같은 `rsync` 명령을 다시 실행하면 이어받습니다. 400Mbps가 실제 전송 구간에서 유지되면 약 30~90분, 불안정하면 1~2시간입니다.

## 6. 서버 파일 검증

```bash
cd /home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge
find models/Wan-AI/Wan2.1-I2V-14B-480P -maxdepth 1 -name '*.incomplete'
find models/Wan-AI/Wan2.1-I2V-14B-480P \
  -maxdepth 1 -name 'diffusion_pytorch_model-*.safetensors' \
  -printf '%f %s\n' | sort
find models/Wan-AI/Wan2.1-T2V-1.3B -maxdepth 1 -type f -printf '%f %s\n' | sort
```

아무 `.incomplete`도 없어야 하며, DiT shard 크기는 다음과 일치해야 합니다.

```text
00001 9847223624
00002 9797021016
00003 9797041744
00004 9692142864
00005 9692142864
00006 9692142864
00007 7061933536
```

CLIP 예상 크기는 `4772359047` bytes입니다. 파일명이 정상이어도 크기가 다르면 손상된 파일입니다.

## 7. 학습 전 확인

```bash
ps -ef | rg 'modelscope download|snapshot_download|train_abot_so100_vace' || true
nvidia-smi
```

필수 조건은 7개 shard 공식 크기 일치, CLIP/T5/VAE 존재, `.incomplete` 없음, 기존 다운로드 프로세스 없음, GPU 1번 사용 가능입니다.

## 8. VACE-only 48시간 학습

Wan 14B DiT와 VAE/CLIP/T5는 고정하고 VACE adapter만 학습합니다. `--max_train_steps 0`으로 step 제한을 없애고 `--max_train_seconds 172800`으로 실제 학습 시작 후 최대 48시간 실행합니다.

```bash
cd /home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge
conda activate inha_sy
export CUDA_VISIBLE_DEVICES=1

PYTHONPATH=src:. python scripts/train_abot_so100_vace.py \
  --abot_root /home/video_generation/inha_challenge/ChoiSeongYong/abot-physworld \
  --action_stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --dataset_base_path /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge/data/train \
  --dataset_metadata_path outputs/abot_so100_train/metadata.jsonl \
  --data_file_keys video --height 480 --width 640 --num_frames 17 \
  --model_id_with_origin_paths 'Wan-AI/Wan2.1-I2V-14B-480P:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-T2V-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-T2V-1.3B:Wan2.1_VAE.pth,Wan-AI/Wan2.1-I2V-14B-480P:models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth' \
  --learning_rate 5e-6 --num_epochs 1000 --max_train_steps 0 \
  --max_train_seconds 172800 --save_steps 100 \
  --output_path outputs/abot_so100_vace_1800 \
  --remove_prefix_in_ckpt pipe.vace. --trainable_models vace \
  --extra_inputs input_image --disable_text_condition true \
  --init_vace_from_dit true --init_vace_from_dit_vace_in_dim 96 \
  --chunk_num_frames 17 --min_stride 1 --max_stride 1 \
  --dataset_video_resize_mode stretch --use_gradient_checkpointing_offload \
  --dataset_num_workers 1
```

학습 시작 확인:

```bash
rg -n 'training clock started|step=|loss|Traceback|CUDA out of memory' \
  outputs/abot_so100_vace_1800/pipeline.log
```

## 주의사항

- `.incomplete` 파일을 정상 파일로 이름만 바꾸지 않습니다.
- 다운로드 중인 서버 프로세스가 있는 상태에서 전송하지 않습니다.
- shard 하나라도 공식 크기가 다르면 학습하지 않습니다.
- 모델과 전송에는 80GB 이상 여유 공간이 필요합니다.
