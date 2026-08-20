# 2026 INHA AI Challenge — Action-Conditioned Robot World Models

초기 로봇 이미지 1장과 16-step 6D action sequence를 입력으로 받아
16-frame 미래 영상을 생성한 2026 인하 인공지능 챌린지 프로젝트입니다.

이 저장소는 대회 기간에 실험한 세 모델 경로와 데이터 감사, 학습, 추론,
제출 전 검증 코드를 함께 보존합니다. 최종 고용량 경로는
**Wan2.1-I2V-14B + ABot-PhysWorld SFT DiT + VACE v2 action adapter**입니다.

> 이 저장소에는 대회 데이터, 사전학습 가중치, 학습 checkpoint, 생성 MP4,
> 제출 CSV가 포함되지 않습니다. 각 파일은 원 배포처와 라이선스를 확인한 뒤
> 별도로 준비해야 합니다.

## 문제와 제약

- 입력: RGB 초기 이미지 480×640 1장, action (16, 6)
- 출력: 미래 영상 16 frames, 6 FPS, 640×480 MP4
- 평가: 0.3 × DINO + 0.3 × Video Feature + 0.4 × Action
- 목표: 낮을수록 좋음
- 데이터: 대회 train만 학습에 사용하고 eval은 학습·검증에 사용하지 않음
- 계산 제한: RTX PRO 6000 96GB 1장 기준 학습 4일, 전체 eval 추론 1시간
- 제출: 최종 MP4 확정 후 원본 submission kit으로 CSV를 한 번 생성

규칙 스냅샷은 [reports/RULES.md](reports/RULES.md), 데이터 감사는
[reports/DATA_AUDIT.md](reports/DATA_AUDIT.md)에 있습니다.

## 최종 모델: ABot/Wan VACE v2

첫 구현은 6D action을 RGB pseudo-trajectory로 렌더링해 VAE로 인코딩했습니다.
v2는 한 관절을 잃고 자연영상 VAE에 제어값 해석을 맡기는 이 경로를 버리고,
모든 관절을 latent VACE context에 직접 주입합니다.

~~~text
initial RGB image ───────────────────────────────┐
                                                │
action [16, 6]                                  │
  ├─ train-fold median/IQR normalization        │
  ├─ absolute command              [16, 6]      │
  ├─ displacement from first action [16, 6]     │
  └─ temporal delta                [16, 6]      │
                 │                              │
                 ▼                              │
        action features [17, 18]                │
                 │                              │
       latent action encoder + Fourier XY       │
                 │                              │
      direct VACE context [B, 96, T, H, W]      │
                 │                              │
                 └──── frozen ABot SFT Wan DiT ◄┘
                                  │
                                  ▼
                    16-frame 640×480 MP4
~~~

- Wan2.1-I2V-14B visual prior와 ABot robot-domain SFT DiT는 고정
- VACE adapter와 18D→96-channel latent action encoder만 학습
- 16개 실제 action 뒤에 Wan tokenizer용 synthetic tail 1개만 추가
- checkpoint에 VACE와 action encoder를 함께 저장
- optimizer, scheduler, RNG 상태를 별도 저장해 정확한 재개 지원
- TeaCache와 persistent-VRAM 모드로 1시간 추론 제한 충족

핵심 구현:

- [integrations/abot_physworld/so100_action_condition_v2.py](integrations/abot_physworld/so100_action_condition_v2.py)
- [scripts/train_abot_so100_vace_v2.py](scripts/train_abot_so100_vace_v2.py)
- [scripts/infer_abot_so100_vace_v2.py](scripts/infer_abot_so100_vace_v2.py)
- [scripts/run_abot_so100_vace_v2_48h.sh](scripts/run_abot_so100_vace_v2_48h.sh)

## 실험 경로와 완료 결과

| 경로 | 역할 | 완료 상태 | 전체 eval 추론 |
|---|---|---|---:|
| DynamiCrafter-plus | 대회 action-conditioned baseline 정제 | gate와 17K final refit 구현 | 216 MP4 경로 검증 |
| Cosmos-Predict2.5 2B | 공개 action-conditioned world model post-training | 14K→32K 완료 | 705.7초 |
| ABot/Wan VACE v2 | 14B robot SFT prior + lossless 6D adapter | 정확한 상태 14K 저장 | step 10K, 3,007.8초 |

ABot v2 시간 인증 실행은 step 10,000, 12 denoising steps, CFG 1.0,
TeaCache threshold 0.2, persistent VRAM으로 216/216개를 약 50분 8초에
완료했습니다. 학습은 14,000 step까지 이어졌지만 표의 추론 시간은 실제
전체 eval 검증을 마친 10,000-step checkpoint 기준입니다.

Cosmos 결과는
[reports/COSMOS_32K_FINAL_MODEL.md](reports/COSMOS_32K_FINAL_MODEL.md),
전체 발전 과정은
[reports/MODEL_IMPLEMENTATION_SUMMARY.md](reports/MODEL_IMPLEMENTATION_SUMMARY.md)에
있습니다. 이 저장소는 리더보드 점수를 재현한다고 주장하지 않으며, 위 수치는
로컬 완료 상태와 wall-clock 기록입니다.

## 저장소 구조

~~~text
configs/                         DynamiCrafter 및 경량 모델 설정
docs/                            다운로드와 서버 전송 문서
integrations/
  abot_physworld/                SO-100 action→VACE adapter
  cosmos_predict25/              Cosmos dataset/config/inference adapter
patches/                         고정 upstream commit용 patch
reports/                         규칙, 데이터, 모델 선택, 실행 결과
scripts/                         준비, 학습, 추론, 감사 실행기
src/inha_worldmodel/             공통 데이터·검증·모델 라이브러리
tests/                           CPU 단위·통합 테스트
artifacts/                       재생성 manifest/fold/statistics; Git 제외
models/                          로컬 가중치; Git 제외
outputs/                         checkpoint/log/video; Git 제외
~~~

## 권장 작업공간과 환경변수

~~~text
workspace/
├── Inha_challenge/
├── abot-physworld/
├── cosmos-predict2.5/           # Cosmos 재현 시에만 필요
└── data_challenge/
    ├── data/train/
    ├── data/eval/
    ├── baseline/
    └── submission_kit/
~~~

~~~bash
export PROJECT_ROOT=/workspace/Inha_challenge
export ABOT_ROOT=/workspace/abot-physworld
export COSMOS_ROOT=/workspace/cosmos-predict2.5
export OPEN_ROOT=/workspace/data_challenge
export PYTHON_BIN=/path/to/python

cd "$PROJECT_ROOT"
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
~~~

## 설치

Python 3.11과 CUDA 호환 PyTorch 환경을 권장합니다. PyTorch는 GPU 드라이버와
CUDA에 맞는 공식 wheel을 먼저 설치합니다.

~~~bash
cd "$PROJECT_ROOT"

"$PYTHON_BIN" -m pip install -e ".[dev]"
"$PYTHON_BIN" -m pip install -r requirements_abot_so100.txt
"$PYTHON_BIN" -m pip install -r "$ABOT_ROOT/requirements.txt"
~~~

## 외부 upstream 고정과 patch

ABot-PhysWorld:

~~~bash
git clone https://github.com/amap-cvlab/ABot-PhysWorld.git "$ABOT_ROOT"
git -C "$ABOT_ROOT" checkout 7d47080ea122346e6b7c1cb37c2a8d43730f624c
git -C "$ABOT_ROOT" apply "$PROJECT_ROOT/patches/abot-physworld-7d47080.patch"
~~~

ABot patch는 robot SFT DiT 선행 로드, 17-frame tail padding, direct VACE
context, bundled state strict load, persistent VRAM, prompt cache, TeaCache,
첫 관측 frame 보존을 포함합니다.

Cosmos-Predict2.5:

~~~bash
git clone https://github.com/nvidia-cosmos/cosmos-predict2.5.git "$COSMOS_ROOT"
git -C "$COSMOS_ROOT" checkout a2c298b0a3df3778b973fe65e9e58877b292d8a7
git -C "$COSMOS_ROOT" apply "$PROJECT_ROOT/patches/cosmos-predict2.5-a2c298b.patch"
~~~

Cosmos 환경은 upstream uv.lock을 사용합니다. 프로젝트 runner가 SO-100
experiment overlay를 생성합니다. 자세한 절차는
[integrations/cosmos_predict25/README.md](integrations/cosmos_predict25/README.md)를
따릅니다.

## 공개 가중치 준비

| 모델 | 원 배포처 | 역할 |
|---|---|---|
| Wan-AI/Wan2.1-I2V-14B-480P | ModelScope/Hugging Face | 14B DiT, CLIP |
| Wan-AI/Wan2.1-T2V-1.3B | ModelScope/Hugging Face | UMT5, Wan VAE |
| amap_cvlab/Abot-PhysWorld | ModelScope | robot SFT DiT |
| nvidia/Cosmos-Predict2.5-2B | Hugging Face | 선택적 Cosmos 경로 |

~~~bash
modelscope download Wan-AI/Wan2.1-I2V-14B-480P \
  --local-dir "$PROJECT_ROOT/models/Wan-AI/Wan2.1-I2V-14B-480P"

modelscope download Wan-AI/Wan2.1-T2V-1.3B \
  --local-dir "$PROJECT_ROOT/models/Wan-AI/Wan2.1-T2V-1.3B"

modelscope download amap_cvlab/Abot-PhysWorld \
  --local-dir "$PROJECT_ROOT/models/Abot-PhysWorld"
~~~

필수 배치는 다음과 같습니다.

~~~text
models/
├── Abot-PhysWorld/abotpw_i2v_480p.safetensors
└── Wan-AI/
    ├── Wan2.1-I2V-14B-480P/
    │   ├── diffusion_pytorch_model-00001-of-00007.safetensors
    │   ├── ...
    │   ├── diffusion_pytorch_model-00007-of-00007.safetensors
    │   └── models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
    └── Wan2.1-T2V-1.3B/
        ├── models_t5_umt5-xxl-enc-bf16.pth
        └── Wan2.1_VAE.pth
~~~

다운로드 복구와 크기 검증은
[docs/WAN14B_WINDOWS_DOWNLOAD_TRANSFER.md](docs/WAN14B_WINDOWS_DOWNLOAD_TRANSFER.md)를
참고하십시오. 가중치는 이 저장소에서 재배포하지 않습니다.

## 데이터 준비

manifest와 leakage-safe fold:

~~~bash
cd "$PROJECT_ROOT"

PYTHONPATH=src:. "$PYTHON_BIN" scripts/build_manifest.py \
  --train-root "$OPEN_ROOT/data/train" \
  --output-dir artifacts/manifests

PYTHONPATH=src:. "$PYTHON_BIN" scripts/build_folds.py \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --output artifacts/folds/folds.json
~~~

실제 감사 결과는 원본 128 repositories, 11,132 episodes,
정제 후 126 repositories, 11,002 episodes입니다. 기본
seeded_group_00_seed_17 fold는 train 8,802 / validation 2,200이며
owner/repository overlap은 0입니다.

train-fold action statistics:

~~~bash
PYTHONPATH=src:. "$PYTHON_BIN" scripts/prepare_so100_action_stats.py \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --train-root "$OPEN_ROOT/data/train" \
  --fold-artifact artifacts/folds/folds.json \
  --fold-id seeded_group_00_seed_17 \
  --output artifacts/cosmos_predict25/action_robust_stats_fold17.json
~~~

ABot metadata:

~~~bash
PYTHONPATH=src:. "$PYTHON_BIN" scripts/prepare_abot_so100.py \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --data-root "$OPEN_ROOT/data/train" \
  --stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --output-root outputs/abot_so100_train \
  --metadata outputs/abot_so100_train/metadata.jsonl
~~~

## ABot VACE v2 학습

step 제한 없이 4일보다 30분 짧은 wall-clock budget 예시입니다.

~~~bash
cd "$PROJECT_ROOT"

export CUDA_VISIBLE_DEVICES=1
export OUTPUT_PATH="$PROJECT_ROOT/outputs/abot_so100_vace_v2_final"
export DIT_CHECKPOINT="$PROJECT_ROOT/models/Abot-PhysWorld/abotpw_i2v_480p.safetensors"

ABOT_ROOT="$ABOT_ROOT" \
OPEN_ROOT="$OPEN_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_PATH="$OUTPUT_PATH" \
DIT_CHECKPOINT="$DIT_CHECKPOINT" \
bash scripts/run_abot_so100_vace_v2_48h.sh \
  --max_train_seconds 343800
~~~

주요 설정은 480×640, 17 model frames, learning rate 5e-6,
weight decay 0.01, VACE+latent action encoder 학습, 1,000-step checkpoint,
최근 4개 및 5,000-step milestone 보존입니다. 저장 전 최소 100GiB 여유
공간을 검사합니다.

정확한 재개:

~~~bash
export RESUME_STEP=14000

ABOT_ROOT="$ABOT_ROOT" \
OPEN_ROOT="$OPEN_ROOT" \
PYTHON_BIN="$PYTHON_BIN" \
OUTPUT_PATH="$OUTPUT_PATH" \
DIT_CHECKPOINT="$DIT_CHECKPOINT" \
bash scripts/run_abot_so100_vace_v2_48h.sh \
  --resume_from_step "$RESUME_STEP" \
  --max_train_seconds 21600
~~~

latest_training_state.pt와 같은 step의 safetensors가 함께 있어야
optimizer, scheduler, RNG까지 복원됩니다.

## ABot VACE v2 추론

eval JSONL:

~~~bash
PYTHONPATH=src:. "$PYTHON_BIN" scripts/prepare_abot_so100_eval.py \
  --eval-root "$OPEN_ROOT/data/eval" \
  --output outputs/abot_so100_eval.jsonl
~~~

실제로 1시간 제한을 통과한 전체 추론 구성:

~~~bash
export CUDA_VISIBLE_DEVICES=0
export INFER_OUTPUT="$PROJECT_ROOT/outputs/abot_so100_vace_v2_inference"

PYTHONPATH=src:. "$PYTHON_BIN" scripts/infer_abot_so100_vace_v2.py \
  --abot-root "$ABOT_ROOT" \
  --jsonl outputs/abot_so100_eval.jsonl \
  --action-stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --checkpoint "$OUTPUT_PATH/step-10000.safetensors" \
  --dit-checkpoint "$DIT_CHECKPOINT" \
  --output-root "$INFER_OUTPUT" \
  --height 480 \
  --width 640 \
  --steps 12 \
  --cfg-scale 1.0 \
  --seed 0 \
  --fps 6 \
  --persistent-vram \
  --tea-cache-l1-thresh 0.2
~~~

성공하면 inference_manifest.json, predictions/inference_provenance.json,
sample_000000.mp4부터 sample_000215.mp4까지 생성됩니다.

## 제출 전 감사와 CSV

submission kit과 독립된 감사:

~~~bash
PYTHONPATH=src:. "$PYTHON_BIN" scripts/audit_pre_submission.py \
  --video-root "$INFER_OUTPUT/predictions" \
  --eval-root "$OPEN_ROOT/data/eval" \
  --provenance "$INFER_OUTPUT/predictions/inference_provenance.json" \
  --output "$INFER_OUTPUT/pre_submission_audit.json"
~~~

최종 MP4가 확정된 뒤에만 원본 submission kit을 실행합니다.

~~~bash
export SUBMISSION_KIT_ROOT="$OPEN_ROOT/submission_kit"
cd "$SUBMISSION_KIT_ROOT"

CUDA_VISIBLE_DEVICES=0 "$PYTHON_BIN" make_submission_csv.py \
  --prediction-root "$INFER_OUTPUT/predictions" \
  --challenge-root "$OPEN_ROOT/data/eval" \
  --output-csv "$INFER_OUTPUT/submission_features.csv" \
  --action-stats-path "$OPEN_ROOT/data/train/so100_action_statistics.json" \
  --action-extractor-ckpt "$SUBMISSION_KIT_ROOT/checkpoints/action_extractor.ckpt"
~~~

생성 CSV를 수정하지 않으며 submission kit을 학습, validation, 모델 선택,
영상 후처리에 사용하지 않습니다.

## Cosmos 32K 경로

~~~bash
cd "$PROJECT_ROOT"
export CUDA_VISIBLE_DEVICES=1

"$COSMOS_ROOT/.venv/bin/python" scripts/run_cosmos_predict25_so100.py \
  --upstream-root "$COSMOS_ROOT" \
  --open-root "$OPEN_ROOT" \
  --output-root outputs/cosmos_predict25_so100_32k \
  --max-iter 32000 \
  --grad-accum-iter 8 \
  --override model.config.net.atten_backend=transformer_engine \
  --execute
~~~

14K 정확 재개, EMA 변환, 추론은
[reports/COSMOS_32K_FINAL_MODEL.md](reports/COSMOS_32K_FINAL_MODEL.md)와
[integrations/cosmos_predict25/README.md](integrations/cosmos_predict25/README.md)를
참고하십시오.

## DynamiCrafter-plus 경로

DynamiCrafter는 owner-disjoint fold, same-step/previous-command alignment,
320/384 gate, cross-clip action sensitivity, immutable final-refit plan,
checkpoint/config/source hash 계약을 구현합니다.

- [reports/CANDIDATE_SELECTION.md](reports/CANDIDATE_SELECTION.md)
- [reports/FINAL_REFIT.md](reports/FINAL_REFIT.md)
- [reports/DYNAMICRAFTER_VALIDATION.md](reports/DYNAMICRAFTER_VALIDATION.md)
- [reports/GPU_RUNBOOK.md](reports/GPU_RUNBOOK.md)

## 테스트

가중치 없이 실행 가능한 CPU 테스트:

~~~bash
cd "$PROJECT_ROOT"
PYTHONPATH=src:. "$PYTHON_BIN" -m pytest -q
~~~

형식과 import 검사:

~~~bash
git diff --check
"$PYTHON_BIN" -m compileall -q src integrations scripts
~~~

## 재현성과 보안 경계

- train과 eval 경로를 분리
- 통계와 fold는 train만 사용
- eval은 고정 checkpoint 추론 입력으로만 사용
- submission kit은 MP4 확정 후 CSV 변환에만 사용
- checkpoint, 데이터, 영상, CSV, credential은 Git에 저장하지 않음
- manifest에 checkpoint, 통계, source hash와 wall-clock 기록
- 절대 경로가 있는 운영 스크립트는 대회 서버 기록용이며, 이 README의
  환경변수 기반 명령을 canonical 재현 경로로 사용

## 외부 프로젝트와 라이선스

- [ABot-PhysWorld](https://github.com/amap-cvlab/ABot-PhysWorld)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)

Wan2.1과 Cosmos 소스는 각 배포처의 라이선스를 따르며 Cosmos 모델 가중치는
NVIDIA Open Model License를 따릅니다. DynamiCrafter와 ABot checkpoint도
원 배포처의 사용 조건을 별도로 확인해야 합니다.

patches 디렉터리는 수정분만 기록하며 원본 저장소나 가중치를 포함하지
않습니다. 공개 배포 전에는 팀 소유 코드의 최종 라이선스를 별도로 결정해야
합니다.
