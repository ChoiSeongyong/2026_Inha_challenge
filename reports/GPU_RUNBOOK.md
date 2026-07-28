# 단일 96GB GPU 실행 순서

확인 시각: 2026-07-25 KST

이 문서는 RTX PRO 6000 96GB급 Linux/CUDA 머신을 받은 즉시 실행할
순서입니다. 긴 학습을 먼저 시작하지 않고, 각 후보가 memory·품질·시간
gate를 통과할 때만 다음 예산을 배정합니다.

## 0. 실행 전 고정

- Python/CUDA/driver/GPU 이름과 `nvidia-smi`를 로그에 저장
- train 원본, manifest, fold artifact SHA-256 확인
- 주 검증은 제공 checkpoint가 보지 않은 episode로 만든
  `official_baseline_seed0_validation`
- `seeded_group_00_seed_17` owner-disjoint 결과는 보조 일반화 지표
- 사전학습 weight URL/revision/license/SHA-256 기록
- submission kit는 학습 환경의 `PYTHONPATH`에 넣지 않음
- eval은 최종 독립 추론 전까지 어떤 검증 script에도 전달하지 않음

필수 CPU gate:

```bash
cd /workspace/Inha_challenge
PYTHONPATH=src:. python -m pytest -q
```

Pristine split과 두 action-stat artifact는 현재 workspace에 고정되어 있으며,
아래 CUDA preflight가 원본 manifest/train metadata와 hash·count를 다시
검증합니다. 기본값이 owner-disjoint인 stats 명령을 무인자로 실행하지
않습니다.

CUDA 환경 설치 뒤 fail-fast gate:

```bash
python scripts/preflight_dynamicrafter_gpu.py \
  --project-root /workspace/Inha_challenge \
  --open-root /workspace/open \
  --baseline-root /workspace/open/baseline
```

이 검사는 CUDA/VRAM, 필수 package, free disk, backbone/provided checkpoint,
manifest/fold/action-stats hash와 checkpoint main/EMA count를 기록합니다.

주 checkpoint-pristine split:

```text
continuation train: 10,454 episodes
validation:             548 episodes
all clean:           11,002 episodes
episode overlap:          0
```

## 1. DynamiCrafter-plus — 첫 주력

대회가 제공한 action model은 1,500 step checkpoint이므로 가장 빠르게
강한 제출을 만들 수 있는 경로입니다.

### 1시간 gate

1. 대회 baseline requirements 설치
   - 이어서 `pip install -e /workspace/Inha_challenge`로 `pyarrow`와
     프로젝트 package 설치
2. 공개 DynamiCrafter backbone 다운로드 및 SHA-256 기록
3. 제공 `baseline_diffusion.ckpt`의 expected action UNet/EMA tensor 100%
   존재·shape match
4. checkpoint-pristine DataModule에서 실제 batch:
   - video `[B,3,16,320,512]`, `[-1,1]`
   - act `[B,16,6]`
   - 기본 `same_step`에서 `act[:,t]`가 raw `action[:,t]`이고 action 15 보존
   - `previous_command`는 10→6fps에서도 source frame `s_t-1`을 정확히 사용
5. BF16/FP16 100 update:
   - OOM 없음
   - loss/gradient finite
   - peak VRAM 기록
6. validation clip 16개를 15-step DDIM으로 생성해:
   - frame count 16
   - cross-clip action이 원래 action보다 paired proxy metric을 악화
   - source frame 보존

gate 실패 시 batch를 4→2→1, gradient accumulation을 반대로 늘립니다.
100-step에서도 cross-clip action 차이가 없으면 긴 학습을 시작하지 않습니다.

### 본 학습

기본 설정은 batch 4, accumulation 2, learning rate `5e-5`, 최대 30k
update, checkpoint 1k update 간격입니다. checkpoint는 main/EMA뿐 아니라
optimizer, scheduler, loop, global step과 data/config contract를 포함합니다.
재개에는 `--resume-checkpoint`만 사용하며 제공된 1,500-step weight 파일을
resume 용도로 쓰지 않습니다. 실제 100-step 시간으로 4일 상한을 넘지
않도록 max step을 먼저 다시 계산합니다.

Checkpoint 선택은 submission kit나 Public score가 아니라 고정
train-holdout sampling으로 합니다.

- native reconstruction L1/SSIM/temporal error
- moving foreground L1
- repository worst quartile
- paired cross-clip action sensitivity
- runtime

기본 Lightning validation은 조건을 무작위로 제거하고 정렬된 앞부분이
소수 repository에 치우치므로 비활성화했습니다. 선택은
`scripts/validate_dynamicrafter_plus.py`의 고정 repository-round-robin
DDIM 결과만 사용합니다. metric은 padding을 제거한 원본 해상도에서
계산합니다.

최소 비교:

```text
provided 1,500-step checkpoint
3k / 5k / 8k / 12k / 20k / 30k
```

DDIM 설정은 holdout에서만 아래를 비교합니다.

```text
steps: 15, 20, 30, 50
eta: 0 고정
timestep spacing: uniform_trailing 우선
guidance scale: 1.0 우선
```

모델 비교 순서:

1. `same_step`과 `previous_command`를 제공 checkpoint의 고정 holdout에서
   짧게 검사하고, 각 alignment는 그 alignment로 별도 학습
2. 320×512 → 384×512(no side padding) → 480×640 순서로 100-step,
   holdout, 전체 216 runtime gate
3. episode-uniform과 owner-tempered(지수 0.5) 비교
4. raw6와 absolute+delta+velocity 18D 비교. 18D는 첫 action MLP의
   main/EMA weight만 6→18 zero-column 확장해 step 0 출력을 동일하게 유지
5. winning 설정만 3k/5k/8k/12k... checkpoint 검증

원래 action과 cross-clip action의 paired report는 checkpoint·DDIM·fold·
선택 sample fingerprint가 모두 같아야 하며, foreground L1의
`control-original` 평균이 양수여야 합니다.

216개 전체 생성은 1시간 상한이므로 I/O와 MP4 encode를 포함한 dry run에서
최대 약 12초/sample을 목표로 합니다. self-check timer는 Python import,
backbone/checkpoint load, hash, sample scan, diffusion, restore, MP4 encode,
provenance write를 모두 포함합니다.

### 최종 전체-data refit

alignment·해상도·sampling·update budget을 checkpoint-pristine 검증에서
완전히 고정한 뒤
`configs/dynamicrafter_plus_refit_all.yaml`을 마지막 overlay로 추가해
manifest-approved 11,002 episodes 전체로 처음부터 한 번 refit합니다.

- continuation 학습: 10,454 episodes / action stats 968,076 frames
- 전체 refit: 11,002 episodes / action stats 1,018,554 frames
- 같은 episode exposure를 맞추려면 선택 update 수에 `11002/10454`를 곱함
- 전체 refit 설정으로 validation script를 실행하면 코드가 거부해야 함
- 최종 inference에도 학습 때의 base/resolution/sampler/refit overlay를
  같은 순서로 전달

## 2. Cosmos-Predict2.5 2B — 성능 상한 후보

`integrations/cosmos_predict25/README.md`의 strict commit과 overlay를
사용합니다. NVIDIA gated-model 약관을 사용자가 직접 수락해야 합니다.

### 필수 gate

- BF16만 사용
- single GPU, context parallel 1
- fold 17 robust stats signature 확인
- 16-frame dataset → WAN 경계 17-frame tail padding → output tail trim
- 6D/15-action 변경으로 shape mismatch는 두 action embedder `fc1.weight`
  외에는 0
- batch 1, accumulation 8로 100 update OOM 없음
- peak VRAM 90GiB 이하
- action shuffle sensitivity 양수
- 4-step DMD2의 전체 eval 추정 시간이 1시간 이하

Teacher post-training이 고정 holdout에서 DynamiCrafter를 명확히 이기지
못하면 DMD2 예산을 쓰지 않습니다.

## 3. Articulated high-resolution fallback

`configs/layered_highres.yaml`은 480×640 원본을 직접 warp하되, mask와
transform 분석은 192×256에서 수행합니다. 배경·질감을 생성형 모델보다
보존하는 안전한 후보입니다.

```bash
PYTHONPATH=src torchrun --standalone --nproc_per_node=1 \
  -m inha_worldmodel.train --config configs/layered_highres.yaml
```

1시간 gate에서 batch 1 peak VRAM, 100 update 속도, moving-foreground
holdout error를 확인합니다. 낮은 해상도 flow 모델보다 foreground와
worst-owner가 개선되지 않으면 중단합니다.

## 4. 모델 선택과 결합

- Pixel-space 영상 평균은 흐림 때문에 금지
- eval sample별 후보 생성/선택/rerank 금지
- 한 개의 fixed checkpoint와 fixed sampler를 OOF에서 선택
- fold 선택 뒤 manifest-approved 전체 train으로 한 번 refit
- 결합이 필요하면 train OOF로 학습한 flow/mask-space fusion만 사용
- Public leaderboard는 파이프라인 확인용이며 hyperparameter 최적화 기준이
  아님

권장 제출 순서:

1. 제공 baseline 그대로 — 계정/팀 구성에 필요한 첫 유효 제출
2. DynamiCrafter-plus fixed checkpoint
3. Cosmos-DMD2 또는 articulated가 grouped OOF에서 명확히 이긴 경우만 교체

하루 3회 제출 제한을 지키고, 각 제출에 checkpoint/config/video audit
SHA-256을 기록합니다.

## 5. 최종 inference gate

```text
sample count       216
each frame count   16
FPS                6.0
resolution         640×480
frame 0            original condition injected before encode
generator runtime  < 1 hour including MP4 encode
submission kit     아직 사용 안 함
```

```bash
PYTHONPATH=src python scripts/audit_videos.py \
  --video-root artifacts/predictions/final \
  --eval-root data/eval \
  --output artifacts/predictions/final/video_audit.json
```

비디오 형식 확인 후 kit과 완전히 분리된 최종 교차 감사를 실행합니다.

```bash
PYTHONPATH=src:. python scripts/audit_pre_submission.py \
  --video-root artifacts/predictions/final \
  --eval-root data/eval \
  --provenance artifacts/predictions/final/inference_provenance.json \
  --output artifacts/predictions/final/pre_submission_audit.json
```

이 감사가 `passed=true`이고 `authorized_next_step`이
`single_final_mp4_to_csv_conversion_only`일 때 모든 MP4와 provenance의
SHA-256이 고정됩니다. 상세 계약은 `reports/PRE_SUBMISSION_AUDIT.md`를
따릅니다. 그 다음에만 원본 submission kit를 수정 없이 CSV 변환 용도로 한
번 실행하고, 생성 CSV도 수정하지 않습니다.

## 즉시 필요한 사용자 정보

- 2~5인 팀 구성 상태와 첫 제출 여부
- 사용할 단일 96GB NVIDIA GPU의 접속 방식/작업 경로
- NVIDIA Cosmos gated license 수락 가능 여부
- 학습 머신의 데이터 위치
