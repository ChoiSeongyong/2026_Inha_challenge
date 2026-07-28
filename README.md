# 2026 INHA AI Challenge

현재 로봇 이미지 1장과 16-step 6D action으로 16-frame 미래 영상을
생성하는 프로젝트입니다. 목표는 규정을 지키면서 Private Score를
최소화하는 것입니다.

## 현재 상태

- 원본 train: 128 repositories, 11,132 episodes, 1,025,666 frames
- 품질 정제 후: 126 repositories, 11,002 episodes
- 주 모델선택 split: `official_baseline_seed0_validation`
  - 제공 checkpoint가 학습하지 않은 train episode만 validation 548개
  - continuation train 10,454 / episode overlap 0
- 보조 일반화 fold: `seeded_group_00_seed_17`
  - train 8,802 / validation 2,200, owner/repository overlap 0
- 측정 상태는 `action[t]`에 대해 `state[t+1]` 반응이 가장 강함
- DynamiCrafter 첫 주력은 공식 checkpoint와 호환되는 `same_step`
  conditioning을 유지해 action 15까지 사용하고, `previous_command`는
  엄격한 holdout에서만 별도 ablation
- eval 입력: 216개, RGB 480×640, action `(16,6)`
- 150개 CPU 테스트와 실제 MP4/Parquet 1-step
  학습→checkpoint→16-frame 6fps 640×480 MP4 추론을 통과
- DynamiCrafter checkpoint는 main UNet 전체/EMA 전체/optimizer/step/config·
  fold·통계 hash를 검사하며, 부분 EMA나 weight-only 재시작을 거부
- holdout 선택 후 정제 train 11,002 episodes 전체 refit 경로 준비

주력 후보 순서는 다음과 같습니다.

1. 대회 제공 action-conditioned DynamiCrafter를 정제 fold에서 추가 학습
2. NVIDIA Cosmos-Predict2.5 2B action-conditioned post-training + 4-step DMD2
3. 고해상도 articulated layer warper
4. flow/residual 모델은 안전한 배관·fallback

## 디렉터리

- `src/inha_worldmodel/`: 데이터, fold, 모델, loss, 학습·추론
- `integrations/cosmos_predict25/`: Cosmos 2.5 독립 어댑터
- `configs/`: flow, articulated, DynamiCrafter 설정
- `scripts/`: manifest/fold/stats/학습/추론/MP4 감사
- `tests/`: CPU 단위·통합 테스트
- `reports/`: 규칙, 데이터, 검증, 모델 전략, 실행 문서
- `outputs/`: checkpoint와 로그
- `artifacts/`: 생성 영상, fold, 통계, provenance
- `data`: `Downloads/open/data` 읽기 링크
- `official_baseline`: 대회 제공 baseline 읽기 링크
- `official_submission_kit`: 대회 제공 submission kit 읽기 링크

## 절대 실행 경계

```text
train 데이터만 사용한 학습/검증
        ↓
eval image/action을 입력으로 고정 checkpoint 독립 추론
        ↓
최종 MP4 216개 확정 및 일반 MP4 형식 감사
        ↓
원본 submission_kit을 수정 없이 CSV 변환에만 1회 실행
        ↓
생성 CSV를 수정하지 않고 제출
```

`official_submission_kit`의 코드, 모델, checkpoint 또는 결과를 학습 loss,
feature extractor, validation metric, 후보 생성·선택, reranking, 후처리,
영상 수정에 사용하지 않습니다.

## 로컬 검증

```bash
cd /Users/choeseong-yong/Inha_challenge

PYTHONPATH=src:. /opt/anaconda3/bin/python -m pytest -q

PYTHONPATH=src /opt/anaconda3/bin/python -m inha_worldmodel.train \
  --config configs/flow_smoke.yaml

PYTHONPATH=src /opt/anaconda3/bin/python -m inha_worldmodel.infer \
  --checkpoint outputs/flow_smoke/best.pt \
  --eval-root data/eval \
  --output-dir artifacts/smoke_videos \
  --device cpu \
  --limit 1
```

중요 artifact는 원본 데이터에서 재생성할 수 있습니다.

```bash
PYTHONPATH=src /opt/anaconda3/bin/python scripts/build_manifest.py
PYTHONPATH=src /opt/anaconda3/bin/python scripts/build_folds.py
PYTHONPATH=src /opt/anaconda3/bin/python \
  scripts/build_checkpoint_pristine_split.py
PYTHONPATH=src /opt/anaconda3/bin/python scripts/prepare_dynamicrafter_stats.py \
  --validation-protocol official_checkpoint_pristine \
  --fold-artifact artifacts/folds/official_baseline_seed0_pristine.json \
  --fold-id official_baseline_seed0_validation \
  --output artifacts/stats/dynamicrafter_action_stats_checkpoint_pristine.json
PYTHONPATH=src /opt/anaconda3/bin/python scripts/prepare_dynamicrafter_stats.py \
  --training-scope all_clean \
  --output artifacts/stats/dynamicrafter_action_stats_all_clean.json
```

## DynamiCrafter-plus GPU 경로

대회 baseline requirements를 별도 CUDA 환경에 설치하고, 공식 notebook에
기재된 공개 backbone을 `open/baseline/checkpoints/backbone.ckpt`에
다운로드합니다. 실제 checksum과 license는
`reports/PRETRAINED_MODELS.md`에 기록해야 합니다.

```bash
export INHA_PROJECT_ROOT=/workspace/Inha_challenge
export INHA_OPEN_ROOT=/workspace/open
export INHA_BASELINE_ROOT="$INHA_OPEN_ROOT/baseline"
export PYTHONPATH="$INHA_PROJECT_ROOT/src:$PYTHONPATH"

# RTX PRO 6000 Blackwell 환경에서는 먼저 cu128 wheel을 명시적으로 설치
python -m pip install torch==2.7.1 torchvision==0.22.1 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$INHA_BASELINE_ROOT/requirements.txt"
python -m pip install -e "$INHA_PROJECT_ROOT"

mkdir -p "$INHA_BASELINE_ROOT/checkpoints"
curl --fail --location \
  https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt \
  --output "$INHA_BASELINE_ROOT/checkpoints/backbone.ckpt"
sha256sum "$INHA_BASELINE_ROOT/checkpoints/backbone.ckpt"

python "$INHA_PROJECT_ROOT/scripts/preflight_dynamicrafter_gpu.py" \
  --project-root "$INHA_PROJECT_ROOT" \
  --open-root "$INHA_OPEN_ROOT" \
  --baseline-root "$INHA_BASELINE_ROOT"

torchrun --standalone --nproc_per_node=1 \
  "$INHA_PROJECT_ROOT/scripts/train_dynamicrafter_plus.py" \
  --baseline-root "$INHA_BASELINE_ROOT" \
  --project-root "$INHA_PROJECT_ROOT" \
  --open-root "$INHA_OPEN_ROOT" \
  --base \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_plus.yaml" \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_checkpoint_pristine.yaml" \
  --train
```

GPU에서는 수동 후보 명령 대신 감사 가능한 계획을 생성·실행합니다.

```bash
python scripts/plan_dynamicrafter_gates.py \
  --project-root "$INHA_PROJECT_ROOT" \
  --open-root "$INHA_OPEN_ROOT" \
  --baseline-root "$INHA_BASELINE_ROOT" \
  --max-steps 1000 \
  --plan-root outputs/dynamicrafter_gate_1000 \
  --output outputs/dynamicrafter_gate_1000/plan.json

python scripts/run_dynamicrafter_gate_plan.py \
  --plan outputs/dynamicrafter_gate_1000/plan.json --execute
```

실행기는 fresh GPU preflight, checkpoint/config/source hash, 정확한 global step,
main/EMA 완전성·finite 여부, 고정 holdout 보고서, action sensitivity 및
보수적 216개/1시간 projection을 모두 통과시킨 뒤
`candidate_selection.json`을 생성합니다.

중단 후에는 제공된 1,500-step weight 파일을 직접 resume하지 않습니다.
동일 plan의 전체 checkpoint·config 계약을 실행기가 재검증한 뒤에만
재개합니다.

```bash
python scripts/run_dynamicrafter_gate_plan.py \
  --plan outputs/dynamicrafter_gate_1000/plan.json \
  --execute --resume
```

해상도와 표본 균형은 YAML overlay로만 비교합니다. 384p는 4:3 영상을
padding 없이 쓰는 우선 고해상도 후보이고, 480p는 4일/1시간 gate를
통과할 때만 유지합니다.

```bash
# 예: 384×512
--base configs/dynamicrafter_plus.yaml \
       configs/dynamicrafter_checkpoint_pristine.yaml \
       configs/dynamicrafter_plus_384.yaml

# 별도 ablation: owner 빈도 제곱근 보정
--base configs/dynamicrafter_plus.yaml \
       configs/dynamicrafter_checkpoint_pristine.yaml \
       configs/dynamicrafter_owner_tempered.yaml

# 별도 ablation: absolute + window delta + velocity 18D
--base configs/dynamicrafter_plus.yaml \
       configs/dynamicrafter_checkpoint_pristine.yaml \
       configs/dynamicrafter_kinematic18.yaml
```

18D variant는 제공 checkpoint의 첫 action MLP main/EMA weight만
`6→18`로 늘리고 새 12개 열을 0으로 초기화합니다. 따라서 학습 시작 전
출력은 6D baseline과 정확히 같고, 나머지 1,106개 main UNet tensor를
그대로 보존합니다.

각 checkpoint는 gate plan에 기록된 정확한 base→pristine→candidate→runtime
config 순서로 동일한 고정 holdout에서 original/cross-clip action을 각각
생성합니다. 수동 validation 명령 대신 위 gate 실행기를 사용해야 checkpoint
계약과 비교 조건이 보존됩니다.

alignment·해상도·sampler·update 수를 고정한 뒤에는 자동 생성된 immutable
plan으로 전체 정제 데이터 refit을 한 번 실행합니다. 수동으로 overlay나
checkpoint를 조합하지 않습니다.

```bash
PYTHONPATH=src:. python scripts/plan_dynamicrafter_final_refit.py \
  --gate-plan outputs/dynamicrafter_gate_1000/plan.json \
  --selection outputs/dynamicrafter_gate_1000/candidate_selection.json \
  --plan-root outputs/dynamicrafter_final_refit

PYTHONPATH=src:. python scripts/run_dynamicrafter_final_refit.py \
  --plan outputs/dynamicrafter_final_refit/plan.json

# preview를 확인한 뒤 GPU에서만 실행
PYTHONPATH=src:. python scripts/run_dynamicrafter_final_refit.py \
  --plan outputs/dynamicrafter_final_refit/plan.json --execute
```

고정 train holdout에서 sampling 설정과 checkpoint를 선택하고 전체 refit을
완료한 뒤에만 eval을 한 번 생성합니다. 학습 때 사용한 overlay를 같은
순서로 모두 넘겨야 embedded contract가 일치합니다.

```bash
PYTHONPATH="$INHA_PROJECT_ROOT/src" python \
  "$INHA_PROJECT_ROOT/scripts/infer_dynamicrafter_plus.py" \
  --config \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_plus.yaml" \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_checkpoint_pristine.yaml" \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_plus_384.yaml" \
    "$INHA_PROJECT_ROOT/configs/dynamicrafter_plus_refit_all.yaml" \
    /path/to/final_runtime_overlay.yaml \
  --checkpoint /path/to/selected.ckpt \
  --final-refit-plan /path/to/final_refit/plan.json \
  --baseline-root "$INHA_BASELINE_ROOT" \
  --project-root "$INHA_PROJECT_ROOT" \
  --open-root "$INHA_OPEN_ROOT" \
  --eval-root "$INHA_OPEN_ROOT/data/eval" \
  --output-dir "$INHA_PROJECT_ROOT/artifacts/predictions/dynamicrafter_plus"
```

Cosmos 2.5 준비·학습·DMD2 명령과 1시간/4일 gate는
`integrations/cosmos_predict25/README.md`를 따릅니다.

## 출력 감사

최종 영상 형식만 submission kit와 무관하게 검사합니다.

```bash
PYTHONPATH=src python scripts/audit_videos.py \
  --video-root artifacts/predictions/final \
  --eval-root data/eval \
  --output artifacts/predictions/final/video_audit.json
```

216개 모두 16 frames, 6fps, 640×480이고 checksum이 고정된 뒤에만 최종
CSV 변환 단계로 넘어갑니다.

## 현재 하드웨어 제약

현재 로컬 머신은 Apple M3 MacBook Air 16GB라 데이터 감사와 CPU 스모크만
가능합니다. 수상권 학습에는 대회 재현 기준인 단일 RTX PRO 6000 96GB급
GPU가 필요합니다. GPU 작업 우선순위와 중단 기준은
`reports/GPU_RUNBOOK.md`, 규정은 `reports/RULES.md`를 확인합니다.
