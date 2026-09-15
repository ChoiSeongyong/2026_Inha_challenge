# 2026 인하 인공지능 챌린지 — Action-Conditioned World Model

초기 로봇 이미지 1장과 16-step 6D action sequence를 입력으로 받아 이후 16-frame 로봇 영상을 생성하는 World Model 프로젝트입니다. [2026 인하 인공지능 챌린지](https://dacon.io/competitions/official/236736/overview/description)를 위해 데이터 점검, 모델 학습·추론, 후보 선택과 제출 전 검증 코드를 구현했습니다.

이 저장소에는 대회 데이터, 사전학습 가중치, 학습 checkpoint, 생성 영상과 제출 CSV가 포함되어 있지 않습니다. 공개 코드는 실험 경로와 재현 절차를 보존하며 리더보드 점수를 별도로 주장하지 않습니다.

## 대회 과제

- **입력**: 480×640 RGB 초기 이미지 1장, `(16, 6)` action sequence
- **출력**: 16 frames, 6 FPS, 640×480 MP4
- **목표**: 현재 관측과 행동을 함께 반영해 로봇의 미래 움직임과 환경 변화를 생성
- **평가**: `0.3 × DINO Component + 0.3 × Video Feature Component + 0.4 × Action Component`, 낮을수록 우수
- **제약**: 단일 RTX PRO 6000 96GB 기준 학습 4일 이내, 전체 eval 추론 1시간 이내

대회 규칙에 맞춰 train 데이터만 학습과 검증에 사용하고, eval은 고정된 모델의 최종 추론에만 사용하도록 경로를 분리했습니다. 공식 submission kit은 생성 영상이 확정된 뒤 CSV 변환에만 사용합니다.

## 구현한 모델 경로

| 모델 경로 | 구현 내용 | 저장소에 기록된 상태 |
| --- | --- | --- |
| **DynamiCrafter-plus** | 대회 video prior를 기반으로 action alignment, 학습 해상도, checkpoint 후보를 비교하고 owner-disjoint holdout과 action-sensitivity gate로 선택 | 학습·선택·final refit·216개 MP4 추론 경로 구현 |
| **Cosmos-Predict2.5 2B** | SO-100 데이터 adapter, 6D action conditioning, zero text embedding과 Blackwell용 attention backend를 추가해 14K checkpoint에서 32K까지 재학습 | 32K EMA로 216개 영상 생성 기록, 약 705.7초 wall time |
| **ABot-PhysWorld + Wan2.1-I2V-14B + VACE v2** | visual/robot prior는 고정하고 VACE adapter와 6D action encoder만 학습 | 14K 학습 상태 저장, 10K checkpoint 전체 eval 추론 약 3,007.8초 |

위 시간은 저장소 보고서에 기록된 단일 RTX PRO 6000 실행 wall time이며 리더보드 성능 수치가 아닙니다.

## 최종 고용량 경로: ABot/Wan VACE v2

초기 구현은 6D action을 RGB pseudo-trajectory로 바꿔 자연영상 VAE에 입력했습니다. v2는 이 과정에서 제어값이 손실될 수 있는 문제를 피하기 위해 action을 latent VACE context에 직접 주입합니다.

```text
action [16, 6]
  ├─ train-fold median/IQR 정규화
  ├─ absolute action       [16, 6]
  ├─ first action 기준 변위 [16, 6]
  └─ temporal delta       [16, 6]
             ↓
      action feature [17, 18]
             ↓
 latent action encoder + Fourier XY
             ↓
 direct VACE context [B, 96, T, H, W]
             ↓
 frozen ABot SFT Wan DiT + initial image
             ↓
 16-frame future video
```

- 16개 action 뒤에 Wan tokenizer용 synthetic tail 1개를 추가합니다.
- Wan2.1 14B DiT와 ABot robot-domain SFT DiT는 고정합니다.
- VACE adapter와 `18D → 96-channel` latent action encoder만 학습합니다.
- checkpoint에는 모델뿐 아니라 optimizer, scheduler와 RNG 상태를 저장해 재개를 지원합니다.
- 추론에서는 TeaCache와 persistent-VRAM 옵션으로 1시간 제한을 맞추도록 구성했습니다.

## 실행 환경

- Python `>=3.10,<3.14` (`pyproject.toml` 기준, 프로젝트 문서는 Python 3.11 권장)
- CUDA 사용이 가능한 PyTorch 환경
- 대회 데이터와 submission kit
- 사용하는 경로에 따라 ABot-PhysWorld, Wan2.1 또는 Cosmos-Predict2.5 upstream
- 사전학습 가중치와 학습용 GPU 저장공간

PyTorch는 GPU 드라이버와 CUDA에 맞는 공식 wheel을 먼저 설치해야 합니다.

```bash
git clone https://github.com/ChoiSeongyong/World_Model-2026_Inha_Challenge.git
cd World_Model-2026_Inha_Challenge

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

ABot 경로는 추가 의존성이 필요합니다.

```bash
python3 -m pip install -r requirements_abot_so100.txt
python3 -m pip install -r /path/to/abot-physworld/requirements.txt
```

Cosmos 경로는 Cosmos-Predict2.5 upstream의 `uv.lock` 환경을 사용합니다. 세 모델의 upstream revision과 patch, 가중치 배치는 [`THIRD_PARTY.md`](THIRD_PARTY.md)와 각 보고서를 확인하십시오.

## 권장 디렉터리

```text
workspace/
├── World_Model-2026_Inha_Challenge/
├── abot-physworld/
├── cosmos-predict2.5/
└── data_challenge/
    ├── data/train/
    ├── data/eval/
    ├── baseline/
    └── submission_kit/
```

```bash
export PROJECT_ROOT=/workspace/World_Model-2026_Inha_Challenge
export ABOT_ROOT=/workspace/abot-physworld
export COSMOS_ROOT=/workspace/cosmos-predict2.5
export OPEN_ROOT=/workspace/data_challenge
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
```

## 기본 실행 흐름

### 1. Manifest와 leakage-safe fold 생성

```bash
python3 scripts/build_manifest.py \
  --train-root "$OPEN_ROOT/data/train" \
  --output-dir artifacts/manifests

python3 scripts/build_folds.py \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --output artifacts/folds/folds.json
```

Fold는 같은 owner/repository의 영상이 train과 validation에 동시에 들어가지 않도록 그룹 단위로 분리합니다.

### 2. 모델별 준비·학습·추론

각 실행기는 `--help`로 필수 경로와 옵션을 확인할 수 있습니다.

```bash
python3 scripts/train_dynamicrafter_plus.py --help
python3 scripts/run_cosmos_predict25_so100.py --help
python3 scripts/train_abot_so100_vace_v2.py --help
python3 scripts/infer_abot_so100_vace_v2.py --help
```

전체 실행 명령과 고정 checkpoint·upstream revision은 다음 문서에 보존되어 있습니다.

- ABot/Wan: [`reports/MODEL_IMPLEMENTATION_SUMMARY.md`](reports/MODEL_IMPLEMENTATION_SUMMARY.md)
- Cosmos: [`reports/COSMOS_32K_FINAL_MODEL.md`](reports/COSMOS_32K_FINAL_MODEL.md)
- DynamiCrafter: [`reports/CANDIDATE_SELECTION.md`](reports/CANDIDATE_SELECTION.md), [`reports/FINAL_REFIT.md`](reports/FINAL_REFIT.md)
- GPU 실행 절차: [`reports/GPU_RUNBOOK.md`](reports/GPU_RUNBOOK.md)

### 3. 제출 전 감사

```bash
python3 scripts/audit_pre_submission.py \
  --video-root /path/to/predictions \
  --eval-root "$OPEN_ROOT/data/eval" \
  --provenance /path/to/inference_provenance.json \
  --output /path/to/pre_submission_audit.json
```

영상의 개수, 이름, 해상도, frame 수, FPS와 provenance를 검사한 뒤 공식 submission kit으로 CSV를 생성합니다. 생성된 CSV는 수정하지 않습니다.

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `src/inha_worldmodel/data.py` | 학습 데이터와 action sequence 로딩 |
| `src/inha_worldmodel/manifest.py` | episode manifest 생성·검증 |
| `src/inha_worldmodel/fold_selection.py` | repository 단위 train/validation 분리 |
| `src/inha_worldmodel/dynamicrafter_*` | DynamiCrafter 학습, checkpoint, 검증과 후보 선택 |
| `src/inha_worldmodel/model.py` | 경량 baseline world model 구성 |
| `src/inha_worldmodel/metrics.py` | holdout용 영상 품질 진단 지표 |
| `src/inha_worldmodel/pre_submission_audit.py` | 제출 영상 계약과 provenance 감사 |
| `integrations/abot_physworld/so100_action_condition_v2.py` | 6D action을 direct VACE latent context로 변환 |
| `integrations/cosmos_predict25/` | Cosmos용 SO-100 dataset과 inference adapter |
| `scripts/train_*` | 모델별 학습 실행기 |
| `scripts/infer_*` | 모델별 영상 생성 실행기 |
| `configs/` | DynamiCrafter와 경량 모델 설정 |
| `patches/` | 고정 upstream revision에 적용한 수정분 |
| `reports/` | 데이터 감사, 모델 선택, 학습·추론 기록 |
| `tests/` | CPU 단위·통합 테스트 |

## 테스트

외부 가중치 없이 실행 가능한 CPU 테스트:

```bash
PYTHONPATH=src:. python3 -m pytest -q
python3 -m compileall -q src integrations scripts
```

이 검사는 코드 계약과 데이터 처리 로직을 확인하며 실제 생성 품질이나 GPU 추론 시간을 검증하지는 않습니다.

## 재현성과 제한 사항

- train과 eval을 분리하고 통계와 fold는 train에서만 계산합니다.
- eval은 고정 checkpoint의 추론 입력으로만 사용합니다.
- submission kit은 최종 MP4를 CSV로 변환하는 단계에서만 사용합니다.
- checkpoint, 원본 데이터, 생성 영상, 제출 CSV와 credential은 Git에서 제외합니다.
- manifest와 실행 기록에 checkpoint·통계·source hash와 wall time을 남깁니다.
- 모델 성능 비교와 최종 선택의 상세 근거는 `reports/`에 있으며, 공개 저장소만으로 대회 점수를 재현할 수는 없습니다.

## 외부 프로젝트

- [ABot-PhysWorld](https://github.com/amap-cvlab/ABot-PhysWorld)
- [Wan2.1](https://github.com/Wan-Video/Wan2.1)
- [NVIDIA Cosmos-Predict2.5](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [DynamiCrafter](https://github.com/Doubiiu/DynamiCrafter)

외부 소스와 모델 가중치는 각 배포처의 라이선스를 따릅니다. 이 저장소의 `patches/`는 수정분만 포함하며 upstream 소스나 가중치를 재배포하지 않습니다. 자세한 고정 revision과 이용 조건은 [`THIRD_PARTY.md`](THIRD_PARTY.md)를 확인하십시오.
