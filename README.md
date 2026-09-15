# 2026 인하 인공지능 챌린지 — Action-Conditioned World Model

초기 로봇 이미지와 16-step 6D action sequence를 입력받아 이후 16-frame 영상을 생성하는 World Model 프로젝트입니다. 데이터 구성, 모델 학습·추론, checkpoint 선택과 제출 영상 검증 코드를 포함합니다.

- 대회: [2026 인하 인공지능 챌린지](https://dacon.io/competitions/official/236736/overview/description)

## 대회 과제

| 항목 | 내용 |
| --- | --- |
| 입력 | 480×640 RGB 이미지 1장, `(16, 6)` action sequence |
| 출력 | 16 frames, 6 FPS, 640×480 MP4 |
| 목표 | 초기 관측과 행동을 반영한 미래 로봇 영상 생성 |
| 평가 | `0.3 × DINO + 0.3 × Video Feature + 0.4 × Action` (낮을수록 우수) |

학습과 검증에는 train 데이터만 사용하며, eval 데이터는 학습이 끝난 모델의 추론에만 사용합니다. 생성된 MP4는 형식 검사를 거친 뒤 공식 submission kit으로 CSV로 변환합니다.

## 구현 모델

| 모델 | 구현 내용 |
| --- | --- |
| **DynamiCrafter-plus** | action alignment와 학습 해상도별 checkpoint를 비교하고, owner-disjoint holdout과 action-sensitivity gate로 후보를 선택 |
| **Cosmos-Predict2.5 2B** | SO-100 dataset adapter, 6D action conditioning, zero text embedding과 Blackwell attention backend 적용 |
| **ABot-PhysWorld + Wan2.1-I2V-14B + VACE v2** | ABot/Wan DiT는 고정하고 VACE adapter와 6D latent action encoder를 학습 |

### 주요 방법: ABot/Wan VACE v2

6D action을 RGB 형태로 변환하지 않고 VACE latent context에 직접 주입합니다.

- train fold의 median과 IQR로 action을 정규화합니다.
- absolute action, 첫 action 기준 변위와 temporal delta를 결합해 18D feature를 만듭니다.
- synthetic tail을 추가한 뒤 latent action encoder와 Fourier XY encoding으로 VACE context를 생성합니다.
- ABot/Wan DiT는 고정하고 VACE adapter와 action encoder만 학습합니다.

## 실행 환경

- Python `>=3.10,<3.14`
- CUDA 환경의 PyTorch
- 대회 데이터 및 submission kit
- 모델별 upstream 저장소와 사전학습 가중치

PyTorch는 CUDA 버전에 맞게 먼저 설치합니다.

```bash
git clone https://github.com/ChoiSeongyong/World_Model-2026_Inha_Challenge.git
cd World_Model-2026_Inha_Challenge

python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -e ".[dev]"
```

ABot/Wan 경로의 추가 패키지:

```bash
python3 -m pip install -r requirements_abot_so100.txt
python3 -m pip install -r /path/to/abot-physworld/requirements.txt
```

Cosmos 경로는 Cosmos-Predict2.5 upstream의 `uv.lock` 환경을 사용합니다. 모델별 upstream revision과 가중치 배치는 [`THIRD_PARTY.md`](THIRD_PARTY.md)를 참고합니다.

대회 데이터, 사전학습 가중치와 checkpoint는 저장소에 포함되어 있지 않습니다.

## 경로 설정

```bash
export PROJECT_ROOT=/workspace/World_Model-2026_Inha_Challenge
export OPEN_ROOT=/workspace/data_challenge
export PYTHONPATH="$PROJECT_ROOT/src:$PROJECT_ROOT"
```

`OPEN_ROOT` 아래에 `data/train`, `data/eval`과 `submission_kit`을 배치합니다.

## 실행 방법

### 1. Manifest와 fold 생성

```bash
python3 scripts/build_manifest.py \
  --train-root "$OPEN_ROOT/data/train" \
  --output-dir artifacts/manifests

python3 scripts/build_folds.py \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --output artifacts/folds/folds.json
```

같은 owner/repository의 영상이 train과 validation에 동시에 포함되지 않도록 그룹 단위로 분리합니다.

### 2. 모델 학습 및 추론

```bash
python3 scripts/train_dynamicrafter_plus.py --help
python3 scripts/run_cosmos_predict25_so100.py --help
python3 scripts/train_abot_so100_vace_v2.py --help
python3 scripts/infer_abot_so100_vace_v2.py --help
```

모델별 전체 명령과 checkpoint 설정은 [`reports/`](reports/)에 정리되어 있습니다.

### 3. 제출 영상 검사

```bash
python3 scripts/audit_pre_submission.py \
  --video-root /path/to/predictions \
  --eval-root "$OPEN_ROOT/data/eval" \
  --provenance /path/to/inference_provenance.json \
  --output /path/to/pre_submission_audit.json
```

영상 개수, 파일명, 해상도, frame 수, FPS와 provenance를 검사합니다.

## 주요 파일

| 경로 | 역할 |
| --- | --- |
| `src/inha_worldmodel/data.py` | 이미지와 action sequence 로딩 |
| `src/inha_worldmodel/manifest.py`, `fold_selection.py` | manifest 생성과 train/validation 분리 |
| `src/inha_worldmodel/dynamicrafter_*` | DynamiCrafter 학습, 검증과 checkpoint 선택 |
| `integrations/abot_physworld/so100_action_condition_v2.py` | 6D action을 VACE latent context로 변환 |
| `integrations/cosmos_predict25/` | Cosmos용 SO-100 dataset 및 inference adapter |
| `scripts/train_*`, `scripts/infer_*` | 모델별 학습 및 추론 실행기 |
| `scripts/audit_pre_submission.py` | 제출 영상 형식과 provenance 검사 |
| `configs/`, `patches/` | 모델 설정과 upstream 수정분 |
| `reports/` | 모델별 실행 방법과 실험 기록 |

## 테스트

```bash
PYTHONPATH=src:. python3 -m pytest -q
python3 -m compileall -q src integrations scripts
```
