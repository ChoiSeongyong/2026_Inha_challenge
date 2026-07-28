# Strict final all-clean refit

이 단계는 고정 train holdout에서 후보 선택이 끝난 뒤, 선택된 구조만 전체 정제
train 데이터로 처음부터 다시 학습합니다. 입력은 다음 두 JSON뿐입니다.

- `plan_dynamicrafter_gates.py`가 만든 뒤 실행·감사된 gate plan
- `select_dynamicrafter_candidate.py`가 만든 candidate-selection JSON

eval 경로, submission kit, 점수, feature, metric override는 받지 않습니다.

## Planner가 다시 확인하는 것

`plan_dynamicrafter_final_refit.py`는 selection의 요약 값을 신뢰하지 않습니다.
각 후보의 original/cross-clip train-holdout 보고서를 현재 파일에서 다시 읽고
SHA-256을 확인한 뒤 동일한 selector를 다시 실행합니다. 재계산된 selection
전체가 입력 JSON과 동일해야 합니다.

그 다음 다음 계약을 확인합니다.

- selection 후보 집합과 gate plan 후보 집합이 정확히 동일
- 선택 후보가 두 hard gate를 통과한 eligible rank 1
- 보고서 경로, checkpoint, ordered config와 gate plan이 일치
- 선택 checkpoint의 현재 SHA-256과 보고서 provenance가 일치
- manifest와 대회 제공 checkpoint-pristine split 아티팩트의 현재 hash가 일치
- split ID가 `official_baseline_seed0_validation`이며 train/validation/all-clean
  episode 수가 각각 `10,454 / 548 / 11,002`
- checkpoint-pristine 원본 train metadata까지 다시 hash 검증
- 선택 학습 scope가 `fold_train`
- gate runtime overlay가 config의 마지막 항목
- 구조 overlay는 gate plan과 정확히 같은 순서

최종 config는 다음 순서로만 만듭니다.

```text
dynamicrafter_plus.yaml
dynamicrafter_checkpoint_pristine.yaml
선택 후보의 나머지 structural overlay들
dynamicrafter_plus_refit_all.yaml
새 final runtime_overlay.yaml
```

Gate runtime overlay와 선택된 fold checkpoint는 최종 명령에서 제거됩니다.
Fold checkpoint는 감사 증거로만 기록되며 초기화에는 사용되지 않습니다.
초기화는 SHA-256
`c66a22652e37001aa6ee5e21c874b0ad67acad707b01a4b9ace8cf584a2517c5`
인 대회 제공 `baseline_diffusion.ckpt`와 공개 backbone에서 다시 시작합니다.

## Update budget

명시적 값이 없으면 다음 식으로 전체-data update 수를 정합니다.

```text
ceil(gate fold_train max_steps × all_clean episodes / fold_train episodes)
```

현재 checkpoint-pristine split 기준 episode 수는 `10,454 → 11,002`입니다.
이 방식은
episode당 update exposure를 보존합니다.

주의: 100-step gate가 단순 CUDA smoke였다면 기본 결과 106 steps도
smoke일 뿐 최종 성능 학습량이 아닙니다. 긴 학습에서 이미 선택한 최종 update
수를 사용하거나 `--final-max-steps`에 **전체-data 최종 update 수**를
명시해야 합니다. 이 값은 더 이상 holdout으로 조정하지 않습니다.

## 계획 생성

```bash
PYTHONPATH=src:. python scripts/plan_dynamicrafter_final_refit.py \
  --gate-plan outputs/dynamicrafter_gate_plan/plan.json \
  --selection outputs/dynamicrafter_gate_plan/candidate_selection.json \
  --plan-root outputs/dynamicrafter_final_refit
```

명시적 최종 budget이 이미 고정된 경우:

```bash
PYTHONPATH=src:. python scripts/plan_dynamicrafter_final_refit.py \
  --gate-plan outputs/dynamicrafter_gate_plan/plan.json \
  --selection outputs/dynamicrafter_gate_plan/candidate_selection.json \
  --plan-root outputs/dynamicrafter_final_refit_explicit \
  --final-max-steps 31573 \
  --checkpoint-every 1000
```

위 `31,573`은 fold_train에서 고정한 30,000 updates를 현재 비율로
all-clean에 환산한 예시입니다.

Planner는 GPU를 실행하지 않습니다. 다음을 생성합니다.

- `plan.json`
- 새 `runtime_overlay.yaml`
- 필수 GPU preflight 명령
- 정확한 final train 명령
- 예상 `.../checkpoints/last.ckpt` 경로
- 선택 보고서에서 고정한 production inference policy
  - DDIM steps, eta, guidance scale/rescale, timestep spacing
  - AMP dtype와 GPU gate에서 실제 통과한 batch size
  - 선택에 사용한 seed

Inference policy는 선택 후보의 `candidate_variant.ddim`과 gate plan을 다시
교차한 값입니다. batch size가 해당 후보의 GPU validation batch size와
다르거나 DDIM steps가 gate plan과 다르면 planner가 거부합니다. 이후 production
inference는 임의 CLI 기본값 대신 이 고정 policy를 사용해야 합니다.

## 실행

먼저 명령을 출력만 합니다.

```bash
PYTHONPATH=src:. python scripts/run_dynamicrafter_final_refit.py \
  --plan outputs/dynamicrafter_final_refit/plan.json
```

검토 후 실제 GPU 머신에서만 실행합니다.

```bash
PYTHONPATH=src:. python scripts/run_dynamicrafter_final_refit.py \
  --plan outputs/dynamicrafter_final_refit/plan.json \
  --execute
```

Executor는 plan과 모든 원본 증거를 다시 검증합니다. 그 후 CUDA/VRAM,
dataset, manifest, folds, all-clean action stats, backbone, 제공 checkpoint를
검사하는 preflight를 항상 새로 실행합니다. Preflight가 실패하면 train
subprocess는 시작되지 않습니다.

기존 workdir나 `last.ckpt`가 있으면 from-scratch 계약을 지킬 수 없으므로
실행을 거부합니다. 학습 명령에는 `--resume-checkpoint`, auto-resume,
validation 또는 test 옵션이 없습니다. 성공 시 `execution_state.json`,
preflight/train 로그와 최종 checkpoint SHA-256을 기록합니다.
