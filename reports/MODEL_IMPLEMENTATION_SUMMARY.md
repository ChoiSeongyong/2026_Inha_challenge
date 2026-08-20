# 구현 모델 및 현재 학습 상태 정리

작성 기준: 2026-08-06

## 한눈에 보는 결론

현재 프로젝트에는 두 개의 주요 모델 경로가 있습니다.

1. **DynamiCrafter-plus**: 기존 대회 제공 video prior를 대회 데이터에 맞게 정제한 안정적인 제출 후보
2. **NVIDIA Cosmos-Predict2.5 2B Action-Conditioned**: 기존 모델을 단순 튜닝하는 대신 새로 도입한 성능 상한 후보. 14,000 step checkpoint에서 재개한 32,000 step 추가 학습이 완료됨

현재 DynamiCrafter-plus와 Cosmos 모두 실제 MP4 생성 경로가 구현되어 있습니다. Cosmos 32,000 step EMA checkpoint로 216개 평가 영상 추론도 완료되었습니다.

### 최신 Cosmos 32,000 step 결과

Cosmos는 14,000 step 학습 checkpoint에서 model·optimizer·scheduler·trainer
state를 복원한 뒤 32,000 step까지 추가 학습했습니다. 최종 checkpoint는 다음과
같습니다.

```text
outputs/cosmos_predict25_so100_continue_14k_to_32k/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/iter_000032000/
```

추론에는 EMA BF16 checkpoint를 사용했습니다.

```text
outputs/cosmos_predict25_so100_continue_14k_to_32k/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/iter_000032000/model_ema_bf16.pt
```

Blackwell GPU에서 `minimal_a2a` backend가 `No available kernel`을 발생시켜
`transformer_engine` backend를 사용했습니다. SO-100에 Reason1 text embedding이
없으므로 dataset adapter는 Cosmos projection 계약에 맞는 zero embedding
`(1, 100352) = (1, 98 × 1024)`를 제공합니다.

실제 1-step 재개 검증과 32,000 step 학습을 모두 통과했습니다.

```text
Iteration 14001 | Loss: 0.0332
Done with training.
```

최종 32,000 step 모델의 216개 전체 추론도 성공했습니다.

```text
outputs/cosmos_predict25_inference_32k_35step_retry2/predictions/
prediction_count: 216
returncode: 0
total_wall_seconds: 705.7124791339738
```

전체 추론 시간은 약 11분 46초로 대회 1시간 제한을 충족합니다. 자세한
구현·재현 설명은 [COSMOS_32K_FINAL_MODEL.md](COSMOS_32K_FINAL_MODEL.md)에
정리되어 있습니다.

---

## 1. DynamiCrafter-plus 경로

### 모델의 역할

DynamiCrafter는 첫 관측 이미지와 action 조건을 받아 이후 로봇 영상을 생성하는 video diffusion 모델입니다. 프로젝트에서는 대회 제공 checkpoint를 기반으로 action conditioning과 학습 해상도, action alignment를 비교하고, 가장 안정적인 설정을 다시 장기 학습했습니다.

### 실제로 튜닝한 부분

DynamiCrafter를 처음부터 새로 학습한 것이 아닙니다. 대회가 제공한 두 종류의 checkpoint를 결합해 시작했습니다.

```text
backbone.ckpt
  └─ VAE, image/text conditioning module, 기본 video prior

baseline_diffusion.ckpt
  └─ 대회 action-conditioned UNet/EMA, 1,500 step 상태
```

그 위에서 다음 학습만 추가했습니다.

1. VAE, text encoder, image encoder는 frozen 상태로 유지
2. action-conditioned diffusion UNet과 EMA를 학습
3. train fold에서만 계산한 action 통계로 첫 action projection을 재parameterize
4. 6차원 action을 UNet에 주입
5. 6 FPS, 16프레임 window를 episode-uniform 방식으로 샘플링
6. owner/repository group이 겹치지 않는 holdout으로 검증

따라서 이 경로의 본질은 **기존 시각적 video prior는 보존하고, SO-100 action에 대한 조건부 생성 능력을 대회 train fold에 맞게 정제하는 것**입니다.

### DynamiCrafter 입력부터 출력까지

```text
초기 RGB frame I[0]       shape: [3, 480, 640]
action sequence A         shape: [16, 6]
                           │
                           ├─ action normalization (train fold 통계)
                           ├─ image encoder / image projection
                           └─ action-conditioned temporal UNet
                                      │
                                      ▼
                         latent video, 16 frames
                                      │
                                      ▼
                         VAE decode → RGB frames
                                      │
                                      ▼
                         16-frame MP4, 6 FPS
```

모델 내부에서는 입력 영상을 latent 공간으로 압축하고, diffusion timestep의 noisy latent를 temporal attention과 image cross-attention이 있는 3D UNet에 넣습니다. action은 6D projection을 거쳐 temporal UNet의 conditioning으로 들어갑니다. frame 0의 appearance는 초기 이미지 conditioning이 담당하고, 이후 frame의 변화는 action conditioning과 temporal prior가 함께 결정합니다.

### action alignment를 비교한 이유

데이터에는 명령값 `action[t]`와 실제 측정 상태 `observation.state[t]`가 모두 있습니다. 전체 train-only 분석에서 다음 관계가 관찰되었습니다.

```text
action[t]          → state[t+1]
```

즉 로봇의 servo lag 때문에 command가 같은 시점의 화면보다 다음 상태와 더 가까운 경향이 있습니다. 그래서 다음 두 모델을 별도로 학습했습니다.

```text
same_step:
    video[t]   ← action[t]

previous_command:
    video[t+1] ← action[t]
```

기존 대회 baseline과의 계약을 보존하기 위해 `same_step`을 기본 후보로 두었고, `previous_command`는 반드시 별도 checkpoint로 학습해 holdout에서 비교했습니다. 이미 학습된 same-step checkpoint에 action 배열만 사후 shift하는 방식은 사용하지 않았습니다.

### gate와 최종 refit의 차이

```text
후보별 1,000 step gate
    ├─ train
    ├─ 원래 action으로 holdout inference
    └─ 다른 clip의 action으로 holdout inference
             │
             ▼
    action sensitivity + metric + runtime 비교
             │
             ▼
    선택 후보(previous320_raw6)
             │
             ▼
    선택 후보만 17,000 step final refit
             │
             ▼
    216개 전체 inference
```

gate는 최종 모델 자체가 아니라 어떤 configuration을 장기 학습할지 고르는 실험입니다. cross-clip 검증에서 action을 바꿨을 때 출력이 전혀 변하지 않는 후보를 제거하고, 원래 action의 생성 품질과 inference 시간도 함께 확인했습니다.

### 구현된 주요 구성

- 초기 이미지에서 로봇·배경의 시각적 특징 추출
- 6차원 SO-100 action conditioning
- 16프레임 continuation 생성
- DDIM 기반 inference
- MP4 저장 및 제출용 provenance 기록
- train-only holdout을 이용한 후보 비교
- 원본 action과 다른 clip의 action을 바꾸는 cross-clip 검증으로 action 반응성 확인

### 비교한 후보 설정

| 후보 | 해상도 | action 표현 | alignment | 의미 |
|---|---:|---|---|---|
| `base320_raw6` | 320 | raw 6D | same-step | 기본 baseline |
| `previous320_raw6` | 320 | raw 6D | previous-command | 실제 servo lag를 반영한 설정 |
| `same384_raw6` | 384 | raw 6D | same-step | 더 높은 학습 해상도 |
| `previous384_raw6` | 384 | raw 6D | previous-command | lag 반영 + 고해상도 |

추가 gate 계획에서는 384/480 해상도 및 kinematic 변형도 검토했지만, 제한된 시간과 1시간 inference 조건을 고려해 모든 후보를 최종 장기 학습으로 가져가지는 않았습니다.

### 후보 선택 결과

gate 결과에서 `previous320_raw6`가 선택되었습니다.

- action alignment: `previous_command`
- action representation: `raw6`
- 학습 해상도: 320 계열
- gate 학습: 1,000 step
- 최종 refit: 17,000 step 경로
- 최종 checkpoint:

```text
outputs/dynamicrafter_plus/final_refit_previous320_raw6_s20260725_u17000/checkpoints/last.ckpt
```

선택 근거는 단순 loss 하나가 아니라 다음을 함께 사용했습니다.

- holdout 원본 조건 성능
- cross-clip action을 넣었을 때 foreground metric이 악화되는지 여부
- 후보별 native metric
- 최악 repository 구간 성능
- 216개 전체 inference runtime projection

### DynamiCrafter 결과 영상

최종 inference 결과는 216개 MP4로 생성되어 있습니다.

```text
artifacts/predictions/final/sample_000000.mp4
...
artifacts/predictions/final/sample_000215.mp4
```

관련 provenance:

```text
artifacts/predictions/final/inference_provenance.json
```

이 파일에는 checkpoint, action 통계, 입력 조건 fingerprint, MP4 hash가 기록되어 있습니다.

### DynamiCrafter의 장단점

장점:

- 이미 대회 입력·출력 계약에 맞는 구현과 MP4 생성 파이프라인이 있음
- 216개 inference와 제출 artifact까지 검증된 안전한 fallback
- 현재 결과를 바로 확인할 수 있음

한계:

- 기본 video prior의 표현력과 action 이해력에 성능 상한이 있음
- 후보 gate가 최종 hidden score를 보장하지 않음
- 장기 학습을 늘리는 것만으로 Cosmos 수준의 새로운 prior를 얻지는 못함

---

## 2. Cosmos-Predict2.5 2B Action-Conditioned 경로

### 모델의 역할

Cosmos-Predict2.5는 첫 이미지와 action sequence를 입력으로 받아 미래 영상을 생성하는 NVIDIA의 action-conditioned world model입니다. DynamiCrafter checkpoint를 다시 튜닝하는 대신, 공개 pretrained world model의 prior를 SO-100 대회 데이터에 맞춰 post-training하는 경로입니다.

### DynamiCrafter와 다른 점

DynamiCrafter는 latent video diffusion UNet을 대회 checkpoint에서 이어 학습한 경로이고, Cosmos는 로봇 action-conditioned world model의 공식 2B pretrained checkpoint에서 시작합니다.

```text
DynamiCrafter:
    대회 제공 video prior + 대회 action UNet
    → action alignment / resolution / update 수 튜닝

Cosmos:
    공개 Cosmos-Predict2.5 2B robot/action-cond prior
    → SO-100 6D action dataset adapter + post-training
```

Cosmos는 rectified-flow DiT 계열과 WAN VAE/tokenizer를 사용하며, action-conditioned network가 image/video latent와 action sequence를 함께 받아 미래 latent를 예측합니다. 따라서 단순히 DynamiCrafter의 config를 바꾼 것이 아니라, dataset loader·temporal padding·action projection·upstream Hydra overlay를 새로 연결한 별도 모델 경로입니다.

### Cosmos 입력부터 출력까지

```text
초기 RGB frame I[0]       480×640
action A                  [16, 6]
                           │
                           ├─ train-fold median/IQR normalization
                           ├─ 6 FPS resampling
                           ├─ native letterbox → 256×320
                           └─ WAN temporal boundary padding
                                      │
                                      ▼
                 Cosmos internal video: 17 frames
                 effective action condition: first 15 actions
                                      │
                                      ▼
                         action-conditioned DiT/flow
                                      │
                                      ▼
                         VAE decode + tail trim
                                      │
                                      ▼
                  resize/restore → 640×480, 16 frames, 6 FPS MP4
```

대회 계약은 16프레임이지만 WAN temporal tokenizer는 16을 직접 표현하지 못합니다. 그래서 학습 경계에서 마지막 frame을 한 번 반복해 17프레임으로 만들고, 모델 결과에서 synthetic tail frame만 제거합니다. 이 padding은 실제 데이터와 제출 결과에 남지 않습니다.

### Cosmos action projection 변경 범위

공식 checkpoint와 대회 action 차원이 다르므로 action input projection 전체를 무시하지 않고, shape가 달라지는 두 weight만 재초기화합니다.

```text
action_embedder_B_D.fc1.weight
action_embedder_B_3D.fc1.weight
```

그 외 checkpoint tensor의 shape가 다르면 즉시 실패하도록 검증합니다. 이 제한은 pretrained weight를 최대한 보존하면서 6D SO-100 action만 연결하기 위한 것입니다.

### 학습 중 loss를 해석하는 방법

Cosmos는 매 step의 loss가 video sample과 noise level에 따라 크게 흔들릴 수 있습니다. 따라서 단일 step loss가 낮다고 모델이 완성됐다고 판단하지 않고 다음을 확인합니다.

- iteration speed가 안정적인지
- GPU가 실제로 사용되는지
- 2,000 step 단위 checkpoint가 완전하게 저장되는지
- 주기적인 sample callback이 끝나는지
- max iteration까지 `Done with training`이 기록되는지
- 완료 checkpoint를 사용한 holdout/inference가 통과하는지

현재 로그에 표시되는 `Iteration 13400 : iter_speed ...` 같은 줄은 누적 학습 iteration이며, `Hit counter: 100/100`은 callback의 주기 카운터이지 전체 학습이 100 step에서 끝났다는 뜻이 아닙니다.

### 대회 입력에 맞춘 adapter

구현 위치:

```text
integrations/cosmos_predict25/
scripts/run_cosmos_predict25_so100.py
scripts/infer_cosmos_predict25_so100.py
```

주요 처리:

- 대회 데이터를 Cosmos dataset 계약으로 변환
- 6 FPS 기준 16프레임 window 구성
- train fold에서만 action robust statistics 계산
- 대회 6D action을 공식 Cosmos action-conditioned 네트워크에 연결
- temporal contract를 위해 학습 경계에서 필요한 tail padding 처리
- native 학습 해상도 256×320 사용
- inference 결과를 640×480, 6 FPS, 16프레임 MP4로 변환
- 모델·checkpoint·조건·hash를 provenance에 기록

### 실행 환경

Cosmos 공식 upstream checkout과 호환되는 별도 `.venv`를 사용합니다.

```text
/home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5/.venv/
```

현재 `inha_sy` 환경에서 직접 upstream을 실행하지 않고, adapter가 공식 upstream의 호환 Python/torchrun을 선택하도록 구현되어 있습니다. 이 방식은 `cosmos_oss`와 `flash-attn` 호환성 문제를 피하기 위한 것입니다.

### 현재 장기 학습 상태

현재 명령의 핵심 설정은 다음과 같습니다.

```text
trainer.max_iter=14000
trainer.grad_accum_iter=8
model.config.net.use_crossattn_projection=false
GPU=1
```

출력 경로:

```text
outputs/cosmos_predict25_so100_final_14k_run2/
```

최근 확인 기준:

- 학습 프로세스 정상 실행 중
- 목표 14,000 step
- 마지막 확인 시 약 12,400~12,550 step
- `iter_000012000` checkpoint 생성 완료
- GPU 사용률 및 iteration 진행 정상
- 알려진 fatal error 없음

checkpoint는 다음 아래에 저장됩니다.

```text
outputs/cosmos_predict25_so100_final_14k_run2/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/
```

### Cosmos 학습 중 샘플 영상

샘플 생성 callback은 동작하지만, 기존 upstream callback은 WandB가 비활성화되면 로컬 MP4를 저장하지 않는 구조였습니다. 이를 수정해 이후 실행에서는 다음 경로에 composite MP4를 저장하도록 했습니다.

```text
cosmos-predict2.5/cosmos_predict2/_src/predict2/callbacks/every_n_draw_sample.py
```

저장 경로:

```text
outputs/cosmos_predict25_so100_final_14k_run2/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/EveryNDrawSample/
```

단, 이미 시작된 현재 프로세스는 시작 시점에 Python 파일을 읽었으므로 이 수정은 현재 실행에 소급 적용되지 않습니다. 현재 학습을 중단할 필요는 없으며, 완료 후 inference에서 생성되는 MP4에는 영향이 없습니다.

샘플 파일은 최종 제출 영상과 다릅니다. 학습 중 `EveryNDrawSample`은 학습 상태를 육안 점검하기 위한 composite preview이고, 최종 제출용 216개 영상은 학습 완료 checkpoint를 고정한 뒤 `scripts/infer_cosmos_predict25_so100.py`가 별도로 생성합니다.

### Cosmos inference의 제한

Cosmos 기본 teacher inference는 denoising step 수가 크기 때문에 216개 전체 inference가 대회 1시간 제한을 넘을 수 있습니다. 따라서 구현된 inference wrapper는 다음 순서로 동작합니다.

1. 소수 sample benchmark 실행
2. 실제 step 수·입력 수·MP4 encoding을 포함한 wall time 측정
3. 보수적인 전체 시간 projection 계산
4. 1시간 제한을 넘을 것으로 판단되면 전체 실행 거부
5. 통과한 경우에만 216개 전체 MP4 생성

즉, Cosmos checkpoint가 학습되었다고 바로 제출하는 것이 아니라, 반드시 inference budget gate를 통과해야 합니다.

---

## 3. 두 모델의 현재 위치 비교

| 항목 | DynamiCrafter-plus | Cosmos-Predict2.5 |
|---|---|---|
| 목적 | 안정적인 대회 제출 후보 | 새 pretrained world model 기반 성능 상한 후보 |
| 학습 상태 | 완료 | 32,000 step 추가 학습 완료 |
| 현재 checkpoint | 있음 | `iter_000032000` |
| 전체 MP4 | 216개 생성 완료 | 216개 생성 완료 |
| 입력 계약 | 대회 계약에 맞게 구현 | 대회 6D action adapter 구현 |
| inference | 이미 검증 | 216개·약 11분 46초로 검증 |
| 위험도 | 낮음 | 중간~높음 |
| 기대 역할 | 제출 가능한 fallback | validation과 runtime이 통과하면 최종 후보 |

## 4. 앞으로의 판단 순서

Cosmos 학습 종료 후 다음 순서로 판단합니다.

1. 최종 checkpoint와 학습 로그 확인
2. train-only holdout validation 실행
3. Cosmos inference benchmark 실행
4. 1시간 전체 inference 가능 여부 확인
5. 통과하면 216개 전체 inference 및 MP4 검증
6. DynamiCrafter 결과와 파일 수·해상도·fps·프레임 수 비교
7. 규칙에 맞는 최종 MP4 디렉터리를 확정
8. MP4가 고정된 뒤 제출 CSV 변환

현재 DynamiCrafter 결과는 삭제하지 않고 안전한 fallback으로 보존합니다. Cosmos가 validation 또는 1시간 inference gate를 통과하지 못하면 DynamiCrafter 결과를 사용할 수 있습니다.

## 5. 핵심 파일 위치

```text
# DynamiCrafter
scripts/infer_dynamicrafter_plus.py
scripts/run_dynamicrafter_gate_plan.py
scripts/run_dynamicrafter_final_refit.py
outputs/dynamicrafter_plus/
artifacts/predictions/final/

# Cosmos
scripts/run_cosmos_predict25_so100.py
scripts/infer_cosmos_predict25_so100.py
integrations/cosmos_predict25/
outputs/cosmos_predict25_so100_continue_14k_to_32k/

# 모델 조사 및 규칙
reports/MODEL_STRATEGY.md
reports/MODEL_SEARCH_COSMOS_SO100.md
reports/RULES.md
reports/PRE_SUBMISSION_AUDIT.md
```
