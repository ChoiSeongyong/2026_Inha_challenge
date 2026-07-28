# 수상권 모델 전략

확인 시각: 2026-07-25 KST

## 결론

현재 승률이 가장 높은 1순위는 대회가 제공한 action-conditioned
DynamiCrafter checkpoint를 정제 fold에서 계속 학습하는 경로입니다.
이미 공개 video prior와 1,500-step 대회 action UNet/EMA가 있어 제한된
4일 예산에서 가장 빨리 강한 후보를 만들 수 있습니다.

후보 우선순위는 다음과 같습니다.

1. DynamiCrafter-plus: strict fold, same-step 기본/action-alignment ablation,
   320→384→480 해상도 gate, exact resume, 고정 DDIM holdout 선택
2. Cosmos-Predict2.5 2B action-conditioned + 4-step DMD2: license·VRAM·
   1시간 gate를 모두 통과할 때만
3. measured-state dynamics + articulated full-resolution warper: 질감·배경
   보존 fallback
4. flow/residual 모델: 데이터·검증·출력 배관과 안전 제출 fallback

3순위 물리 구조의 핵심 식은 다음과 같습니다.

```text
s[0] = InitialStateEncoder(image_feature(I[0]))
s[t] = s[t-1]
     + gain * (action[t-1] - s[t-1])
     + ServoResidualGRU(...)

video[t] = ArticulatedRenderer(
    initial_image=I[0],
    initial_state=s[0],
    target_state=s[t],
    state_delta=s[t]-s[0],
)
```

전체 train 1,025,666 frame을 검사한 결과 모든 관절에서
`action[t] ↔ observation.state[t+1]`이 최적 정렬입니다. 따라서 출력
frame 0은 입력 이미지 그대로이고, frame `t>0`은 기본적으로
`action[t-1]`이 움직임을 구동해야 합니다. Raw action을 곧바로 frame
pose로 쓰는 모델보다 이 구조가 실제 servo lag를 반영합니다.

다만 제공 DynamiCrafter는 time축 전체를 비인과적으로 보는 official
same-step action contract로 학습됐습니다. 이를 무조건 한 칸 shift하면
action 15를 완전히 버리므로, 첫 continuation은 `same_step`을 유지합니다.
`previous_command`는 10→6fps source timeline까지 정확히 구현한 뒤
held-out train에서 별도 학습·비교합니다.

## 규칙상 절대 경계

공식 평가는 낮을수록 좋은
`0.3 × DINO + 0.3 × Video Feature + 0.4 × Action`이며, Public은 30%,
Private은 70%입니다. 단일 RTX PRO 6000 96GB 기준 학습은 4일,
전체 eval 추론은 1시간 이내입니다.

- 대회가 제공한 train만 학습·검증에 사용합니다.
- eval image/action은 최종 독립 추론 입력으로만 사용합니다.
- eval pseudo labeling, test-time optimization, eval 분포를 이용한
  sampling·expert 선택·threshold tuning을 하지 않습니다.
- 공개 weight와 허용 라이선스를 가진 사전학습 모델만 로컬에서 사용하고,
  모델 이름·버전·URL·license·checksum을 기록합니다.
- 2026-07-24 운영진 공지에 따라 submission kit의 내부 코드, 모델,
  checkpoint, weight, 출력 결과를 loss, metric, validation, candidate
  selection, reranking, 후처리 또는 영상 수정에 사용하지 않습니다.
- 최종 MP4가 전부 고정된 뒤 원본 kit를 수정 없이 CSV 변환에만 한 번
  사용하며, CSV도 수정하지 않습니다.

공식 문서:

- https://dacon.io/competitions/official/236736/overview/description
- https://dacon.io/competitions/official/236736/overview/evaluation
- https://dacon.io/competitions/official/236736/overview/rules
- https://dacon.io/competitions/official/236736/talkboard/417050

## 데이터가 말하는 모델 구조

### 전체 규모와 품질

- 55 owners, 128 LeRobot repositories
- 11,132 episodes, 1,025,666 frames
- 11,098 episodes가 16 frames 이상
- 127 repositories가 6 FPS, 1 repository가 10 FPS
- 123 repositories가 480×640, 나머지는 720p 또는 1080p
- 6D 순서는 shoulder pan/lift, elbow flex, wrist flex/roll, gripper
- eval은 216 samples, 각 initial RGB 480×640과 `(16, 6)` action
- eval action과 전체 train의 연속 16-step action exact match는 없음

명백한 noise와 중복은 `reports/DATA_AUDIT.md`와 manifest 정책을 따릅니다.
특히 서로 다른 command가 거의 같은 영상에 연결된
`Gano007/so100_medic` 전체, exact content duplicate의 non-canonical
사본, 16-frame 미만 episode는 기본 제외합니다. 같은 trajectory가 복사된
repository pair는 같은 validation group에 넣습니다.

### Command와 measured state는 다르다

Parquet에는 다음 두 열이 모두 있습니다.

```text
action:             servo target command, shape (T, 6)
observation.state:  measured joint state, shape (T, 6)
```

전체 frame의 same-step `|action[t]-state[t]|` MAE:

```text
[1.525, 3.408, 3.249, 2.063, 1.902, 3.231]
```

한 step 이동한 `|action[t]-state[t+1]|` MAE:

```text
[0.817, 2.263, 1.935, 1.201, 1.221, 2.635]
```

모든 차원에서 +1 step이 최적입니다. Train-only 1차 servo 회귀
`state[t+1]-state[t] = alpha*(action[t]-state[t]) + bias`의 alpha는:

```text
[1.073, 0.940, 0.735, 1.004, 0.553, 0.390]
```

그리퍼와 wrist roll은 특히 느리고, shoulder pan과 wrist flex는 약간
overshoot합니다. 단순 전역 action normalization이나 raw action 직접
conditioning만으로는 이 현상을 표현하기 어렵습니다.

### Train 내부 action mode 차이

Train-only 진단에서 action 좌표계와 동작 속도는 repository별로 크게
다릅니다. 예시:

- `vladfatu`: 2 env, 100 episodes, 11,997 frames. 16-frame total action
  variation 중앙값 약 148, 거의 정적인 창 약 28.2%.
- `sihyun77`: 8 env, 303 episodes, 42,950 frames. 중앙값 약 211,
  거의 정적인 창 약 15.6%.

이 차이는 absolute와 relative representation, train-only calibration
adapter의 필요성을 보여 줍니다. 그러나 eval을 분석해 특정 두 owner만
expert로 만들거나 eval 비율로 loss를 재가중하면 규칙 위반으로 해석될 수
있습니다. 모든 train repository를 대상으로 cluster를 만들고 동일한
절차로 학습해야 합니다.

## Articulated fallback 아키텍처

### 1. Initial-state encoder와 servo residual dynamics

Shared image backbone의 pooled robot feature에서 초기 measured state와
GRU context를 예측합니다. 이후 target command와 이전 예측 state의 servo
error, 이전 velocity를 입력으로 state를 rollout합니다.

```text
e[t] = projected_action[t-1] - predicted_state[t-1]
nominal[t] = predicted_state[t-1] + gain * e[t] + bias
residual[t] = GRU(
    predicted_state[t-1],
    projected_action[t-1],
    e[t],
    velocity[t-1],
)
predicted_state[t] = nominal[t] + residual[t]
```

구현 원칙:

- `forward`는 image feature와 action만 받습니다.
- measured state는 train-only auxiliary loss의 target으로만 사용합니다.
- teacher forcing으로 true state를 rollout 입력에 넣지 않습니다.
- frame 0 state는 action과 무관하게 image에서만 추정합니다.
- `action[t]`는 frame/state `t+1`을 구동합니다.
- state target normalization statistics도 fold-train에서만 계산합니다.
- state loss는 per-joint robust scale을 적용한 Smooth L1,
  velocity loss, initial-state loss로 구성합니다.

초기 구현은 `src/inha_worldmodel/state_dynamics.py`에 독립 모듈로 두고,
향후 visual backbone과 renderer 사이에 결합합니다.

### 2. Domain-general articulated layered warper

카메라와 배경이 달라져도 첫 이미지에서 보이는 로봇 geometry를 기준으로
움직이도록 설계합니다.

1. Image encoder가 multi-scale feature와 soft layer mask를 예측합니다.
2. 약 7개 slot을 background, upper arm, forearm, wrist, gripper, object,
   unknown/disocclusion에 대응시킵니다.
3. 각 link와 time step에 affine 2×3과 coarse TPS control-point offset을
   예측합니다.
4. Distal link transform은 parent transform과 local transform을 합성해
   SO-100 kinematic chain prior를 반영합니다.
5. 낮은 해상도에서 flow/mask를 예측하되, 원본 480×640 initial image를
   full-resolution `grid_sample`로 warp해 질감과 배경을 보존합니다.
6. 생성 residual은 disocclusion, deforming gripper, moving object에만
   bounded하게 허용합니다.

관절별 causal dependency prior:

- upper arm: shoulder pan/lift
- forearm: shoulder pan/lift + elbow
- wrist: shoulder + elbow + wrist flex/roll
- gripper: upstream joints + gripper command/state
- object: contact 이전 identity, contact 이후 gripper transform에 soft attach

Object slot은 gripper 변화, state trajectory, initial image의
object-gripper geometry로 contact probability를 예측합니다.

### 3. Action/state representation

State dynamics에는 raw command와 measured state가 같은 물리 좌표계로
들어가야 합니다. Renderer에는 다음을 별도 projection으로 제공합니다.

- absolute predicted state
- `state[t] - state[0]`
- 1차·2차 state difference
- raw absolute command
- command error `action[t-1] - state[t-1]`
- gripper state/change
- time Fourier feature
- train-fold global robust scaling

Absolute를 없애면 실제 robot pose를 잃고, relative만 없으면 repository별
calibration offset에 과적합합니다. 둘을 모두 사용합니다.

## Train-only optical-flow 보조 감독

공개 pretrained optical-flow 모델을 사용한다면 train 영상에만 frozen
teacher로 적용합니다. TorchVision RAFT는 공식 구현과 공개 weight를
제공하지만, 코드 라이선스와 weight/학습 데이터 조건이 동일하다고 가정하지
말고 정확한 weight terms를 확인·기록해야 합니다.

- RAFT 문서:
  https://docs.pytorch.org/vision/stable/auto_examples/others/plot_optical_flow.html
- TorchVision license:
  https://github.com/pytorch/vision/blob/main/LICENSE

권장 cache:

- episode당 고정된 16-frame train clip 1개
- `I[0]→I[t]`와 `I[t]→I[t+1]`
- 224×320 정도의 int16 quantized flow
- uint8 forward-backward confidence/occlusion mask

RAFT는 반사면, textureless 영역, disocclusion에서 틀릴 수 있으므로
forward-backward consistency와 photometric residual로 신뢰도를 제한합니다.
Flow는 보조 loss이며 pixel reconstruction이 주 loss입니다.

Renderer loss 후보:

- confidence-masked flow EPE
- Charbonnier reconstruction
- SSIM
- 제출킷과 무관한 독립 LPIPS
- foreground-weighted reconstruction
- direct/composed flow consistency
- background identity
- mask entropy/sparsity
- TPS bending와 piecewise-rigid Jacobian
- temporal velocity/acceleration
- train-only inverse dynamics/action consistency

## Train-only calibration adapters

특정 eval domain을 직접 선택하지 않고 다음 절차를 고정합니다.

1. 128 train environments 각각에서 action/state quantile, IQR, total
   variation, servo residual, gripper transition 통계를 계산합니다.
2. train만으로 K=4~8 cluster를 정합니다.
3. shared visual encoder/renderer에 cluster별 작은 FiLM/LoRA/TPS head를
   붙입니다.
4. inference router는 16-step raw command summary와 initial image의
   foreground robot-pose feature만 사용합니다.
5. leave-environment-out OOF prediction으로 router를 calibration합니다.
6. router entropy가 높으면 generic adapter로 되돌아갑니다.

배경이 router label이 되지 않도록 foreground pooling, background
randomization 또는 environment-adversarial loss를 사용합니다. Hard routing
대신 adapter weight를 soft blend하면 경계 sample의 실패가 줄어듭니다.

## 검증 설계

### 분할

- 같은 episode의 겹치는 window는 절대 다른 fold로 보내지 않습니다.
- exact/cross-repository duplicate group 전체를 한 fold에 둡니다.
- Stage A: 같은 environment 안의 whole-episode holdout
- Stage B: leave-repository/environment-out
- Stage C: leave-owner-out

Stage B와 C를 model selection의 주 기준으로 사용합니다. Camera/domain
generalization이 중요한 만큼 전체 평균뿐 아니라 worst environment,
bottom quartile, task별 metric을 함께 기록합니다. Action/state statistics,
cluster, router, proxy metric은 매 fold의 train 부분에서만 fit합니다.

### 제출킷과 독립적인 metric

- Charbonnier, PSNR, SSIM
- 독립 LPIPS
- GT/prediction optical-flow EPE와 warp error
- temporal LPIPS, velocity, acceleration error
- foreground-weighted metric
- fold-train에서 별도로 학습한 inverse-state/action proxy MAE

Checkpoint composite는 각 metric을 fold 내 robust z-score로 바꾼 뒤 예를
들어 다음으로 시작합니다.

```text
0.20 * LPIPS
+ 0.20 * (1 - SSIM)
+ 0.25 * flow_error
+ 0.35 * inverse_action_error
```

평균 점수만 최소화하지 말고 `mean + 0.25 × worst-domain regret`도 함께
최소화합니다. Public leaderboard 30%는 최종 확인용이며 checkpoint나
per-sample candidate 선택에 사용하지 않습니다.

## Sampling과 augmentation

- episode-uniform을 기준으로 owner-tempered 지수 0.5를 별도 비교
- low/medium/high motion과 gripper transition/contact window를 균형화
- 정적 window도 충분히 유지해 background identity를 학습
- 유일한 10 FPS repository는 6 FPS로 시간 resample하거나 fps token 부여
- 4:3을 주력으로 하고 16:9/high-resolution source는 동일 pad 정책 적용
- 제공 train 안에서만 background swap과 appearance randomization
- image/video/flow에 동일한 affine, perspective, crop, color, codec 변환
- action geometry를 정확히 변환하지 않는 horizontal flip은 금지
- 비가역 contact가 있으므로 temporal reversal은 기본 금지

## 단계별 실험

### P0: DynamiCrafter-plus

1. 공개 backbone + 대회 제공 1,500-step main UNet/EMA를 expected-model
   tensor 기준으로 100% 검사
2. strict owner/repository/duplicate-group holdout에서 100-step smoke
3. same-step/previous-command alignment를 각각 별도 학습해 비교
4. 320×512, 384×512, 480×640 quality/runtime 비교
5. episode-uniform과 owner-tempered sampling 비교
6. raw6와 absolute+delta+velocity 18D 비교. 첫 action MLP의 새 열은
   0-init하여 제공 checkpoint와 step-0 함수가 같음
7. 3k/5k/8k/12k... checkpoint를 고정 DDIM holdout으로 선택
8. 선택 recipe를 manifest-approved 11,002 episodes 전체로 한 번 refit

모든 비교 metric은 padding을 제거한 원본 train-video 해상도에서 계산하고,
같은 sample의 cross-clip action control이 원래 action보다 나빠지는지
paired gate로 확인합니다.

### P1: Cosmos-Predict2.5 2B

Cosmos-Predict2.5에는 robot/action-conditioned 2B와 action-conditioned
distillation recipe가 공개되어 있습니다.

https://github.com/nvidia-cosmos/cosmos-predict2.5

다만 공개 robot recipe의 7D Bridge action과 이 대회의 6D absolute SO-100
joint command는 의미가 다릅니다. 6D action adapter를 새로 학습하고
temporal/action LoRA로 미세조정한 뒤 DMD2 4-step distillation이 필요합니다.
100-step wall-time과 전체 216-sample dry run이 각각 4일/1시간 제한을
통과할 때만 유지합니다. Custom 6D recipe가 준비되지 않은 Cosmos 3는 현재
주력에서 제외합니다. NVIDIA source license와 model license는 별도로
검토·기록합니다.

### P2: Articulated 고해상도 fallback

1. Dense direct-flow U-Net + identity background + bounded residual
2. Initial-state encoder + causal servo residual GRU
3. soft link layers + per-link affine
4. affine + TPS + kinematic chain prior
5. object/contact slot

각 단계가 leave-environment-out OOF를 개선할 때만 다음 복잡도를 유지합니다.
Unsupervised slot이 collapse하면 dense flow가 안전망입니다. Articulated
결과를 조건으로 한 작은 deterministic residual refiner는 배경을 다시
그리지 않는 경우에만 유지합니다.

## Ensemble과 최종 영상

- Pixel-space seed/model 평균은 흐림을 만들기 쉬우므로 사용하지 않습니다.
- EMA/SWA, flow/mask-space soft blend, OOF prediction으로 학습한 작은
  fusion refiner를 우선합니다.
- Candidate/seed 선택은 train OOF metric만 사용합니다.
- Submission kit 결과로 seed 선택, rerank, postprocess를 하지 않습니다.
- 출력은 정확히 16 frames, 6 FPS, 480×640입니다.
- Frame 0은 exact initial image로 덮어씁니다.
- 고정 seed와 codec 설정으로 MP4를 만들고 decoded frame count, resolution,
  FPS, color range만 일반 도구로 QA합니다.
- 216 samples / 3600 seconds = 16.7 seconds/sample이 절대 상한입니다.
  Kit와 I/O margin을 남겨 generator는 10~12 seconds/sample 이하를 목표로
  합니다.

Articulated warper는 한 번의 병렬 forward로 충분합니다. DynamiCrafter는
holdout에서 고른 15~50 DDIM step 중 전체 1시간 gate를 통과하는 하나만
고정합니다. Cosmos는 4-step DMD2 gate를 사용합니다.

## 4일 GPU 우선순위

```text
0.25 day  DynamiCrafter 환경·100-step·action-sensitivity gate
0.75 day  alignment/resolution/sampling/DDIM ablation
1.75 day  선택 DynamiCrafter recipe 학습과 checkpoint 검증
0.75 day  manifest-approved 전체-data refit
0.5 day   216-sample total-wall dry run, MP4 감사, 재현성
```

중도 탈락 기준을 미리 정합니다.

- 환경 holdout을 개선하지 못하는 module
- 전체 inference 1시간을 넘기는 model
- 4일 내 full training이 불가능한 candidate
- 특정 train 배경에만 반응하는 router
- cross-clip action에도 거의 같은 영상을 만드는 generator
- residual이 배경 전체를 다시 그리는 model

## 가장 큰 위험

1. Eval 분포 관찰을 학습 설계에 반영한 것으로 해석되어 실격
2. Action과 measured state의 한-step lag를 무시해 robot pose가 앞서거나
   뒤처짐
3. 새 카메라에서 image-to-state 또는 layer mask가 실패
4. Unsupervised layer collapse
5. Contact/object motion을 identity background로 덮어버림
6. Public 30% 과적합
7. 공개 pretrained weight의 license 증빙 부족
8. Training 4일 또는 inference 1시간 재현 실패

따라서 최종 구현에는 eval group 수, eval-derived threshold, 특정 eval
domain용 expert, submission-kit-derived metric이 전혀 남지 않아야 합니다.
점수와 코드 검증을 동시에 통과하는 것이 실제 1위의 조건입니다.
