# Cosmos-Predict2.5 32K 추가 학습 모델

## 1. 최종 모델 개요

이 모델은 NVIDIA Cosmos-Predict2.5 2B action-conditioned 모델을 기반으로
SO-100 로봇 데이터에 추가 학습한 모델이다.

입력은 현재 로봇 이미지 1장과 6차원 action sequence 16개이며, 학습 데이터의
action 통계로 정규화한다. 출력은 미래 영상 16 frames, 6 fps, 640×480 MP4다.

최종 추가 학습은 14,000 step checkpoint에서 재개하여 32,000 step까지
진행했다.

## 2. 최종 checkpoint

학습용 DCP checkpoint:

```text
outputs/cosmos_predict25_so100_continue_14k_to_32k/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/iter_000032000/
```

추론용 EMA checkpoint:

```text
outputs/cosmos_predict25_so100_continue_14k_to_32k/inha_cosmos_predict25/so100_6d/predict25_2b_action_16f/checkpoints/iter_000032000/model_ema_bf16.pt
```

`iter_000014000`에서 model·optimizer·scheduler·trainer state를 함께
복원했다. checkpoint는 2,000 step 간격으로 저장되었다.

## 3. 학습 설정

- Backbone: Cosmos-Predict2.5 2B action-conditioned
- 학습 입력 해상도: 320×256
- 출력 계약: 640×480, 16 frames, 6 fps
- Action dimension: 6
- Action chunk: 15
- Gradient accumulation: 8
- GPU: 단일 NVIDIA RTX PRO 6000 Blackwell 96GB
- Attention backend: `transformer_engine`
- 최종 step: 32,000

재개에 사용한 핵심 설정:

```text
checkpoint.load_path=.../iter_000014000
checkpoint.load_training_state=true
checkpoint.strict_resume=false
model.config.net.atten_backend=transformer_engine
```

## 4. 구현상 중요한 수정

### 4.1 Cosmos upstream 네트워크 호환성

Cosmos upstream action network가 상위 설정의
`temporal_compression_ratio`를 직접 받지 못하므로 네트워크 생성 전에 해당
항목을 제거했다.

### 4.2 SO-100 zero text embedding

SO-100 데이터에는 Reason1 텍스트 embedding이 없으므로 zero embedding을
사용한다. Cosmos action-conditioned projection과 cross-attention 입력 계약에
맞춰 샘플마다 다음 형태로 제공한다.

```text
(1, 100352) = (1, 98 × 1024)
```

구현 위치:

```text
integrations/cosmos_predict25/so100_dataset.py
```

### 4.3 Blackwell attention backend

기본 `minimal_a2a` backend는 현재 PyTorch/CUDA 조합에서
`No available kernel` 오류를 발생시켰다. `transformer_engine` backend로
변경한 뒤 14,000 step checkpoint에서 실제 1-step 추가 학습을 통과했다.

검증 결과:

```text
Iteration 14001
Loss: 0.0332
Done with training.
```

## 5. 학습 검증 결과

최종 학습 로그에는 다음 완료 상태가 기록되어 있다.

```text
Done with training.
```

최신 checkpoint:

```text
iter_000032000
```

## 6. 최종 추론 결과

32,000 step EMA checkpoint로 평가 샘플 216개를 추론했다.

결과 위치:

```text
outputs/cosmos_predict25_inference_32k_35step_retry2/predictions/
```

추론 manifest:

```text
prediction_count: 216
returncode: 0
total_wall_seconds: 705.7124791339738
```

전체 추론 시간은 약 11분 46초로, 단일 RTX PRO 6000 기준 1시간 제한을
충족한다.

추론 wrapper:

```text
scripts/infer_cosmos_predict25_so100.py
```

공식 Cosmos inference subprocess는 반드시 upstream uv 환경의
`cosmos-predict2.5/.venv/bin/python`으로 실행한다. 이미 다운로드된
Cosmos/Wan/Reason1 artifact를 사용하도록 `HF_HUB_OFFLINE=1`과 프로젝트의
Hugging Face cache도 전달한다.

## 7. 제출 파일 생성

최종 MP4가 확정된 뒤 공식 submission kit의 `make_submission_csv.py`만
사용하여 CSV를 생성한다. submission kit의 모델이나 checkpoint는 학습,
검증, 모델 선택, 영상 선택, 영상 후처리에 사용하지 않는다.

최종 제출 파일:

```text
outputs/cosmos_predict25_inference_32k_35step_retry2/submission_features.csv
```

CSV는 생성 후 수정하지 않고 데이콘에 제출한다.

## 8. 재현 시 주의사항

1. 추론에는 반드시 `model_ema_bf16.pt`를 사용한다.
2. 학습 재개에는 EMA 파일이 아니라 DCP checkpoint directory를 사용한다.
3. 학습 시 `transformer_engine` backend를 유지한다.
4. Cosmos upstream `.venv`와 프로젝트 `inha_sy` 환경을 혼동하지 않는다.
5. full inference는 `num_steps=35` benchmark manifest를 사용한다.
6. 최종 MP4와 CSV는 submission kit로 생성한 뒤 수동 수정하지 않는다.
