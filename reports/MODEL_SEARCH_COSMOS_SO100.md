# SO-100 대회용 신규 모델 조사와 구현 결정

작성일: 2026-08-06

## 결론

현재 프로젝트의 DynamiCrafter를 다시 튜닝하는 대신, 1순위 후보로
NVIDIA **Cosmos-Predict2.5 2B Robot / Action-Conditioned**를 선택했다.
이 모델은 첫 이미지와 action sequence를 받아 미래 영상을 생성하는 2B
rectified-flow DiT world model이다. 대회의 입력 계약과 모델의 입력 의미가
직접 대응하고, 공식 action-conditioned post-training 문서와 Blackwell
실행 경로가 공개되어 있다는 점이 가장 큰 장점이다.

이 선택은 “성능이 반드시 더 높다”는 보장이 아니다. 대회 평가 데이터는
학습에 사용할 수 없으므로, 제공된 train fold의 held-out validation과
action-shuffle 검증을 통과한 뒤에만 실제 장기 학습 checkpoint를 선택해야
한다.

## 후보 비교

| 후보 | 장점 | 이번 대회에서의 결정 |
|---|---|---|
| Cosmos-Predict2.5 2B Action-Cond | 첫 이미지+action→미래 영상, 공식 6D adapter 작성 가능, 공식 post-training/DMD2 경로 | **실행 후보** |
| Cosmos 3 | 더 최신 physical-AI world model, action-conditioned forward dynamics | custom SO-100 6D post-training recipe와 고정 action layout이 없어 보류 |
| OSCAR-2B | skeleton-conditioned robot video에 강한 최신 연구 후보 | 대회 action `[16,6]`를 skeleton video로 바꾸는 별도 kinematics/conditioner가 필요하고 공개 inference 중심이라 위험 |
| VidMan/Vidar | robot world model 및 inverse-dynamics 연구 근거 | 공개 checkpoint·대회 입력 계약·재현 가능한 학습 recipe가 부족 |
| DynamiCrafter Plus | 현재 결과가 있고 즉시 제출 가능 | 신규 성능 상한 후보가 아니라 안전한 fallback |

## 대회 규칙과의 적합성

공식 대회 입력은 initial image와 action sequence이고 미래 video를 생성하는
문제다. 평가 데이터는 학습에 사용할 수 없고, 공개 pretrained model은
라이선스가 허용하는 경우 사용할 수 있다. 학습은 1× RTX PRO 6000 기준
4일, 평가 inference는 1시간 제한이 있으므로 단일 GPU 경로와 빠른 추론을
별도로 고려했다.

## 구현한 구조

1. `so100_dataset.py`: audited train fold만 읽고 6 Hz로 window를 만들며,
   median/IQR action normalization을 train fold에서만 적합한다.
2. Cosmos의 공식 global action-conditioned network를 사용한다. 공식
   checkpoint는 7D×12 action 입력이고, 대회는 6D×15 action이므로 두 개의
   action input `fc1` weight만 재초기화한다. 다른 shape mismatch는 즉시
   실패한다.
3. WAN temporal contract 때문에 대회의 16프레임을 학습 경계에서 17프레임으로
   한 번만 tail-pad하고, 결과에서 synthetic tail 한 프레임을 제거한다.
4. 공식 checkpoint의 native spatial resolution인 256×320으로 학습/추론하고,
   추론 후 640×480/6fps/16-frame MP4로 변환한다. 2B backbone을 480×640으로
   직접 학습하지 않아 VRAM과 pretrained spatial contract를 보존한다.
5. `run_cosmos_predict25_so100.py`가 upstream source validation, fold stats,
   Hydra overlay, 공식 `scripts.train` 실행을 하나로 묶는다.
6. `infer_cosmos_predict25_so100.py`와 `cosmos_action_loader.py`가 공식
   `examples/action_conditioned.py`를 호출해 eval 216개를 처리하고,
   submission-ready MP4를 별도 폴더에 만든다. 제출 CSV는 submission kit로만
   생성한다.

## 중요한 제한

Cosmos teacher는 공식 기본 설정상 35 denoising steps라 216개 inference가
대회 1시간을 넘을 수 있다. 구현한 inference 래퍼는 같은 step 수의 8-sample
benchmark가 없으면 전체 실행을 거부하고, 측정 wall time의 1.2배 projected
time이 55분 이상이면 실행하지 않는다. 따라서 1시간 제한을 추정으로
무시하지 않는다.

benchmark를 통과하지 못하지만 teacher가 held-out validation에서 기존
모델보다 실제로 좋아진 경우에만, 동일한 SO-100 6D/15-action/17-frame
계약으로 DMD2 학생 모델을 추가 학습해야 한다. 공식 Bridge용 DMD2 명령을
그대로 실행하면 action 차원·길이 계약이 달라 실패하므로 그대로 복사하지
않는다. 현재 구현의 확정 범위는 teacher post-training, benchmark gate,
추론, MP4 검증, 제출용 artifact 생성이다.

## 참고 자료

- [Cosmos-Predict2.5 공식 저장소](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [공식 Robot Action-Conditioned inference 안내](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference_robot_action_cond.md)
- [공식 Action-Conditioned post-training/DMD2 안내](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md)
- [공식 2B action-conditioned checkpoint 정보](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [공식 대회 규칙](https://dacon.io/competitions/official/236736/overview/rules)
