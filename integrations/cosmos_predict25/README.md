# Cosmos-Predict2.5 2B SO-100 adapter

이 디렉터리는 대회 manifest와 NVIDIA Cosmos-Predict2.5를 연결하는 독립
어댑터다. 공식 제출 키트를 import하거나 읽지 않는다. 현재 구현 과정에서는
체크포인트를 다운로드하지 않았고 학습·증류·추론도 실행하지 않았다.

## 고정 계약

| 항목 | 값 |
|---|---:|
| Cosmos 학습 비디오 | RGB `uint8 [C,16,256,320]` |
| 제출 비디오 | MP4 `640×480`, 16 frames, 6 fps |
| 조건 프레임 | 맨 앞 clean frame 1장 |
| 원본 action | `float32 [16,6]` |
| 인과 조건 | `action[0:15] -> video[1:16]`, 즉 `[15,6]` |
| 마지막 action | `action[15]`, out-of-horizon metadata로만 보존 |
| 시간 해상도 | 6 fps |
| 정규화 | train fold에서만 적합한 median/IQR, 기본 clip ±8 |
| validation 단위 | audit를 통과한 `folds.json`의 exact episode keys |
| Cosmos 네트워크 | `cosmos_v1_2B_action_conditioned` |
| 초기 체크포인트 | `2B/robot/action-cond` |
| 모델 입력 | WAN VAE 경계에서만 17프레임으로 패딩 |

`observation.state`는 `return_measured_state=True`일 때
`measured_state_target [15,6]`으로만 반환한다. action 조건으로 섞지 않는다.
manifest의 `include_for_training=false` episode는 항상 제외하며, sparse episode
ID를 연속 정수라고 가정하지 않는다. 기본 실험은
`seeded_group_00_seed_17`을 사용해 owner, repository, validation group이
train/validation 양쪽에 나타나지 않게 한다.

### 16프레임과 WAN VAE

Predict2.5의 WAN tokenizer는 latent `L`개를 pixel `(L-1)*4+1`개로
복원한다. 따라서 네이티브 pixel 길이는 1, 5, 9, 13, 17, ...이고 16은
직접 표현할 수 없다. 이 어댑터는 다음 경계를 강제한다.

1. 데이터셋과 평가 계약은 그대로 16프레임/15 action이다.
2. 학습 collate에서 frame 15를 한 번 반복해 17프레임으로 만든다.
3. `state_t=5`로 Cosmos를 실행한다.
4. 생성 결과의 synthetic tail frame 16만 제거한다.

`prepare_cosmos_config.py`는 `state_t=4`/16프레임처럼 실행 시 실패할 조합을
거부한다. 15 action은 action-chunk 네트워크의 4-action latent grouping으로
나누어지지 않으므로 전역 action-conditioned 네트워크를 사용한다.

## 파일과 인터페이스

- `so100_dataset.py`
  - `fit_robust_action_stats(...)`
  - `SO100CosmosDataset(...)`
  - `cosmos_collate_fn(...)`
- `inference_adapter.py`
  - `prepare_eval_condition(image, actions, ...)`
  - `load_eval_condition(image_path, actions_path, ...)`
  - `trim_cosmos_generated_video(video)`
- `prepare_cosmos_config.py`
  - main 소스 계약 검증
  - JSON patch spec 생성
  - upstream Hydra overlay 생성
  - action embedder 이외 shape mismatch를 거부하는 CPU guard

Dataset 샘플의 핵심 키는 다음과 같다.

```python
sample = {
    "video": ...,                  # uint8 [3,16,256,320] (native checkpoint size)
    "action": ...,                 # normalized float32 [15,6]
    "raw_action": ...,             # raw float32 [15,6]
    "fps": ...,                    # 6.0
    "padding_mask": ...,           # aspect padding mask [1,256,320]
    "num_conditional_frames": 1,
}
```

Upstream `ActionConditionedConditioner`가 읽는 key는 단수형 `action`이다.
10 fps episode는 실제 timestamp에 맞춰 6 fps index
`[0,2,3,5,7,8,10,12,13,15,17,18,20,22,23,25]`로 리샘플한다.
이미지는 찌그러뜨리지 않고 공식 action-cond 체크포인트의 256×320
native resolution으로 letterbox한다. 추론 후 별도 래퍼가 640×480 MP4로
변환한다. 480×640로 2B 백본을 직접 학습하면 공개 체크포인트의 공간
계약과 VRAM 예산을 동시에 깨므로 사용하지 않는다.

## CPU 준비와 검증

작업 디렉터리에서 train-fold 통계부터 만든다.

```bash
cd /Users/choeseong-yong/Inha_challenge

python integrations/cosmos_predict25/so100_dataset.py fit-stats \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --data-root /Users/choeseong-yong/Downloads/open/data/train \
  --fold-artifact artifacts/folds/folds.json \
  --fold-id seeded_group_00_seed_17 \
  --output artifacts/cosmos_predict25/action_robust_stats_fold17.json

python -m pytest tests/test_cosmos_adapter.py -q
```

adapter는 fold artifact의 전체 audit와 선택 fold audit가 통과했는지 확인하고,
artifact의 `source_manifest.sha256`를 실제 manifest SHA256과 비교한다. 이어서
`train_episode_keys`와 `validation_episode_keys`가 included episode의 정확한
비중복 partition인지 재검증한다. 통계 JSON의 `split_signature`에는 fold ID,
fold artifact 자체 SHA256, source manifest SHA256가 들어간다. artifact가
재생성되거나 바뀐 통계를 실수로 재사용하면 dataset 생성이 실패한다.

기존 `val_groups` 또는 `val_fraction`/`split_seed` API도 하위 호환을 위해
유지하지만 audited fold와 동시에 지정할 수는 없다.

평가 입력 하나를 독립 NPZ로 점검하는 명령은 다음과 같다.

```bash
python integrations/cosmos_predict25/inference_adapter.py \
  --image /path/to/000.png \
  --actions /path/to/000.npy \
  --stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --sample-id 000 \
  --output artifacts/cosmos_predict25/eval_000_condition.npz
```

출력 sidecar JSON에는 사용하지 않은 `action[15]`가 그대로 기록된다. 후보
생성, 후보 선택, 제출 키트 호출은 이 어댑터의 범위가 아니다.

## Upstream main overlay

검토한 upstream commit은
[`a2c298b0`](https://github.com/nvidia-cosmos/cosmos-predict2.5/tree/a2c298b0a3df3778b973fe65e9e58877b292d8a7)이다.
moving `main`을 그대로 신뢰하지 말고 실제 학습 run은 이 commit에 고정한다.

기존 upstream checkout을 검증하고 overlay를 생성하는 명령:

```bash
cd /Users/choeseong-yong/Inha_challenge

python integrations/cosmos_predict25/prepare_cosmos_config.py \
  --upstream-root /path/to/cosmos-predict2.5 \
  --strict-commit \
  --output artifacts/cosmos_predict25/config_report.json \
  --write-overlay /path/to/cosmos-predict2.5/cosmos_predict2/experiments/inha_so100.py
```

이 명령은 로컬 소스만 읽고 설정 파일을 만들며 모델을 다운로드하지 않는다.
생성되는 overlay는 다음 설정을 강제한다.

- `action_dim=6`
- `num_action_per_chunk=15`
- dataset 16프레임, WAN model 17프레임, `state_t=5`
- 256×320 native model resolution, 6 fps, conditional frame 1장
- `context_parallel_size=1`
- 공식 `2B/robot/action-cond` UUID
  `38c6c645-7d41-4560-8eeb-6f4ddc0e6574`
- 체크포인트에서 아래 두 입력 projection weight만 skip/reinitialize
  - `action_embedder_B_D.fc1.weight`
  - `action_embedder_B_3D.fc1.weight`

두 bias와 action embedder의 나머지 layer는 불러온다. 다른 tensor의 shape가
다르면 `filter_checkpoint_shape_mismatches(...)`가 실패해야 한다.
`strict_resume=False`는 위 두 누락 weight를 허용하기 위한 것이며, 광범위한
무시 정책으로 사용하면 안 된다.

## 학습 명령 — 사용자가 실행

먼저 Hugging Face에서 [Cosmos-Predict2.5-2B 모델 페이지](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)의
접근 조건을 승인하고, 서버에서 Cosmos 전용 환경에 로그인한다. `hf` 명령은
기본 `inha_sy` 환경이 아니라 upstream `.venv`에 설치되어 있다.

```bash
/home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5/.venv/bin/hf auth login
```

로그인 후 다음 명령을 사용한다. 래퍼는 `inha_sy`에서 호출해도 공식
`cosmos-predict2.5/.venv/bin/torchrun`과 `cosmos_oss`를 자동으로 선택한다.
아래 학습 명령을 실행하면 upstream이 공식 체크포인트를 내려받을 수 있다.

```bash
conda activate inha_sy
cd /home/video_generation/inha_challenge/ChoiSeongYong/Inha_challenge
export CUDA_VISIBLE_DEVICES=1

python scripts/run_cosmos_predict25_so100.py \
  --upstream-root /home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5 \
  --open-root /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge \
  --output-root outputs/cosmos_predict25_so100_60k \
  --max-iter 60000 \
  --grad-accum-iter 8 \
  --execute
```

래퍼는 먼저 upstream commit/source contract, audited train fold, train-only
action 통계를 검증하고, `cosmos_predict2/experiments/inha_so100.py` overlay를
생성한 다음 공식 `scripts.train`을 단일 GPU로 실행한다. `--execute`를 빼면
명령과 설정만 기록한다. `--execute`는 공식 checkpoint 다운로드/사용을
시작할 수 있으므로 NVIDIA 모델 사용권을 먼저 수락해야 한다.

학습 로그와 checkpoint는 `--output-root` 아래의 공식 Cosmos 출력 구조에
생성된다. 중단 후 재개할 때는 같은 output root와 같은 config를 유지하고,
upstream 공식 checkpoint resume 규칙에 맞춰 실행한다. 새 output root를
만들어 초기 checkpoint부터 다시 시작하면 안 된다.

## 추론 명령 — 16개 action으로 16프레임 생성

학습 checkpoint를 공식 `convert_distcp_to_pt.py`로 `model_ema_bf16.pt`로
변환한 뒤 다음 래퍼를 사용한다.

먼저 동일한 checkpoint와 동일한 `num_steps`로 8개 샘플 benchmark를 실행한다.

```bash
python scripts/infer_cosmos_predict25_so100.py \
  --upstream-root /home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5 \
  --eval-root /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge/data/eval \
  --stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --checkpoint outputs/cosmos_predict25_so100_60k/.../model_ema_bf16.pt \
  --output-root outputs/cosmos_predict25_benchmark_35step \
  --num-steps 35 \
  --limit 8 \
  --execute
```

benchmark의 보수적 1.2배 projected wall time이 55분 미만일 때만 전체
inference가 허용된다. 최종 명령은 반드시 `--budget-manifest`를 포함한다.

```bash
python scripts/infer_cosmos_predict25_so100.py \
  --upstream-root /home/video_generation/inha_challenge/ChoiSeongYong/cosmos-predict2.5 \
  --eval-root /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge/data/eval \
  --stats artifacts/cosmos_predict25/action_robust_stats_fold17.json \
  --checkpoint outputs/cosmos_predict25_so100_60k/.../model_ema_bf16.pt \
  --output-root outputs/cosmos_predict25_inference_35step \
  --num-steps 35 \
  --budget-manifest outputs/cosmos_predict25_benchmark_35step/inference_manifest.json \
  --execute
```

이 명령은 공식 action-conditioned inference를 호출해 256×320에서 16프레임을
생성하고, 제출용 `predictions/*.mp4`를 640×480/6fps로 만든다. 최종 대회
제출 전에는 먼저 submission kit과 분리된 MP4 audit를 실행한다.

```bash
python scripts/audit_videos.py \
  --video-root outputs/cosmos_predict25_inference_35step/predictions \
  --eval-root /home/video_generation/inha_challenge/ChoiSeongYong/data_challenge/data/eval \
  --expected-count 216 \
  --output outputs/cosmos_predict25_inference_35step/video_audit.json
```

audit가 통과한 뒤에만 MP4를 공식 kit의 `input_videos/`에 복사하고
`make_submission_csv.py`를 실행한다. 모델 래퍼는 CSV를 만들거나 후처리하지
않는다. inference 래퍼가 함께 저장하는
`predictions/inference_provenance.json`은 모델·checkpoint·조건·MP4 hash를
고정하는 용도다.

Upstream 공식 문서도 action-conditioned model은 multi-GPU/context parallel을
지원하지 않는다고 명시하므로 `nproc_per_node=1`,
`context_parallel_size=1`을 유지한다.

## RTX PRO 6000 96GB: 1시간 gate와 4일 예산

1시간 gate와 100-step GPU smoke test를 통과하기 전에는 4일 run을 시작하지
않는다. 전체 inference는 같은 step 수의 benchmark manifest가 없으면
실행 자체를 거부한다.

Gate 통과 조건:

1. CPU 테스트와 upstream strict-commit/config validation이 모두 통과한다.
2. 64개 window decode에서 shape, finite action, 인과 index 오류가 0개다.
3. BF16, batch 1, gradient accumulation 8로 OOM 없이 최소 100 update가 돈다.
4. peak allocated memory가 90 GiB 이하이고 loss/gradient가 모두 finite다.
5. 작은 고정 validation subset에서 초기 checkpoint보다 video proxy가
   개선되며, action shuffle 시 성능이 나빠져 action을 실제로 사용함이 보인다.
6. generated frame 0이 clean condition을 보존하고 tail trim 후 정확히
   16프레임이다.

권장 96시간 배분:

- 0–1h: 위 gate. 하나라도 실패하면 중단하고 원인 수정.
- 1–40h: 2B teacher post-training, 자주 checkpoint/고정 validation.
- 40–56h: checkpoint 비교, action perturbation과 10 fps subset 별도 점검.
- 56–82h: teacher가 확실히 개선된 경우에만 DMD2 증류.
- 82–94h: 선택 checkpoint의 전체 eval inference와 artifact 검증.
- 94–96h: 재실행·복구 버퍼.

긴 run에서는 train loss 최저값이 아니라 validation video 품질과
action-sensitivity를 함께 사용해 checkpoint를 고른다.

## DMD2 4-step — SO-100 계약 검증 전에는 실행 금지

공식 [action post-training 문서](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md)는
다음 DMD2 experiment와 inference `--num-steps 4`를 제시한다.

```bash
torchrun --nproc_per_node=1 --master_port=12341 -m scripts.train \
  --config=cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  -- experiment=dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320_no_s3
```

```bash
python examples/action_conditioned.py \
  --config-file cosmos_predict2/_src/interactive/configs/registry_predict2p5.py \
  --checkpoint-path /path/to/model_ema_bf16.pt \
  --experiment dmd2_trigflow_distill_cosmos_predict2_2B_action_conditioned_bridge_13frame_256x320_no_s3 \
  --num-steps 4
```

이 공식 recipe는 Bridge 7D/12-action/13-frame용이다. SO-100
6D/15-action/16→17 adapter에는 그대로 실행하면 안 된다. teacher가 validation에서
개선되고도 benchmark를 통과하지 못할 때만, teacher와 동일한 dataset, network,
action shape, tail-padding contract를 반영한 별도 DMD2 experiment를 추가한다.

## 공식 자료와 라이선스

- [Cosmos-Predict2.5 source](https://github.com/nvidia-cosmos/cosmos-predict2.5)
- [Robot action-conditioned inference](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/inference_robot_action_cond.md)
- [Action-conditioned post-training / DMD2](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/post-training_video2world_action.md)
- [General distillation guide](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/docs/distillation.md)
- [Official 2B model card/checkpoint repository](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- [Source LICENSE: Apache-2.0](https://github.com/nvidia-cosmos/cosmos-predict2.5/blob/main/LICENSE)
- [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

코드는 Apache-2.0이고 model weight는 NVIDIA Open Model License 조건을
따른다. 대회 제출·재배포 전에 양쪽 조건과 대회 규정을 별도로 확인해야 한다.

## Cosmos3를 이번 경로에서 제외한 이유

[공식 Cosmos 저장소](https://github.com/NVIDIA/cosmos)의 현재 roadmap은
customization/post-training recipe를 `Coming Soon`으로 표시한다. 제공된
robot embodiment action dimension도 고정되어 있으며, 이 대회의 custom
6D action을 fine-tune하는 공개 recipe가 아직 없다. 따라서 재현 가능한
6D post-training 경로가 있는 Predict2.5를 사용하고, Cosmos3 recipe가 실제
공개되면 다시 평가한다.

## 알려진 위험

- 16프레임이 WAN의 네이티브 길이가 아니므로 반복 tail supervision이
  마지막 동작을 약간 정적으로 만들 수 있다.
- 공식 action checkpoint는 Bridge의 7D 상대 Cartesian action에 맞춰져
  있고 SO-100 입력은 6D raw command다. robust normalization과 입력
  projection 재초기화가 representation gap을 완전히 해결하지는 않는다.
- action-chunk 네트워크 대신 전역 action embedding을 써야 하므로 긴 horizon
  시간 국소성이 약할 수 있다.
- 공식 Bridge loader와 동일하게 text embedding은 기본 zero placeholder다.
- 10→6 fps는 선택 시점의 raw command를 사용하며 중간 10 Hz command를
  적분하지 않는다.
- letterbox padding bar가 학습 분포에 들어간다.
- audited fold는 owner/repository/validation-group 겹침을 막지만, 서로 다른
  owner가 수집한 의미적으로 유사한 장면까지 모두 탐지하는 보장은 없다.
- upstream은 현재 제한적 유지보수 상태다. commit pin과 source-contract
  validation 없이 moving main을 사용하면 안 된다.
