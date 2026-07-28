# Final pre-submission audit

이 감사기는 이미 확정된 단 하나의 216개 MP4 묶음이 최종 CSV 변환 단계로
넘어갈 수 있는지를 확인합니다. Submission kit을 import·실행하지 않으며 CSV,
score, metric, feature, embedding, candidate 또는 reranking 입력을 받지
않습니다. 경로 이름이나 provenance 값이 submission-kit 디렉터리를 가리키는
경우에도 읽기 전에 거부합니다.

## 실행 시점

고정 holdout에서 checkpoint·sampler·해상도·action 정렬을 모두 선택하고,
전체 정제 train 데이터 refit과 최종 216-sample 추론이 끝난 뒤 한 번
실행합니다. `scripts/infer_dynamicrafter_plus.py`가 같은 출력 디렉터리에
기록한 `inference_provenance.json`이 필요합니다.

```bash
PYTHONPATH=src:. python scripts/audit_pre_submission.py \
  --video-root artifacts/predictions/final \
  --eval-root data/eval \
  --provenance artifacts/predictions/final/inference_provenance.json \
  --output artifacts/predictions/final/pre_submission_audit.json
```

감사 파일은 불변 증거로 취급하므로 기존 파일을 덮어쓰지 않습니다. 재실행할
때는 새 출력 경로를 사용합니다.

## 통과 조건

- eval image/action ID가 정확히 일치하고 개수가 정확히 216개
- 최상위 출력 디렉터리에 ID별 MP4가 정확히 하나씩 있으며 누락, 추가,
  중첩 MP4가 없음
- 모든 MP4가 16 frames, 6 FPS, 640×480이고 독립 비디오 감사가 성공
- MP4 SHA-256이 inference provenance와 독립 비디오 감사에서 동일
- 각 decoded frame 0이 해당 eval PNG와 codec 허용 범위 안에서 동일
  - mean absolute pixel error `<= 8`
  - PSNR `>= 28 dB`
- provenance가 sample당 하나의 고정 후보만 생성했음을 명시
  - candidate count `1`
  - selection `none`
  - reranking `false`
  - evaluation feedback `false`
- checkpoint main action UNet와 EMA가 모두 완전하게 load됨
- checkpoint 내 contract와 inference가 재구성한 contract가 완전히 동일
- 실제 checkpoint, ordered config, action statistics, clean manifest,
  fold artifact, 공개 backbone과 제공 초기 checkpoint의 파일 크기와
  SHA-256이 provenance와 동일
- 216개 eval PNG/action NPY의 현재 SHA-256이 추론 당시 조건 해시와 동일
- 최종 provenance 원자적 쓰기까지 포함하도록 1초 여유를 더한 인증
  wall-time 상한이 양수이고 `3600`초 미만
- provenance가 `submission_kit_used=false`이고 score/feature 계열 증거
  필드를 포함하지 않음

통과 보고서는 MP4 묶음과 condition 묶음에 각각 순서가 고정된 aggregate
SHA-256을 기록합니다. `authorized_next_step`은
`single_final_mp4_to_csv_conversion_only`입니다. 이는 모델 선택이나
재생성을 허가한다는 뜻이 아니며, 원본 submission kit을 수정 없이 단 한 번
MP4→CSV 변환에 사용하는 단계만 허가합니다.

## 실패 처리

어떤 검사라도 실패하면 exit code 1과 `passed=false` 보고서를 남기며 최종
변환을 중단합니다. MP4, provenance 또는 source artifact를 수정한 뒤 기존
감사 결과를 재사용해서는 안 됩니다. 수정이 필요한 경우 최종 추론을 새
디렉터리에서 다시 실행하고 새 감사 보고서를 생성합니다.

구현은 다음 파일에 있습니다.

- `src/inha_worldmodel/pre_submission_audit.py`
- `scripts/audit_pre_submission.py`
- `tests/test_pre_submission_audit.py`
