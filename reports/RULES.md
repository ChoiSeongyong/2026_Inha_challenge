# 대회 규칙 스냅샷

확인 시각: 2026-07-25 KST

## 문제와 점수

- 현재 이미지와 Action Sequence로 16-frame 로봇 미래 영상을 생성합니다.
- `Score = 0.3 × DINO + 0.3 × Video Feature + 0.4 × Action`
- 낮을수록 좋습니다.
- Public은 eval 30%, Private은 나머지 70%이며 최종 순위는 Private 100%입니다.

공식 문서:

- https://dacon.io/competitions/official/236736/overview/description
- https://dacon.io/competitions/official/236736/overview/evaluation
- https://dacon.io/competitions/official/236736/overview/rules

## 모델링 규정

- 대회에서 제공한 train 데이터만 학습에 사용할 수 있습니다.
- eval 데이터는 학습, pseudo labeling 또는 validation에 사용할 수 없습니다.
- 원격 API 모델은 사용할 수 없습니다.
- 공식 가중치가 공개되고 허용 라이선스가 있는 사전학습 모델은 사용할 수
  있습니다. 사용 모델마다 이름, 버전, 원본 URL, 라이선스를 기록합니다.
- Python만 사용합니다.
- 단일 RTX PRO 6000 96GB 기준 학습은 최대 4일, 전체 eval 추론은 1시간
  이내여야 합니다.

## Submission Kit 격리

2026-07-24 운영진 공지:

https://dacon.io/competitions/official/236736/talkboard/417050

제출킷은 이미 확정된 MP4를 CSV로 변환하는 마지막 단계에만 사용합니다.
내부 코드·모델·checkpoint·실행 결과를 다음 용도로 사용하지 않습니다.

- 학습 loss 또는 auxiliary loss
- feature extractor 또는 local metric
- validation, ablation 또는 hyperparameter 선택
- 여러 생성 영상의 선택·reranking
- 영상 후처리 또는 영상 픽셀 수정

최종 추론 코드가 모든 MP4를 확정한 뒤 원본 제출킷을 수정 없이 실행합니다.
생성된 CSV도 수정하지 않습니다.

## 제출과 재현

- 팀당 하루 최대 3회 제출할 수 있습니다.
- 팀 구성 전 최소 1회 제출이 필요합니다.
- 2026-08-17 23:59까지 2~5인 팀 구성이 필수입니다.
- 대회 종료: 2026-08-20 18:00 KST
- 상위 팀 재현 코드 제출: 2026-08-21 14:00 KST
- 상대 경로, OS와 라이브러리 버전, 전처리·학습·추론 코드, 최종 weight가
  필요합니다.
- 최종 Private 채점 파일 1개를 선택해야 합니다. 새 제출을 하면 기존 선택이
  초기화될 수 있습니다.

## 내부 준수 체크

- [ ] train/eval 경로가 학습 코드에서 물리적으로 분리됨
- [ ] group holdout이 repo/dataset 단위임
- [ ] 사용한 사전학습 weight의 라이선스 기록
- [ ] 공식 제출킷이 학습 환경에서 import되지 않음
- [ ] 최종 MP4 생성 시 seed와 checkpoint 기록
- [ ] 전체 학습 4일 이내 로그
- [ ] 전체 추론 1시간 이내 로그
- [ ] 최종 MP4 216개가 각각 16 frames, 6 fps
- [ ] MP4 확정 후 원본 제출킷 1회 실행
- [ ] CSV 무수정 SHA-256 기록
