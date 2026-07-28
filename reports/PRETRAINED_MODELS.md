# 사전학습 모델·가중치 레지스트리

확인 시각: 2026-07-25 KST

이 파일은 실제 학습/추론에 사용한 공개 가중치의 출처, 사용 조건,
checksum을 고정한다. 다운로드 뒤 checksum을 채우지 않은 weight는 최종
학습에 사용하지 않는다.

## 대회 제공 action-conditioned DynamiCrafter

- 용도: 1순위 빠른 fine-tuning baseline
- 대회 제공 파일: `baseline/checkpoints/baseline_diffusion.ckpt`
- 학습 상태: epoch 9, global step 1,500
- 무결성 audit: main action UNet 1,107 tensors + EMA 1,109 tensors,
  expected shape 기준 완전
- 이 파일에는 optimizer/scheduler state가 없으므로 fine-tuning 초기화에만
  사용하고 exact training resume에는 사용하지 않음
- 18D action ablation은 `action_embed.0.weight`의 main/EMA 두 tensor만
  6→18로 확장하고 새 열을 0으로 채워 초기 함수를 정확히 보존함
- 로컬 SHA-256:
  `c66a22652e37001aa6ee5e21c874b0ad67acad707b01a4b9ace8cf584a2517c5`
- backbone 원본:
  `https://huggingface.co/Doubiiu/DynamiCrafter_512/resolve/main/model.ckpt`
- 공식 model card:
  `https://huggingface.co/Doubiiu/DynamiCrafter_512`
- 코드 라이선스: 번들된 Apache-2.0
- weight 사용 조건: model card상 personal/research/non-commercial
- 상태: 대회가 backbone 다운로드 코드와 fine-tuned checkpoint를 직접
  제공했으므로 이 대회의 연구 목적 사용 경로로 채택한다. 상업적 재사용은
  별도 검토가 필요하다.
- backbone 다운로드 후 기록할 항목:
  - [ ] 실제 URL/revision
  - [ ] 파일 크기
  - [ ] SHA-256
  - [ ] 다운로드 시각

## NVIDIA Cosmos-Predict2.5 2B Robot / Action-Cond

- 용도: 2순위 고성능 world-foundation-model post-training, 이후 DMD2 4-step
- upstream repository:
  `https://github.com/nvidia-cosmos/cosmos-predict2.5`
- 검토한 commit:
  `a2c298b0a3df3778b973fe65e9e58877b292d8a7`
- model card:
  `https://huggingface.co/nvidia/Cosmos-Predict2.5-2B`
- model license: NVIDIA Open Model License
- code license: Apache-2.0 및 repository 내 third-party notices
- 상태: adapter만 구현. weight 다운로드·학습·추론은 아직 하지 않음.
- 접근 전 필요한 조치:
  - [ ] Hugging Face gated-model 약관을 사용자가 직접 수락
  - [ ] 정확한 action-conditioned checkpoint UUID/revision 고정
  - [ ] 모든 다운로드 파일 SHA-256 기록
  - [ ] `Built on NVIDIA Cosmos` 등 license attribution을 최종 코드에 포함

## 금지된 사용

- 원격 inference API를 호출하지 않는다.
- 출처·공개 여부·license가 불명확한 checkpoint를 사용하지 않는다.
- submission kit의 모델/checkpoint는 이 레지스트리에 추가하지 않으며,
  최종 MP4의 CSV 변환 이외 용도로 절대 사용하지 않는다.
