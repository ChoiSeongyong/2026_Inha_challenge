# 초기 데이터·환경 감사

확인 시각: 2026-07-25 KST

## 데이터

- 원본 압축: `/Users/choeseong-yong/Downloads/open.zip` (약 9.2GB)
- 압축 해제 데이터: `/Users/choeseong-yong/Downloads/open`
- train 데이터셋: 128개 LeRobot 형식 데이터셋
- train episode/video/parquet: 11,132개
- train frame: 1,025,666개
- eval sample: 216개
- eval image: RGB 640×480
- eval action: `(16, 6)`
- train 대부분과 eval 출력 FPS: 6

eval action sequence와 train action window의 bitwise exact match를 전체
1,025,666 frame에서 검사했으며 exact match는 없었습니다.

## 중요한 분포 특성

eval 영상 도메인은 시각적으로 두 그룹입니다.

- sample 000000–000153: 회색 바닥, 고정 상단 카메라, 주황/흰색 로봇
- sample 000154–000215: 나무 테이블, 고정 상단 카메라, 검정 로봇

두 그룹의 action offset이 크게 다릅니다. 전역 절대 action z-score만 쓰는
공식 baseline보다 다음 표현이 도메인 전이에 유리할 가능성이 큽니다.

- `action[t] - action[0]`
- 1차·2차 temporal difference
- train-only robust scale
- absolute action과 relative action을 함께 쓰되 별도 projection

eval은 추론 입력으로만 사용하며 위 관찰을 label, pseudo target 또는
validation으로 사용하지 않습니다.

## 로컬 컴퓨팅

- Apple M3, 8 CPU cores
- unified memory 16GB
- NVIDIA GPU 없음
- 저장 공간 약 145GB 여유

로컬에서는 데이터 파이프라인, 단위 테스트, 축소 모델 스모크 테스트만
수행합니다. 전체 수상권 학습과 1시간 추론 검증에는 단일 96GB GPU 환경이
필요합니다.

## 1차 모델 방향

1. repo/dataset 단위 leave-domain-out 검증
2. initial image를 보존하는 action-conditioned dense flow/occlusion/residual 모델
3. train 데이터만으로 image reconstruction, temporal consistency,
   edge/structure, flow smoothness loss 최적화
4. 규정상 허용되는 공개 사전학습 visual backbone을 선택적으로 사용
5. diffusion 계열은 별도 강한 후보로 학습하되 train OOF 지표만으로 비교
