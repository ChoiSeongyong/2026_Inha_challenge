# 학습 데이터 전수 감사와 제외 정책

확인 시각: 2026-07-25 KST

이 문서는 `/Users/choeseong-yong/Downloads/open`을 읽기 전용으로 검사한
결과입니다. 공식 제출킷은 import하거나 실행하지 않았습니다.

## 원본 식별과 무결성

`/Users/choeseong-yong/Downloads/open.zip`이 대회 데이터임을 다음 근거로
확인했습니다.

- DACON 데이터 페이지가 배포 파일명을 `open.zip`으로 명시합니다.
- 공식 페이지의 `data/train`, `data/eval/{images,actions}`,
  `submission_kit`, `baseline` 구조가 압축 내부와 정확히 같습니다.
- 루트 README 제목이 `INHA AI Challenge`이며 문제 설명도 현재 이미지와
  Action Sequence를 이용한 미래 영상 생성으로 일치합니다.

공식 데이터 페이지:

https://dacon.io/competitions/official/236736/data

무결성 수치는 다음과 같습니다.

- 압축 파일: 9,209,529,683 bytes, 24,016 entries
- `zipinfo -t` 검사: 오류 없음
- 압축 해제 파일: 23,027 files, 9,290,445,710 bytes
- 압축의 파일 수·비압축 크기와 실제 압축 해제 결과가 정확히 일치
- MP4 11,132개, Parquet 11,132개, eval PNG 216개, eval NPY 216개

따라서 `/Downloads/open`은 완전히 압축 해제된 상태입니다.

## Train 요약

- owner 55개, LeRobot repository 128개
- episode/Parquet/MP4 각각 11,132개
- 총 1,025,666 frames
- 고유 task 문자열 101개
- action과 observation state는 모두 6차원
- 127 repositories는 6 FPS, `dragon-95/so100_sorting`만 10 FPS
- 해상도: 480×640 123 repos, 720×1280 2 repos, 1080×1920 3 repos
- codec: AV1 81 repos, H264 47 repos

Episode 길이 통계:

| 통계 | frames |
|---|---:|
| 최소 | 1 |
| 1% | 29 |
| 10% | 49 |
| 25% | 60 |
| 중앙값 | 80 |
| 75% | 108 |
| 90% | 127 |
| 99% | 382.21 |
| 최대 | 1,072 |
| 평균 | 92.137 |

16 frames 미만 episode는 34개, 총 109 frames이며 이 중 27개가 1-frame
episode입니다. 16-frame 모델 학습에서는 사용할 수 없습니다.

## Parquet 전수 검사

11,132개 Parquet의 1,025,666 rows를 모두 읽어 검사했습니다.

실제 핵심 스키마:

```text
action: list<float>, 모든 row 길이 6
observation.state: list<float>, 모든 row 길이 6
timestamp: double
frame_index: int64
episode_index: int64
index: int64
task_index: int64
```

- 11,006 files: 위 핵심 7개 열
- 96 files: `grid_position: list<int64>[2]` 추가
- 30 files: `next.reward: int64`, `next.done: bool` 추가
- 전부 Parquet 2.6, Snappy, 1 row group
- 모든 열과 list element의 null 수 0
- action/state/timestamp의 NaN·Inf 수 0
- 파일 내부 완전 중복 row 수 0
- 모든 `frame_index`가 `0..length-1`
- 파일 row 수와 `episodes.jsonl`의 length가 전부 일치

메타 선언과 실제 물리 타입이 다른 예외가 있습니다.

- `timestamp`: `info.json`은 float32, 실제 Parquet은 double
- `lirislab/guess_who_so100`의 `grid_position`: 메타는 float32, 실제는 int64

로더는 메타 dtype을 강제하지 말고 실제 Parquet 값을 명시적으로 cast해야
합니다.

전체 action 통계:

```text
mean = [2.960132, 117.618275, 109.812742, 61.608533, -29.527531, 9.775668]
std  = [26.418274,  50.195746,  46.623486, 29.761411,  62.539355, 16.391966]
min  = [-122.607422, -15.820312, -269.296875, -112.851562, -253.125, -8.362370]
max  = [125.244141, 210.673828, 190.458984, 122.255859, 243.808594, 119.407890]
```

제공된 `so100_action_statistics.json`의 count 974,661은 오류가 아닙니다.
공식 baseline이 16 frames 미만을 제거한 뒤 seed 0으로 episode를 섞어 5%
validation 554개/50,896 rows를 제외한 10,544 train episodes의 row 수와
정확히 일치합니다. 다만 이 random episode split은 아래 중복이 train/val에
나뉠 수 있으므로 신뢰할 수 있는 검증 분할로 사용하면 안 됩니다.

## 중복과 메타 오류

### Exact Parquet/action 중복

정확히 같은 Parquet 및 action sequence의 excess episode는 423개입니다.

| repository pair | 중복 episodes |
|---|---:|
| `Chojins/chess_game_001_blue_stereo` ↔ `bensprenger/chess_game_001_blue_stereo` | 306 |
| `Ityl/so100_recording2` ↔ `lirislab/red_cube_into_blue_cube` | 45 |
| `Beegbrain/pick_lemon_and_drop_in_bowl` ↔ `lirislab/lemon_into_bowl` | 40 |
| `Beegbrain/sweep_tissue_cube` ↔ `lirislab/sweep_tissue_cube` | 30 |
| `pranavsaroha/so100_legos4` 내부 | 2 |

영상 bytes가 다르더라도 동일 trajectory가 train과 validation 양쪽으로
갈 수 있으므로 위 repository pair는 반드시 하나의 validation group으로
묶습니다. 생성된 manifest의 `validation_group`이 이를 반영합니다.

### Exact video와 완전 중복

- exact MP4 duplicate groups: 48
- duplicate MP4 excess files: 92
- action과 MP4가 모두 같은 exact content duplicate: 47쌍
  - Ityl ↔ lirislab 45쌍
  - pranavsaroha 내부 2쌍

각 exact content group은 정렬상 첫 episode만 canonical로 유지하고 나머지는
`exact_content_duplicate_noncanonical`로 제외합니다.

### 심각한 action-video 불일치

`Gano007/so100_medic`의 50 episodes 중 46개 MP4가 byte-for-byte 동일한
72-frame 영상입니다. 반면 이 46개의 `(72, 6)` action sequence는 모두
서로 다릅니다. 동일한 미래 영상에 서로 다른 행동 조건이 붙은 강한 label
noise이므로 4개의 나머지 episode까지 포함해 repository 전체 50개를
제외합니다.

### Episode ID 오류

`pranavsaroha/so100_legos4`:

- `episode_000021.parquet` 내부 `episode_index`가 19
- `episode_000043.parquet` 내부 `episode_index`가 42
- 두 파일은 각각 episode 19, 42의 Parquet와 MP4 복사본
- repository 내부 `index` 658 rows가 중복

또한 아래 메타에는 index gap이 있습니다.

- `bensprenger/chess_game_001_blue_stereo`: 306 없음
- `bensprenger/right_arm_p_brick_in_box_with_y_noise_v0`: 4, 11, 27 없음
- `roboticshack/left-arm-grasp-lego-brick`: 49 없음
- `roboticshack/team-7-left-arm-grasp-motor`: 63 없음

따라서 episode 파일을 `range(total_episodes)`로 만들면 안 되며 반드시
`episodes.jsonl`에 실제로 나열된 index를 사용해야 합니다.

## 기본 제외 권고

`scripts/build_manifest.py`의 기본 정책은 다음 130 episodes를 제외하고
11,002 episodes/1,018,554 action rows를 유지합니다.

| 이유 | 표시 횟수 |
|---|---:|
| `repository_video_action_conflict` | 50 |
| `exact_content_duplicate_noncanonical` | 47 |
| `too_short_for_sequence` | 34 |
| `embedded_episode_index_mismatch` | 2 |

이유 사이 중복이 있어 고유 제외 episode 수는 130입니다. 구체적으로:

- `Gano007/so100_medic` 전체 50개 제외
- exact content 중복 47개에서 non-canonical 사본 제외
- 16 frames 미만 34개 제외
- pranavsaroha 내부 ID mismatch 2개는 이미 exact duplicate 제외와 겹침

이 정책은 명백한 오류만 제거합니다. Parquet/action만 같고 영상 bytes가
다른 376개의 cross-repository 복사 trajectory는 일단 학습에 남기되,
반드시 동일 validation group에 배치합니다. 추후 decoded-frame 유사도와
OOF 실험으로 중복 영상을 확인한 뒤 학습 weight를 줄이거나 한쪽 repository를
제거할 수 있습니다.

## Eval과 제출 구조

- eval ID: `sample_000000`–`sample_000215`, 216개 연속·유일
- PNG: 모두 RGB 480×640, exact duplicate 없음
- action: 모두 float32 `(16, 6)`, NaN·Inf 없음, exact duplicate 없음
- 별도 target CSV는 없음
- 각 초기 PNG와 16-step action으로 16-frame MP4를 생성

`submission_kit/sample_submission.csv`:

- shape `(648, 3)`
- 열: `sample_id`, `feature_component`, `feature_json`
- ID마다 3 rows
- Video Feature `(512,)`, DINO `(16, 384)`, Action `(1, 1)`
- missing/malformed JSON/duplicate sample-component key 없음

루트 README는 root의 `sample_submission.csv`를 언급하지만 실제 파일은
`submission_kit/sample_submission.csv`에만 있습니다.

## Manifest 생성

```bash
python scripts/build_manifest.py \
  --train-root data/train \
  --output-dir artifacts/manifests
```

PyArrow가 필요합니다. 생성물:

- `artifacts/manifests/train_episodes.jsonl`
- `artifacts/manifests/duplicate_groups.json`
- `artifacts/manifests/action_qc.json`

학습 코드는 `include_for_training=true`만 선택하고, 분할 시
`validation_group` 전체를 한쪽 fold에 두어야 합니다. 원본 파일은 수정하거나
삭제하지 않습니다.
