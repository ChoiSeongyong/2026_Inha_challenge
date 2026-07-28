# 누수 없는 다중 검증 fold

확인 시각: 2026-07-25 KST

## 목적

영상 world model의 random episode split은 같은 owner의 카메라·배경·로봇
설정과 중복 trajectory가 train/validation 양쪽에 들어가 성능을 크게
부풀릴 수 있습니다. 이 프로젝트의 검증 fold는 train manifest의 다음 두
관계를 동시에 보존합니다.

- `owner`: 같은 수집자/환경 계열을 분리하지 않음
- `validation_group`: exact 또는 cross-repository duplicate 관계로 연결된
  repository를 분리하지 않음

평가 데이터와 submission kit는 fold 생성·감사 과정에서 읽지 않습니다.

## 구현

- 핵심 API: `src/inha_worldmodel/validation.py`
- CLI: `scripts/build_folds.py`
- 단위 테스트: `tests/test_validation.py`
- 실제 생성물: `artifacts/folds/folds.json`

실행:

```bash
python scripts/build_folds.py
```

기본 설정:

```text
manifest: artifacts/manifests/train_episodes.jsonl
output: artifacts/folds/folds.json
seeded holdout seeds: 17, 29, 43, 71, 101
seeded validation target: episode 기준 20%
leave-owner-out small-owner bundle 기준: 약 200 episodes
owner bundle seed: 0
```

Frame 수를 기준으로 seeded holdout을 맞추려면 다음과 같이 실행합니다.

```bash
python scripts/build_folds.py --balance-by frames
```

## 누수 단위

Owner node와 validation-group node로 bipartite graph를 만들고, 각 episode가
속한 owner와 group을 edge로 연결합니다. Graph의 connected component
전체가 하나의 `leakage-unit`입니다.

```text
owner A ─ group 1
   │
   └──── group 2

owner B ─ shared duplicate group ─ owner C
```

위 예시에서 owner A의 두 group은 같은 unit이며, owner B와 C도 shared
group 때문에 같은 unit입니다. 이 unit보다 작은 단위로는 어떤 fold도
나누지 않습니다. 따라서 다음 세 조건이 구조적으로 보장됩니다.

```text
train_episode_keys ∩ validation_episode_keys = ∅
train_owners ∩ validation_owners = ∅
train_validation_groups ∩ validation_validation_groups = ∅
```

Repository overlap도 추가로 0인지 검사합니다.

## Fold A: 반복 seeded group holdout

각 seed마다 leakage unit을 hash 기반으로 초기 선택한 뒤, target episode
또는 frame 비율과의 차이를 줄이는 deterministic one-toggle/one-swap
optimization을 적용합니다.

이 방식의 목적:

- 같은 seed와 manifest SHA-256에서 완전히 재현 가능
- owner와 duplicate group을 절대 분할하지 않음
- 단순 hash threshold보다 validation 크기를 target에 가깝게 맞춤
- 서로 다른 seed의 domain 조합에서 model ranking 안정성을 확인

현재 실제 manifest에서는 다섯 fold 모두 정확히 2,200 / 11,002 episodes,
약 20.0%를 validation으로 선택했습니다.

| Fold | Validation episodes | Frames | Frame 비율 | Owners | Groups |
|---|---:|---:|---:|---:|---:|
| seed 17 | 2,200 | 184,974 | 18.16% | 11 | 31 |
| seed 29 | 2,200 | 198,258 | 19.46% | 13 | 29 |
| seed 43 | 2,200 | 189,634 | 18.62% | 6 | 11 |
| seed 71 | 2,200 | 184,835 | 18.15% | 6 | 11 |
| seed 101 | 2,200 | 231,368 | 22.72% | 10 | 24 |

Episode 기준 20%를 정확히 맞춘 대신 frame 비율은 episode 길이 차이 때문에
달라집니다. Frame-balanced 실험이 필요하면 별도 artifact를 다른 출력
경로에 생성해 비교해야 합니다.

## Fold B: bundled leave-owner-out

각 leakage unit을 한 번씩 validation에 두는 leave-owner-out 검증입니다.
작은 owner를 하나씩 검증하면 sample 수가 너무 작아 metric variance가
커지므로 약 200 episodes가 되도록 작은 unit을 deterministic
load-balancing으로 묶습니다.

- 큰 owner/component는 단독 fold
- 작은 owner/component는 2~3 owner bundle
- duplicate group으로 연결된 owner들은 크기와 관계없이 같은 fold
- 모든 included episode는 전체 leave-owner-out fold를 통틀어 정확히 한
  번 validation에 등장

실제 결과:

```text
leave-owner-out folds: 30
validation episode 범위: 198–2,007
validation owner 수 범위: 1–3
validation frame 범위: 12,938–170,122
missing validation episodes across folds: 0
repeated validation episodes across folds: 0
```

약 200은 hard minimum이 아니라 bundle balancing target입니다. 가장 큰
연결 성분은 다른 unit과 섞지 않고 단독 fold로 둡니다.

## 실제 manifest 감사 결과

Source:

```text
path:
  artifacts/manifests/train_episodes.jsonl
SHA-256:
  f9129309cb897f68ef28a933ee53602cc3dbfe4a79def036dc630f7d0f29cc37
total records: 11,132
included episodes: 11,002
excluded episodes: 130
owners represented: 54
leakage units: 52
```

Owner 수보다 unit 수가 적은 것은 cross-owner duplicate group 때문입니다.
실제 포함 episode에서 확인된 큰 cross-owner 연결 성분:

- `Beegbrain + lirislab`: 584 episodes, 14 validation groups
- `Chojins + bensprenger`: 2,007 episodes, 7 validation groups

두 연결 성분은 어느 fold에서도 분리되지 않습니다.

최종 artifact 감사:

```text
total folds: 35
seeded group folds: 5
unique seeded validation sets: 5
leave-owner-out folds: 30
unique fold IDs: 35
all per-fold audits passed: true
all leave-owner-out episodes validated exactly once: true
overall audit passed: true
```

35개 모든 fold에서 다음 값이 0입니다.

```text
duplicate train episode keys
duplicate validation episode keys
train/validation episode overlap
train/validation owner overlap
train/validation validation_group overlap
train/validation repository overlap
missing episodes
unknown episodes
```

## JSON 구조와 사용

`folds.json`은 source manifest SHA-256, 설정, leakage unit, 명시적
train/validation episode key, owner, repository, validation group,
count/fraction, per-fold audit를 모두 포함합니다.

학습 실행은 다음 원칙을 따라야 합니다.

1. `source_manifest.sha256`이 현재 manifest와 일치하는지 확인합니다.
2. 선택한 `fold_id`의 `train_episode_keys`만 학습에 사용합니다.
3. Action/state normalization, calibration cluster, router, proxy metric,
   pretrained-cache statistics를 train side에서만 fit합니다.
4. `validation_episode_keys`는 model update나 early-training statistics에
   사용하지 않습니다.
5. 실험 로그에 fold ID와 manifest SHA-256을 저장합니다.
6. `fold.audit.passed`가 false인 artifact로는 학습을 시작하지 않습니다.

권장 사용:

- 빠른 iteration: seeded fold 17 하나
- architecture 결정: seeded 5-fold 평균과 worst fold
- camera/owner 일반화 감사: leave-owner-out 중 대표적인 large, medium,
  bundled-small owner fold
- 최종 후보: 가능한 범위에서 전체 leave-owner-out OOF 또는 owner 규모별
  stratified subset

Private score를 선택하는 기준은 Public 30%가 아니라 이 grouped OOF의
평균, worst-domain, bottom-quartile 안정성이어야 합니다.
