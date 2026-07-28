# DynamiCrafter-plus held-out-train sampling validation

## Purpose and hard boundary

`scripts/validate_dynamicrafter_plus.py` measures a trained
DynamiCrafter-plus checkpoint on a fixed, explicitly bounded subset of
`ManifestSO100DataModule.val_dataset`.

The validator has three deliberate boundaries:

- It accepts no evaluation-data path and no submission-kit path.
- It reads only the configured training root,
  `${INHA_OPEN_ROOT}/data/train`, and refuses paths containing `eval`,
  `submission_kit`, or `official_submission_kit`.
- It keeps held-out target frames outside the model input. The official model
  receives the observed first frame plus black future placeholders, then the
  official `DDIMSampler` generates the future frames.

The submission kit is neither imported nor used for metrics. Metrics come from
`inha_worldmodel.metrics.reconstruction_metrics`.

## GPU command

Run this after installing the dependencies shipped with the official baseline
and making the baseline checkpoints available:

```bash
export INHA_PROJECT_ROOT=/workspace/Inha_challenge
export INHA_BASELINE_ROOT=/workspace/open/baseline
export INHA_OPEN_ROOT=/workspace/open

python scripts/validate_dynamicrafter_plus.py \
  --config configs/dynamicrafter_plus.yaml \
  --checkpoint outputs/dynamicrafter_plus/checkpoints/last.ckpt \
  --sample-limit 64 \
  --batch-size 2 \
  --ddim-steps 30 \
  --eta 0 \
  --timestep-spacing uniform_trailing \
  --action-control original \
  --seed 20260725 \
  --output-json outputs/dynamicrafter_plus/holdout_validation.json
```

CUDA is required for sampling. Official baseline imports are delayed so the
pure selection, aggregation, report, and CLI-contract tests can still run on a
CPU-only development machine.

`--sample-limit` is mandatory. The command rejects zero, a negative value, or
a limit larger than the held-out dataset. It does not silently fall back to a
full validation pass.

## Fixed sample protocol

The data module performs the repository or audited-fold holdout and fits action
normalization statistics from the training partition only. Before sampling,
the validator confirms that the train and validation repository sets are
disjoint.

The validation dataset is fixed at epoch 0 and loaded with zero workers so
window starts do not depend on worker assignment. Selection is deterministic:

1. Hash-order repositories using `--seed`.
2. Hash-order clips within each repository.
3. Select clips round-robin across repositories until `--sample-limit` is
   reached.

This reduces domination by a large repository while preserving an exact,
repeatable clip set. The JSON contains both a complete-fold fingerprint and a
selected-sample fingerprint. A changed manifest, fold, seed, or dataset
identity therefore produces a different audit trail.

The validator refuses `training_scope=all_clean`: once the selection fold has
been consumed by the final refit, it is no longer a legal held-out checkpoint
selection set.

## Metrics and tail reporting

Each generated clip is compared with its held-out target in `[0, 1]` after the
observed first frame is hard-preserved, matching production output behavior.
Before metrics, both prediction and target are unpadded and restored to the
original train-video resolution. This makes 320×512, 384×512, and 480×640
variants comparable in a common image space and prevents black padding from
artificially improving the score.
The report stores these native reconstruction metrics:

- `l1`, `psnr`, and `ssim`
- `edge_l1` and `temporal_l1`
- `motion_amplitude_error`
- `foreground_l1` and `background_l1`
- `first_frame_l1` and `moving_fraction`

The same metrics are aggregated overall and per repository. The report also
lists the worst repository quartile, ranked by `foreground_l1` by default.
Use `--ranking-metric` to choose another supported lower-is-better metric.
For fewer than four represented repositories, the aggregation still includes
at least the single worst repository.

The metrics are diagnostic proxies for checkpoint selection. They are not a
claim about the hidden competition score.

## Action-conditioning gate

For every serious checkpoint, generate a second report on the exact same
selected clips with distribution-matched actions taken from the next fixed
clip:

```bash
python scripts/validate_dynamicrafter_plus.py \
  --config configs/dynamicrafter_plus.yaml \
  --checkpoint outputs/dynamicrafter_plus/checkpoints/last.ckpt \
  --sample-limit 64 \
  --action-control cross_clip \
  --output-json outputs/dynamicrafter_plus/holdout_cross_clip.json

python scripts/compare_dynamicrafter_action_control.py \
  --original outputs/dynamicrafter_plus/holdout_validation.json \
  --cross-clip outputs/dynamicrafter_plus/holdout_cross_clip.json
```

The comparison rejects changed checkpoints, DDIM settings, contracts, folds,
or sample identities. `mean_control_minus_original > 0` is the minimum gate:
wrong actions must make the same held-out targets harder to reconstruct.

## JSON provenance

The output is written atomically and is not overwritten unless `--overwrite`
is supplied. It includes:

- hashes and byte sizes for the tested checkpoint, ordered configs, manifest,
  action statistics, and validation script;
- hashes of configured source checkpoints when those files exist;
- the action-statistics training-fold fingerprint;
- train and validation repository IDs, dataset size, fold fingerprint, and
  selection fingerprint;
- seed, dataset epoch, action alignment, DDIM steps, eta, guidance settings,
  timestep spacing, AMP dtype, and batch size;
- per-sample repository, episode, start index, dataset index, and metrics;
- data setup, model setup, sampling, metric, total, and per-sample runtimes;
- GPU name, peak allocated CUDA memory, PyTorch version, and CUDA version.

The report explicitly records `validation_scope: held_out_train_only` and
`submission_kit_used: false`. Incomplete runs do not produce a valid report:
the generated sample identities and count must exactly match the requested
selection.

## Runtime interpretation

The validation sample cap controls model-development cost. It is separate from
the competition's one-hour full evaluation inference limit. A 64-clip
validation runtime neither proves nor disproves compliance with that limit.
Benchmark the final inference program independently on the target GPU and the
full permitted evaluation workload.

For checkpoint screening, start with 32 or 64 fixed clips. Re-run the most
promising checkpoints on a larger fixed cap, such as 128, if GPU time permits.
Keep the seed, DDIM parameters, batch size, and fold fingerprint identical when
comparing checkpoints.

## CPU-only verification

The local repository can verify the non-GPU contract with:

```bash
python -m pytest tests/test_dynamicrafter_validation.py -q
python -m py_compile \
  src/inha_worldmodel/dynamicrafter_validation.py \
  scripts/validate_dynamicrafter_plus.py \
  tests/test_dynamicrafter_validation.py
python scripts/validate_dynamicrafter_plus.py --help
```

These checks do not run the official model or produce performance numbers.
Actual reconstruction metrics require the official dependencies, checkpoint,
training data, and CUDA GPU.

Fine-tuned checkpoints must contain a complete main action UNet, complete EMA,
and an embedded contract matching alignment, stats/fold fingerprint, fold ID,
manifest/fold/config hashes. The trusted provided 1,500-step baseline is the
only intended use of `--allow-legacy-checkpoint`; the flag cannot bypass an
existing contract mismatch.
