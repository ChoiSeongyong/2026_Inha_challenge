# DynamiCrafter candidate selection contract

`scripts/select_dynamicrafter_candidate.py` compares only paired JSON reports
created by `scripts/validate_dynamicrafter_plus.py` from the fixed train
holdout. It has no eval-data or submission-kit input.

## Required reports

Every candidate needs two otherwise identical runs:

1. `--action-control original`
2. `--action-control cross_clip`

The selector rejects a pair unless checkpoint, config/action contract, DDIM
settings, fold, exact selected samples, metric implementation, and runtime
environment match. The cross-clip report must use the next fixed selected
clip's action and must never reuse the recipient clip's action.

Across candidates, the following must remain identical:

- train/validation fold and repository sets
- full validation-dataset and selected-sample fingerprints
- dataset index, repository, episode, start frame, and metric resolution
- selection seed/strategy and metric contract
- GPU/CUDA/PyTorch environment
- original/cross-clip donor mapping

Checkpoint, training resolution, action representation, training sampling
strategy, update count, and DDIM settings may differ. These candidate-specific
fields are copied into the audit output.

## Selection

Example:

```bash
PYTHONPATH=src python scripts/select_dynamicrafter_candidate.py \
  --candidate raw6-320-step3000 \
    outputs/validation/raw6-320-step3000-original.json \
    outputs/validation/raw6-320-step3000-cross.json \
  --candidate kinematic18-384-step5000 \
    outputs/validation/kinematic18-384-step5000-original.json \
    outputs/validation/kinematic18-384-step5000-cross.json \
  --output-json outputs/validation/candidate-selection.json
```

Hard eligibility gates:

- mean `cross_clip - original` foreground L1 is strictly positive
- at least 50% of paired clips have a positive delta
- guarded 216-video runtime projection is at most 3,600 seconds

The runtime proxy is:

```text
300 second reserve
+ 1.25 × (
    observed data/model setup
    + 216 × observed variable seconds per holdout sample
  )
```

This is deliberately conservative but is not a production runtime
measurement. The selected candidate must still pass a complete 216-video dry
run including input scan, generation, restoration, MP4 encoding, hashing, and
provenance writing.

After the gates, candidates are ranked lexicographically. For each native
metric in the configured order, overall performance is compared first and the
repository worst quartile second. Lower is better except for SSIM and PSNR.
No metric weights are fitted or hand-tuned, and no composite pretending to
reproduce the hidden Dacon score is produced.

The output records source-report paths/checksums, cohort and pair-contract
fingerprints, every native ranking component, both hard gates, the runtime
projection, candidate metadata, and the selected candidate. If every candidate
fails a gate, `selected_candidate` is `null` and the CLI exits with status 2.
