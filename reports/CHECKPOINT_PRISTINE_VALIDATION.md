# Provided-checkpoint pristine validation

The bundled official loader sorts repositories, removes examples shorter than
16 frames, shuffles with `random.Random(0)`, and assigns the first 5% to
validation. Reproducing that contract gives:

- 11,098 eligible official examples
- 554 episodes not used to train the provided 1,500-step checkpoint
- 6 of those 554 rejected by the independent manifest quality policy
- 548 usable checkpoint-pristine validation episodes
- 10,454 continuation-training episodes
- zero train/validation episode overlap and complete coverage of all 11,002
  manifest-approved episodes

The committed artifact is
`artifacts/folds/official_baseline_seed0_pristine.json`. It binds the exact
manifest, train metadata, bundled config/loader/data-module files, and provided
checkpoint by SHA-256. It records that neither evaluation data nor the
submission kit was used.

Rebuild and re-audit:

```bash
PYTHONPATH=src python scripts/build_checkpoint_pristine_split.py
```

Rebuild its train-only action normalization:

```bash
PYTHONPATH=src python scripts/prepare_dynamicrafter_stats.py \
  --train-root data/train \
  --manifest artifacts/manifests/train_episodes.jsonl \
  --fold-artifact artifacts/folds/official_baseline_seed0_pristine.json \
  --fold-id official_baseline_seed0_validation \
  --validation-protocol official_checkpoint_pristine \
  --output artifacts/stats/dynamicrafter_action_stats_checkpoint_pristine.json
```

For training and validation, merge configs in this order:

```text
configs/dynamicrafter_plus.yaml
configs/dynamicrafter_checkpoint_pristine.yaml
<resolution/action/sampling candidate overlays>
```

This is a checkpoint-pristine **episode** split, not an owner/repository
generalization split: 117 repositories have episodes in both partitions. Use it
as the cleanest measure of improvement over the provided checkpoint, and keep
the existing owner-disjoint fold 17 as a secondary robustness check.
