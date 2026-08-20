# ABot-PhysWorld SO-100 implementation

This implementation uses the public Wan2.1-I2V-14B visual prior with the
ABot-PhysWorld VACE action adapter. The 14B DiT is frozen and only VACE is
updated, which is the only practical high-capacity path under the one-GPU,
48-hour budget.

The contest has six-dimensional joint actions, while the public ABot A2V
interface expects a spatial control video. `integrations/abot_physworld/
so100_action_map.py` provides the single train/inference contract:

- action row `t` controls generated frame `t+1`;
- frame 0 is blank because it is the observed initial image;
- 16 future actions become 17 VACE condition frames;
- train-fold median/IQR statistics are used for both train and eval;
- RGB maps encode a bounded pseudo-kinematic SO-100 chain, endpoint trail and
  gripper state without pretending that missing camera calibration exists.

The scripts are:

1. `prepare_abot_so100.py`: converts the cleaned manifest into ABot JSONL and
   an episode action cache.
2. `train_abot_so100_vace.py`: trains only VACE with a hard wall-clock budget,
   periodic checkpoint saves and resume support. A step cap is optional.
3. `prepare_abot_so100_eval.py`: creates the 216-sample eval JSONL.
4. `infer_abot_so100.py`: generates exactly 16-frame 480x640 MP4s and writes
   `inference_manifest.json` with authenticated wall-time fields.

The implementation deliberately does not use evaluation videos during
training and does not download external data. The public Wan/ABot weights are
still required locally before training or inference.
