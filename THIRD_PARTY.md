# Third-party software and model notice

This source-only project integrates with third-party repositories and model
weights that are not redistributed here.

| Component | Upstream | Pinned revision / identifier | Terms |
|---|---|---|---|
| ABot-PhysWorld | https://github.com/amap-cvlab/ABot-PhysWorld | `7d47080ea122346e6b7c1cb37c2a8d43730f624c` | Check the upstream repository and ModelScope model page |
| Wan2.1 | https://github.com/Wan-Video/Wan2.1 | `Wan-AI/Wan2.1-I2V-14B-480P`, `Wan-AI/Wan2.1-T2V-1.3B` | Apache-2.0 files were present in the inspected model distribution; verify the selected revision |
| Cosmos-Predict2.5 | https://github.com/nvidia-cosmos/cosmos-predict2.5 | `a2c298b0a3df3778b973fe65e9e58877b292d8a7` | Source: Apache-2.0; model weights: NVIDIA Open Model License |
| DynamiCrafter | https://github.com/Doubiiu/DynamiCrafter | `Doubiiu/DynamiCrafter_512` | Check the model card; the inspected weights carry separate usage conditions |

The files under `patches/` contain only local modifications against the pinned
upstream source revisions. Applying a patch does not replace the upstream
license, copyright notice, or third-party notices.

Competition data, official baseline files, the submission kit, model weights,
fine-tuned checkpoints, generated videos, and submission CSV files are excluded
from this repository.
