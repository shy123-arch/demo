---
license: mit
tags:
  - robotics
  - humanoid
  - whole-body-control
  - motion-tracking
  - behavior-foundation-model
  - pytorch
  - isaac-lab
  - arxiv:2607.15163
pipeline_tag: reinforcement-learning
---

# Scaling Behavior Foundation Models for Humanoid Robots


Official release of relevant resources for the paper **Scaling Behavior Foundation Model for Humanoid Robots**.

[![Project Page](https://img.shields.io/badge/Project-Website-4285F4?logo=googlechrome&logoColor=white)](https://scalebfm.github.io/)
[![Paper](https://img.shields.io/badge/arXiv-2607.15163-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.15163)
[![Code](https://img.shields.io/badge/GitHub-Code-181717?logo=github&logoColor=white)](https://github.com/zengweishuai/ScaleBFM)
[![Issues](https://img.shields.io/badge/GitHub-Issues-181717?logo=github&logoColor=white)](https://github.com/zengweishuai/ScaleBFM/issues)

<p align="center">
  <img src="assets/scalebfm_teaser_figure.png" alt="ScaleBFM capabilities: dexterous manipulation, natural and agile locomotion, and whole-body coordinated loco-manipulation" width="100%">
</p>


ScaleBFM studies how Behavior Foundation Models (BFMs) for humanoid control can be scaled through the coordinated design of three components:

1. a unified whole-body motion-tracking objective that reformulates diverse humanoid control tasks as the reproduction of integrated whole-body trajectories in the global frame;
2. the strategic coordination of training-data quantity and diversity to enable effective scaling; and
3. an expressive and scalable model architecture, termed **Humanoid Transformer**, designed to learn structured behavioral representations.

The resulting controller supports diverse whole-body behaviors, including dexterous manipulation, natural and agile locomotion, and coordinated loco-manipulation.

## Download

Install the Hugging Face command-line client:

```bash
pip install -U huggingface_hub
```

Download the complete repository:

```bash
hf download WeishuaiZeng/ScaleBFM --local-dir ScaleBFM
```

Download the checkpoints:

```bash
hf download WeishuaiZeng/ScaleBFM \
  --include "checkpoint/**" \
  --local-dir ScaleBFM
```

Download the compiled checkpoints:

```bash
hf download WeishuaiZeng/ScaleBFM \
  --include "compiled_checkpoint/**" \
  --local-dir ScaleBFM
```

Download the packaged ScaleBridge environment:

```bash
hf download WeishuaiZeng/ScaleBFM \
  --include "environment/**" \
  --local-dir ScaleBFM
```

Download only the test data:

```bash
hf download WeishuaiZeng/ScaleBFM \
  --include "test_set/**" \
  --local-dir ScaleBFM
```

## Usage

First, clone the [ScaleBFM code repository](https://github.com/zengweishuai/ScaleBFM), which contains ScaleRetarget, ScaleTrack, and ScaleBridge:

```bash
git clone https://github.com/zengweishuai/ScaleBFM.git
```

Then follow the instructions for the component that matches your task:

1. **Motion Retargeting:** Follow the [ScaleRetarget instructions](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleRetarget) to prepare human motion data and retarget it to the robot embodiment.
2. **Training and Playing policies:** Follow the [ScaleTrack instructions](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleTrack) to train policies or run the released checkpoints.
3. **Deployment:** Follow the [ScaleBridge instructions](https://github.com/zengweishuai/ScaleBFM/tree/main/ScaleBridge) to deploy trained policies.


## Repository contents

```text
ScaleBFM/
├── assets/
│   ├── scalebfm_teaser_figure.png
├── checkpoint/
│   ├── humanoid_transformer_m/
│   │   └── model_22200.pt
│   └── humanoid_transformer_xl/
│       └── model_22200.pt
├── compiled_checkpoint/
│   ├── humanoid_transformer_m/
│   │   ├── aarch64/
│   │   │   ├── model_22200_tensorrt.pt
│   │   │   └── model_22200_tensorrt_metadata.json
│   │   └── linux/
│   │       ├── mode_table.pt
│   │       ├── model_22200_tensorrt.pt
│   │       └── model_22200_tensorrt_metadata.json
│   └── humanoid_transformer_xl/
│       ├── aarch64/
│       │   ├── model_22200_tensorrt.pt
│       │   └── model_22200_tensorrt_metadata.json
│       └── linux/
│           ├── mode_table.pt
│           ├── model_22200_tensorrt.pt
│           └── model_22200_tensorrt_metadata.json
├── environment/
│   └── scalebridge.tar.gz
└── test_set/
    ├── BONES_Test_Set/
    │   └── BONES_Test_Set_processed.zip
    └── Ours_Test_Set/
        ├── Ours_Test_Set_orig_bvh.zip
        ├── Ours_Test_Set_retargeted_pkl.zip
        └── Ours_Test_Set_processed.zip
```


## Citation

If you find this work useful, please cite:

```bibtex
@article{zeng2026scaling,
  title   = {Scaling Behavior Foundation Model for Humanoid Robots},
  author  = {Zeng, Weishuai and Yin, Kangning and Niu, Xiaojie and
             Lu, Shunlin and Zhong, Weixiang and Chen, Jiahe and
             Jia, Feiyu and Chen, Xiao and Wang, Zirui and Xu, Furui and
             Zhou, Ming and Li, Kailin and Zhang, Weinan and Wang, He and
             Yi, Li and Lin, Dahua and Pang, Jiangmiao and Wang, Jingbo},
  journal = {arXiv preprint arXiv:2607.15163},
  year    = {2026}
}
```

## License

The ScaleBFM release is provided under the license identified in this repository. Third-party datasets and derived artifacts remain subject to their respective source licenses and terms.

Specifically, `test_set/BONES_Test_Set/BONES_Test_Set_processed.zip` is derived from [BONES-SEED](https://huggingface.co/datasets/bones-studio/seed) and is governed by the [BONES Motion Capture Dataset License Agreement](https://bones.studio/info/seed-license), not this repository's MIT license. Before downloading or using this file, you must review and agree to the BONES license. If you do not agree to its terms or do not meet its eligibility requirements, do not download or use the BONES-derived test data. By downloading or using the processed test set, you acknowledge that you have reviewed and agreed to the applicable BONES terms, including its use restrictions and attribution requirements.


## Acknowledgements

We thank [Bones Studio](https://bones.studio/) for providing motion data used in the test set. The use of the underlying dataset is subject to the BONES Motion Capture Dataset License Agreement.
