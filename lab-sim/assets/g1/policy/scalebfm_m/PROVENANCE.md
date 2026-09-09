# ScaleBFM M assets

- Official code: https://github.com/zengweishuai/ScaleBFM
- Code revision: abd6f17c02fe0baabc14709feb8d9ea4959aa621
- Weights: https://huggingface.co/WeishuaiZeng/ScaleBFM/tree/main/checkpoint/humanoid_transformer_m
- Metadata and mode table: compiled_checkpoint/humanoid_transformer_m/linux in the same model repository.
- Downloaded 2026-09-09. File checksums are recorded in sha256.json.

This integration uses the original model_22200.pt PyTorch actor and task embedder, not the RTX 4090 TensorRT engine. The actor architecture is copied unchanged into lab_sim/vendor/scalebfm_network.py. Pure PyTorch quaternion functions, export wrapper, mode mask builder, and XML parser are extracted from ScaleTrack/scripts/pretrain/rsl_rl/play_export_check_humanoid_transformer.py into lab_sim/vendor/scalebfm_export.py, without its IsaacLab launcher.

kinematics.xml is the ScaleTrack 29-DOF FK model; deployment.xml is the ScaleBridge G1 Dex3 deployment model, used for joint dynamics parameters only. Robot assets retain their upstream Unitree license (see ../../NOTICE.md). The upstream repository does not include a root LICENSE at this revision; no additional license is granted here. See README.upstream.md for the model card.
