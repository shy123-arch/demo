# Unitree G1 asset provenance

`robot.xml` and the STL meshes in `meshes/` are a portable copy of the local
G1 description at:

```
/mnt/workspace/Wilson/parameter-fusion/psi0-regularized-finetune-20260829-v1/
training/Psi0-regularized/real/assets/g1/
```

The source package identifies the model as Unitree Robotics' G1 and says that
`g1_body29_hand14` is modified from
[`g1_29dof_with_hand_rev_1_0`](https://github.com/unitreerobotics/unitree_ros/blob/master/robots/g1_description/g1_29dof_with_hand_rev_1_0.urdf).
The source repository's `real/LICENSE` carries the following attribution and
license: **Copyright 2024 HangZhou YuShu TECHNOLOGY CO., LTD. (Unitree
Robotics), Apache License 2.0**. Keep that attribution and the Apache 2.0
license terms when redistributing these files. The upstream Unitree model and
trademarks remain the property of Unitree Robotics.

The source MJCF SHA-256 before this integration was:

```
c7bec8183aebdc23be735bdc6705a0adc4b0fba230ecbd46a42f0fe4a7269951
```

This integration only changes the MJCF wrapper: it relocates the robot to the
lab spawn pose, removes the source floor/lighting tail, adds a 500 Hz option,
adds small joint damping/armature defaults for stable manipulation, adds the
`g1_head`, `left_palm_site`, and `right_palm_site` frames, and adds the
optional `pelvis_support` equality weld. Link inertias, mesh geometry,
collision geoms, joint ranges, and motor mappings are retained from the source.

The asset is provided without any additional warranty. Confirm the upstream
license and any hardware or trademark requirements before publishing a
downstream package.
