# G1 + Dex1-1 asset provenance

`model.xml` was generated from the 29-DoF G1 MJCF used by the local motion-tracking
controller and the Unitree Dex1-1 resource from `HIW-500-controoler`. The generated
model removes the former Dex3 bodies, adds two Dex1-1 grippers and head/wrist cameras,
and applies the simulation-only jaw closure correction documented in
`docs/g1_dex1_teleop.md`.

Only meshes referenced by the final model are retained. Run
`scripts/build_g1_dex1_asset.py` only when intentionally rebuilding from the upstream
development resources; normal simulation uses this bundled asset directly.

Redistribution of these robot resources requires confirming their upstream licenses.
