# New grasping and mobility experiments

The user requests grasping and movement tasks. Preserve camera-only environment
observations plus joint encoders and IMU. Official ScaleBFM remains the body
policy. No base support, grasp weld, external force, object truth position,
contact feedback, or evaluator success feedback in runtime control. Independent
evaluation may read truth after execution. Preset initial workstations are
allowed and must be disclosed. Produce actual physical trials, truthful failed
results, and 50 FPS synchronized replay. Python: `.venv/bin/python`, EGL works.

Root owns new image perception and high-level grasp servo/runner, integration,
rendering and documentation. Do not modify another agent's files.

## grasp_hand

Own only `lab_sim/grasp_geometry.py`, optional scripts prefixed `grasp_hand_`,
and `outputs/grasp_hand_probe/`. Study the existing right Dex3 hand joint signs,
collision geometry and cylinder grip feasibility. Build robot-only forward
kinematics / an offline geometric probe to recommend wrist quaternion, grasp
center offset from wrist, open and closed seven-joint targets for a bottle.
Use the robot XML geometry; actual task target size/position must come from RGB
in the final controller. Existing bottle diameter is about 52 mm; testing its
physical grasp feasibility offline is allowed, but disclose such development
calibration. No welds or artificial fingertip geometry. If helpful test isolated
hand physics, clearly distinguish that from unsupported whole-body success.
Report interface and findings promptly so root can integrate. No shared IO or
servo edits. Proposed helper: hand preset dicts keyed by right hand joint names,
and wrist-frame contact/grasp center.

## manipulation_io

Own only `lab_sim/vision_robot_io.py`. Extend current adapter compatibly for
grasping: explicit `set_hand_targets(dict)` accepts named hand joint positions,
clipped/validated within robot limits, drives normal joint PD and respects
actuator force bounds. Default stays zero/open. Optional `hand_sensors()` returns
only named finger encoders, not contacts. Add optional initialization-only
`bottle_shift=(0,0,0)` argument and log it, never expose to controller; preserve
existing call arguments and default states. Independently evaluate new tasks
`grasp_lift` and `pick_place` using per-physics-tick audit/traces: bottle lifted
>= 30 mm clear of initial support for >= .5 s, actual finger/bottle contact
during lifted hold, no non-hand support while airborne; pick_place additionally
ends released, upright and on table after >= 80 mm planar transport. Write
explicit metrics including max lift, airborne duration/hand contact overlap,
final planar displacement/tilt, final hand contact and support, any assist.
Evaluate truth only after controller finishes and stop future stepping.
Expose no contact or pose feedback. Keep existing button/box criteria working.
Root will build grasp policy and runner. Report readiness/interface promptly.

## scalebfm_mobility

Own only new `lab_sim/scalebfm_mobility.py`, optional `scripts/scalebfm_mobility_*`
and `outputs/scalebfm_mobility/`. Independently attempt a short unsupported
ScaleBFM stepping/movement task using only robot sensors at runtime. Existing
`ProprioScaleBFM` anchors both feet, so implement a separate gait-phase or
encoder-derived support-foot odometry subclass for motion (no root truth,
no contact truth). A user-commanded relative gait target is allowed; observe RGB
for obstacle/target monitoring if feasible, never take scene object poses as
inputs. Try bounded short forward/side steps under official five-point policy.
You may use VisionRobotIO at an open initial station, sensors/images/step only;
independent evaluator may measure actual displacement afterward from saved
trajectory and report falls/slip honestly. Keep failed trials. Save trajectory,
controller trace and initialization/report in existing formats for root replay.
Do not silently substitute GR00T for ScaleBFM; if ScaleBFM gait is not feasible,
report evidence and a clearly separated fallback recommendation. Root can keep
working on grasping in parallel. Do not edit shared controller/IO/scene files.

Root has started grasp_tasks.py and lift_v1. Mobility assignment remains available for a free agent; root will implement it if no agent is active there.
