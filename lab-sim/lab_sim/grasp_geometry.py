"""Robot-side Dex3 preshape for visually measured upright cylindrical objects.

No simulator state, object identifiers, object poses, or scene geometry are read
here. Dimensions below describe the right hand's deployed kinematic geometry.
The separate ``outputs/grasp_hand_probe`` fixture is mechanical calibration; its
supported isolated wrist experiments are not autonomous robot task results.
"""
from dataclasses import dataclass
import numpy as np

RIGHT_HAND_JOINT_NAMES = tuple(
    'right_hand_' + joint + '_joint'
    for joint in ('thumb_0', 'thumb_1', 'thumb_2', 'index_0', 'index_1',
                  'middle_0', 'middle_1')
)


@dataclass(frozen=True)
class CylinderGrasp:
    """Targets in actuator-name order and a wrist-frame grasp center."""
    open_q: np.ndarray
    closed_q: np.ndarray
    center_in_wrist: np.ndarray
    wrist_wxyz: np.ndarray
    visual_width_m: float


def grasp_targets(width_m: float) -> CylinderGrasp:
    """Return a right-hand preshape for a measured 40–64 mm cylinder.

    The wrist's local z axis stays vertical, local x points forward, and local
    +y points into the palm's grasp space. Translate the wrist so
    ``wrist_position + R @ center_in_wrist`` matches the RGB-estimated grasp
    center. Interpolate open→closed over at least 1.2 seconds before lifting.
    Width must come from visual reconstruction, not a scene object property.
    The nominal 52 mm mechanical fixture supports a ~100 mm isolated lift;
    whole-body balance, visual error, reachability and transport need validation
    independently. Wider/narrower objects are rejected, not silently clamped.
    """
    width = float(width_m)
    if not np.isfinite(width) or not .040 <= width <= .064:
        raise ValueError('visually measured cylinder width must be 0.040–0.064 m')
    # Index/middle pivots: x=.1192, y=-.0046; vertical spacing ±.0285 m.
    # Put the cylinder just beyond the inner palm pad. The measured radius
    # expands the desired free-space center; these constants are hand geometry.
    center = np.array([.1192 + .0008, -.0046 + .0286 + width / 2, 0.])
    # Preshape calibration: smaller visually measured cylinders need more
    # closure. The 52 mm reference is a calibration diameter, never a runtime
    # object dimension fetched from the environment.
    narrower = max(0., .052 - width)
    finger = .7 + 12.5 * narrower
    thumb = -.4 - 10. * narrower
    return CylinderGrasp(
        open_q=np.zeros(7),
        closed_q=np.array([0., thumb, -.4, finger, finger, finger, finger]),
        center_in_wrist=center,
        wrist_wxyz=np.array([1., 0., 0., 0.]),
        visual_width_m=width,
    )
