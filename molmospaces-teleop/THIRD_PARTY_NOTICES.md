# Third-party notices for the teleoperation integration

The standalone motion-tracking integration contains material migrated from local
development repositories:

- motion-tracking inference/reference code, ONNX checkpoint, and the walk smoke
  reference originated from `HoloTeleop`;
- the G1 body resource originated from the same controller project;
- Dex1-1 geometry originated from `HIW-500-controoler` / Unitree Dex1-1 resources;
- the PICO bridge expects XRoboToolkit PC Service, `xrobotoolkit_sdk`, and
  `general_motion_retargeting` at runtime.

The local source trees did not expose a clear license file during integration. These
files may be used for local development in this checkout, but their redistribution
status must be confirmed with the respective owners before publishing this repository
or its binary assets. This notice does not replace the upstream licenses.
