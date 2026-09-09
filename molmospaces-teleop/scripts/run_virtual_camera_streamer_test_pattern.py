#!/usr/bin/env python3
"""Send a moving test pattern to XRoboToolkit Remote Vision."""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from molmo_spaces.utils.virtual_camera_streamer import VirtualCameraStreamer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=13579)
    parser.add_argument("--transport", choices=["TCP", "UDP", "AUTO"], default="TCP")
    parser.add_argument("--fps", type=float, default=25.0)
    parser.add_argument("--camera-name", default="head_camera")
    parser.add_argument("--stereo", action="store_true", help="send a side-by-side stereo test pattern")
    parser.add_argument("--left-camera-name", default="left_eye_camera")
    parser.add_argument("--right-camera-name", default="right_eye_camera")
    return parser.parse_args()


def make_frame(index: int, width: int = 640, height: int = 480, eye: str = "MONO") -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    x = np.linspace(0, 255, width, dtype=np.uint8)
    y = np.linspace(0, 255, height, dtype=np.uint8)
    if eye == "LEFT":
        frame[..., 0] = 70
        frame[..., 1] = y[:, None]
        frame[..., 2] = x[None, :]
    elif eye == "RIGHT":
        frame[..., 0] = x[None, :]
        frame[..., 1] = y[:, None]
        frame[..., 2] = 70
    else:
        frame[..., 0] = x[None, :]
        frame[..., 1] = y[:, None]
        frame[..., 2] = 80
    center = ((index * 9) % width, height // 2)
    cv2.circle(frame, center, 48, (255, 255, 255), -1)
    cv2.putText(
        frame,
        f"MolmoSpaces {eye}",
        (32, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"frame {index}",
        (32, height - 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    return frame


def main() -> None:
    args = parse_args()
    streamer = VirtualCameraStreamer(
        camera_name=args.camera_name,
        left_camera_name=args.left_camera_name if args.stereo else None,
        right_camera_name=args.right_camera_name if args.stereo else None,
        video_mode="stereo_sbs" if args.stereo else "mono",
        host=args.host,
        port=args.port,
        transport=args.transport,
    )
    streamer.start()
    period = 1.0 / max(args.fps, 1.0)
    index = 0
    print(
        "[run_virtual_camera_streamer_test_pattern] On PICO: Remote Vision -> "
        f"{'PICO4U or ZEDMINI' if args.stereo else 'WEBCAM'} -> Listen -> this PC IP"
    )
    try:
        while True:
            if args.stereo:
                streamer.submit_observation(
                    {
                        args.left_camera_name: make_frame(index, eye="LEFT"),
                        args.right_camera_name: make_frame(index, eye="RIGHT"),
                    }
                )
            else:
                streamer.submit_observation({args.camera_name: make_frame(index)})
            index += 1
            time.sleep(period)
    except KeyboardInterrupt:
        pass
    finally:
        streamer.stop()


if __name__ == "__main__":
    main()
