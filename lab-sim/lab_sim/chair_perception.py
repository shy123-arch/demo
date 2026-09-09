"""Constrained RGB-only localization of an exposed gray chair back.

The appearance prior is a dark, slightly blue gray planar back with a visible
top edge and descending sides.  Both top corners must be observed in stereo;
the seat and the bottom of the back may be outside the image or hand-occluded.
No object dimensions, scene coordinates, simulator state, or depth are used.
The result is the measured front-surface top-edge midpoint, not a guessed
contact point.  A separate dark cap is excluded using image intensity, so
its shallow, viewpoint-dependent silhouette cannot supply stereo corners.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .stereo_perception import StereoPerception


class ChairStereo(StereoPerception):
    """Locate a complete exposed top edge in the left camera's metric frame.

    Image coordinates are x right / y down; camera coordinates are x right /
    y down / z forward.  This appearance-specific detector deliberately abstains
    on multiple similar backs, an occluded top edge, or weak stereo geometry.
    Confidence describes image geometry quality and is not a probability.
    """

    @staticmethod
    def _sample_fraction(mask, points):
        pixels = np.rint(points).astype(int)
        good = ((pixels[:, 0] >= 0) & (pixels[:, 0] < mask.shape[1]) &
                (pixels[:, 1] >= 0) & (pixels[:, 1] < mask.shape[0]))
        values = np.zeros(len(pixels), dtype=bool)
        values[good] = mask[pixels[good, 1], pixels[good, 0]] > 0
        return float(values.mean())

    @staticmethod
    def _surface_regions(hsv):
        """Separate the broad back surface from its darker cap and seat.

        The coarse hue/saturation prior can join the back, cap, seat and nearby
        similarly colored objects.  The dominant observed intensity within each
        connected region identifies the broad planar face.  Its band is derived
        from those pixels, with no expected object location, depth or dimensions.
        """
        appearance = cv2.inRange(hsv, np.array((94, 32, 45)), np.array((115, 90, 175)))
        contours = cv2.findContours(appearance, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        for contour in contours:
            if cv2.contourArea(contour) < 600:
                continue
            region = np.zeros_like(appearance)
            cv2.drawContours(region, [contour], -1, 255, -1)
            region &= appearance
            values = hsv[..., 2][region > 0]
            histogram = np.bincount(values, minlength=256).astype(float)
            # Smooth quantization peaks without mixing the separate dark cap.
            histogram = np.convolve(histogram, np.ones(5), mode="same")
            mode = int(histogram.argmax())
            tolerance = max(10., .10 * mode)
            surface = region.copy()
            surface[np.abs(hsv[..., 2].astype(float) - mode) > tolerance] = 0
            for face in cv2.findContours(surface, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
                if cv2.contourArea(face) >= 600:
                    yield surface, face, mode, tolerance

    def _back_candidates(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        candidates, rejected = [], []
        for mask, contour, value_center, value_tolerance in self._surface_regions(hsv):
            area = float(cv2.contourArea(contour))
            if area < 600:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            bbox = [int(x), int(y), int(width), int(height)]
            if width < 35 or height < .18 * width:
                rejected.append({"bbox": bbox, "reason": "insufficient_back_surface"})
                continue
            hull = cv2.convexHull(contour)
            polygon = cv2.approxPolyDP(hull, .002 * cv2.arcLength(hull, True), True)[:, 0].astype(float)
            possible_edges = []
            border_top_edge = False
            for index in range(len(polygon)):
                a, b = polygon[index], polygon[(index + 1) % len(polygon)]
                if a[0] > b[0]:
                    a, b = b, a
                delta = b - a
                length = float(np.linalg.norm(delta))
                if (length < max(30., .75 * width) or abs(delta[1]) > .55 * delta[0]
                        or .5 * (a[1] + b[1]) > y + .28 * height):
                    continue
                if (min(a[0], b[0]) <= 2 or max(a[0], b[0]) >= self.width - 3
                        or min(a[1], b[1]) <= 2 or max(a[1], b[1]) >= self.height - 3):
                    border_top_edge = True
                    continue
                tangent = delta / length
                inward = np.array((-tangent[1], tangent[0]))
                samples = a + np.linspace(.035, .965, 80)[:, None] * delta
                # Convex-hull edges alone could bridge an occluding hand.  Demand
                # directly observed surface below almost the entire top edge.
                inside = self._sample_fraction(mask, samples + inward * 3.)
                above = self._sample_fraction(mask, samples - inward * 3.)
                deeper = self._sample_fraction(mask, samples + inward * max(7., .025 * length))
                if inside < .94 or above > .18 or deeper < .90:
                    continue
                # Both ends must have descending visible surface.  This prevents
                # an arbitrary horizontal stripe from standing in for a back.
                side_points = np.vstack((a + .04 * delta + inward * max(10., .07 * length),
                                         b - .04 * delta + inward * max(10., .07 * length)))
                if self._sample_fraction(mask, side_points) < 1.:
                    continue
                # A cut-out corner can expose a shorter straight top segment.
                # Its convex hull bridges the missing corner to the true side.
                # Require both adjoining hull sides to be directly observed near
                # the corners too, instead of accepting that synthetic junction.
                side_support = []
                for corner, other in ((a, b), (b, a)):
                    corner_index = int(np.argmin(np.linalg.norm(polygon - corner, axis=1)))
                    neighbors = (polygon[(corner_index - 1) % len(polygon)],
                                 polygon[(corner_index + 1) % len(polygon)])
                    neighbor = max(neighbors, key=lambda point: np.linalg.norm(point - other))
                    side = neighbor - corner
                    side_length = float(np.linalg.norm(side))
                    if side_length < 8 or side[1] < 4:
                        side_support.append(0.)
                        continue
                    side_tangent = side / side_length
                    side_inward = np.array((-side_tangent[1], side_tangent[0]))
                    if side_inward @ (other - corner) < 0:
                        side_inward *= -1
                    distance = min(.8 * side_length, max(12., .10 * length))
                    side_samples = corner + np.linspace(.12, 1., 25)[:, None] * distance * side_tangent
                    side_support.append(self._sample_fraction(mask, side_samples + 3. * side_inward))
                if min(side_support) < .92:
                    continue
                possible_edges.append((length, a, b, inside, above, deeper, side_support))
            if not possible_edges:
                rejected.append({"bbox": bbox, "reason": "top_corner_outside_image" if border_top_edge
                                 else "top_edge_missing_or_occluded"})
                continue
            _, a, b, inside, above, deeper, side_support = max(possible_edges, key=lambda edge: edge[0])
            region = np.zeros_like(mask)
            cv2.drawContours(region, [contour], -1, 255, -1)
            region = (region > 0) & (mask > 0)
            hue = float(np.median(hsv[..., 0][region]))
            saturation = float(np.median(hsv[..., 1][region]))
            candidates.append({"pixel": ((a + b) / 2).tolist(), "bbox": bbox,
                               "top_edge": [a.tolist(), b.tolist()], "area": area,
                               "hue": hue, "saturation": saturation,
                               "surface_value_center": value_center,
                               "surface_value_tolerance": value_tolerance,
                               "top_edge_observed_fraction": inside,
                               "top_edge_background_fraction": 1 - above,
                               "surface_below_edge_fraction": deeper,
                               "corner_side_observed_fractions": side_support,
                               "score": float(.5 * inside + .25 * (1 - above) + .25 * deeper)})
        return candidates, rejected

    def detect(self, left_rgb, right_rgb, task="push_chair") -> dict[str, Any]:
        if task != "push_chair":
            raise ValueError("ChairStereo task must be 'push_chair'")
        images = [np.asarray(left_rgb), np.asarray(right_rgb)]
        for rgb in images:
            if rgb.shape != (self.height, self.width, 3) or rgb.dtype != np.uint8:
                raise ValueError(f"Expected RGB uint8 images shaped {(self.height, self.width, 3)}")
        left, left_rejected = self._back_candidates(images[0])
        right, right_rejected = self._back_candidates(images[1])
        diagnostics = {"camera_frame": "left_camera_x_right_y_down_z_forward", "task": task,
                       "point_semantics": "observed_chair_back_front_surface_top_edge_midpoint",
                       "left_candidates": left, "right_candidates": right,
                       "left_rejected": left_rejected, "right_rejected": right_rejected,
                       "calibration": {"fx": self.fx, "fy": self.fy, "cx": self.cx,
                                       "cy": self.cy, "baseline_m": self.baseline},
                       "confidence_is_calibrated_probability": False}
        result = {"camera_point": None, "left_pixel": None, "right_pixel": None,
                  "confidence": 0., "bbox": None, "diagnostics": diagnostics}
        pairs = []
        for l in left:
            for r in right:
                lp, rp = np.asarray(l["top_edge"]), np.asarray(r["top_edge"])
                disparities = lp[:, 0] - rp[:, 0]
                epipolar = float(np.max(np.abs(lp[:, 1] - rp[:, 1])))
                edge_length_ratio = abs(math.log(np.linalg.norm(lp[1] - lp[0]) /
                                                 np.linalg.norm(rp[1] - rp[0])))
                if (np.min(disparities) <= 1.25 or np.max(disparities) > .9 * self.width
                        or epipolar > 3.5 or edge_length_ratio > .22
                        or abs(l["hue"] - r["hue"]) > 8
                        or abs(l["saturation"] - r["saturation"]) > 18):
                    continue
                points = np.asarray([self._triangulate(a, b) for a, b in zip(lp, rp)])
                score = min(l["score"], r["score"]) * math.exp(-.10 * epipolar - .6 * edge_length_ratio)
                pairs.append((score, l, r, points, disparities, epipolar))
        pairs.sort(key=lambda pair: pair[0], reverse=True)
        diagnostics["stereo_pair_count"] = len(pairs)
        if not pairs:
            diagnostics["status"] = "target_missing_occluded_or_no_stereo_match"
            return result
        if len(pairs) > 1 and pairs[0][0] - pairs[1][0] < .12:
            diagnostics.update(status="ambiguous_multiple_similar_targets",
                               pair_scores=[float(pair[0]) for pair in pairs])
            return result
        score, l, r, points, disparities, epipolar = pairs[0]
        point = points.mean(axis=0)
        diagnostics.update(status="detected", disparity_px=float(disparities.mean()),
                           corner_disparities_px=disparities.tolist(),
                           epipolar_error_px=epipolar, camera_corners=points.tolist(),
                           left_polygon=l["top_edge"], right_polygon=r["top_edge"],
                           depth_sensitivity_m_per_disparity_px=float(np.max(points[:, 2] / disparities)))
        result.update(camera_point=point.tolist(), left_pixel=l["pixel"], right_pixel=r["pixel"],
                      confidence=float(score), bbox=l["bbox"], right_bbox=r["bbox"])
        return result
