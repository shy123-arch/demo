"""Small, image-only stereo detector for the camera control experiment.

Inputs are rectified RGB images and camera calibration.  No simulator, scene,
object dimensions, world pose, depth buffer or instance labels are used.  The
appearance priors are the task descriptions: a blue round button, or a muted
teal rectangular lid.  This is deliberately a constrained detector, not a
general-purpose object recognition model.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np


class StereoPerception:
    """Detect a target surface point, expressed in the *left* camera frame.

    Camera axes are x right, y down, z forward.  ``camera_point`` is ``None``
    when either image lacks an unambiguous, fully visible matching target.
    Confidence is a heuristic image-quality score, not a probability.
    """

    def __init__(self, width=640, height=480, fovy=75, baseline=.065):
        self.width, self.height = int(width), int(height)
        self.fovy, self.baseline = float(fovy), float(baseline)
        if min(self.width, self.height) < 2 or not 0 < self.fovy < 180 or self.baseline <= 0:
            raise ValueError("Invalid stereo calibration")
        self.fx = self.fy = self.height / (2 * math.tan(math.radians(self.fovy) / 2))
        self.cx, self.cy = (self.width - 1) / 2, (self.height - 1) / 2

    @staticmethod
    def _ellipse_error(contour, ellipse):
        center, axes, angle = ellipse
        delta = contour[:, 0, :].astype(float) - np.asarray(center)
        theta = math.radians(angle)
        rotation = np.array([[math.cos(theta), -math.sin(theta)],
                             [math.sin(theta), math.cos(theta)]])
        local = delta @ rotation
        radius = np.linalg.norm(local / (np.asarray(axes) / 2), axis=1)
        return float(np.median(np.abs(radius - 1)))

    @staticmethod
    def _quad_center(points):
        """Image of a quadrilateral's diagonal intersection, not its centroid."""
        p = points.astype(float)
        matrix = np.column_stack((p[2] - p[0], -(p[3] - p[1])))
        if abs(np.linalg.det(matrix)) < 1e-6:
            return p.mean(axis=0)
        t = np.linalg.solve(matrix, p[1] - p[0])[0]
        return p[0] + t * (p[2] - p[0])

    def _candidates(self, rgb, task):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        if task == "button":
            lower, upper = (87, 135, 70), (123, 255, 255)
        else:
            lower, upper = (82, 52, 35), (112, 155, 230)
        mask = cv2.inRange(hsv, np.asarray(lower), np.asarray(upper))
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        candidates, rejected = [], []
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area < 24 or len(contour) < 5:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            bbox = [int(x), int(y), int(w), int(h)]
            if x <= 1 or y <= 1 or x + w >= self.width - 1 or y + h >= self.height - 1:
                rejected.append({"bbox": bbox, "reason": "image_border_truncation"})
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = float(4 * math.pi * area / max(perimeter ** 2, 1))
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            solidity = float(area / max(hull_area, 1))
            approximation = cv2.approxPolyDP(contour, .025 * perimeter, True)
            ellipse = cv2.fitEllipse(contour)
            ellipse_error = self._ellipse_error(contour, ellipse)
            ellipse_ratio = min(ellipse[1]) / max(ellipse[1])
            features = {"area": area, "circularity": circularity, "solidity": solidity,
                        "polygon_vertices": len(approximation), "ellipse_error": ellipse_error}
            if task == "button":
                if circularity < .74 or solidity < .94 or ellipse_ratio < .5 or ellipse_error > .085:
                    rejected.append({"bbox": bbox, "reason": "not_round", **features})
                    continue
                center = np.asarray(ellipse[0], dtype=float)
                score = float(np.clip(.45 * circularity + .25 * solidity +
                                      .3 * (1 - ellipse_error / .15), 0, 1))
                polygon = None
            else:
                rectangle = cv2.minAreaRect(contour)
                rectangle_area = float(np.prod(rectangle[1]))
                rectangle_fill = area / max(rectangle_area, 1)
                polygon_area = cv2.contourArea(approximation)
                quad_fill = min(area, polygon_area) / max(area, polygon_area, 1)
                features.update(rectangle_fill=rectangle_fill, quad_fill=quad_fill)
                if (len(approximation) != 4 or not cv2.isContourConvex(approximation)
                        or solidity < .94 or rectangle_fill < .56 or quad_fill < .88):
                    rejected.append({"bbox": bbox, "reason": "not_complete_quadrilateral", **features})
                    continue
                polygon = approximation[:, 0, :].astype(float)
                center = self._quad_center(polygon)
                score = float(.5 * quad_fill + .3 * solidity + .2 * rectangle_fill)
            fill = np.zeros_like(mask)
            cv2.drawContours(fill, [contour], -1, 255, -1)
            hue = float(np.median(hsv[..., 0][fill > 0]))
            candidates.append({"pixel": center.tolist(), "bbox": bbox, "score": score,
                               "hue": hue, "polygon": None if polygon is None else polygon.tolist(),
                               **features})
        return candidates, rejected

    def _triangulate(self, left_pixel, right_pixel):
        u, v = np.asarray(left_pixel, dtype=float)
        disparity = float(u - right_pixel[0])
        z = self.fx * self.baseline / disparity
        return np.array([(u - self.cx) * z / self.fx, (v - self.cy) * z / self.fy, z])

    def _quad_stereo(self, left, right):
        """Match cyclic vertices using epipolar geometry; recover visible corners."""
        lp, rp = np.asarray(left["polygon"]), np.asarray(right["polygon"])
        possible = []
        for direction in (rp, rp[::-1]):
            for offset in range(4):
                candidate = np.roll(direction, offset, axis=0)
                disparity = lp[:, 0] - candidate[:, 0]
                if np.min(disparity) <= 1.25:
                    continue
                error = float(np.mean(np.abs(lp[:, 1] - candidate[:, 1])))
                possible.append((error, candidate))
        if not possible:
            return None
        error, rp = min(possible, key=lambda item: item[0])
        if error > 3:
            return None
        points = np.asarray([self._triangulate(l, r) for l, r in zip(lp, rp)])
        center = points.mean(axis=0)
        _, _, vh = np.linalg.svd(points - center)
        normal = vh[-1]
        if normal @ center > 0:
            normal = -normal  # Surface normal faces the cameras.
        return {"left_polygon": lp.tolist(), "right_polygon": rp.tolist(),
                "camera_corners": points.tolist(), "camera_surface_normal": normal.tolist(),
                "corner_epipolar_error_px": error,
                "plane_residual_m": float(np.max(np.abs((points - center) @ normal)))}

    @staticmethod
    def _shared_edge_error(polygon, lid_polygon):
        """Find edge adjacency in pixels; no fixed ROI or physical size prior."""
        best = math.inf
        for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
            length = np.linalg.norm(b - a)
            if length < 4:
                continue
            for c, d in zip(lid_polygon, np.roll(lid_polygon, -1, axis=0)):
                other_length = np.linalg.norm(d - c)
                if other_length < 4:
                    continue
                alignment = abs(float((b - a) @ (d - c))) / (length * other_length)
                if alignment < .85:
                    continue
                error = min(max(np.linalg.norm(a - c), np.linalg.norm(b - d)),
                            max(np.linalg.norm(a - d), np.linalg.norm(b - c)))
                # Antialiasing and a thin differently colored rim can separate edges.
                scale = max(5., .075 * min(length, other_length))
                best = min(best, error / scale)
        return float(best)

    def _white_side_candidates(self, rgb, lid):
        """Find complete white quadrilaterals sharing an observed lid edge."""
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        white_level = int(round(.92 * np.percentile(hsv[..., 2], 99.5)))
        mask = cv2.inRange(hsv, np.array((0, 0, white_level)), np.array((179, 35, 255)))
        candidates = []
        lid_polygon = np.asarray(lid["polygon"])
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            area = cv2.contourArea(contour)
            if area < 24:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if x <= 1 or y <= 1 or x + w >= self.width - 1 or y + h >= self.height - 1:
                continue
            poly = cv2.approxPolyDP(contour, .018 * cv2.arcLength(contour, True), True)
            if len(poly) != 4 or not cv2.isContourConvex(poly):
                continue
            solidity = area / max(cv2.contourArea(cv2.convexHull(contour)), 1)
            if solidity < .95:
                continue
            polygon = poly[:, 0, :].astype(float)
            adjacency_error = self._shared_edge_error(polygon, lid_polygon)
            if adjacency_error > 1:
                continue
            candidates.append({"pixel": self._quad_center(polygon).tolist(),
                               "polygon": polygon.tolist(), "bbox": [x, y, w, h],
                               "area": area, "adjacency_error": adjacency_error})
        return candidates

    def _white_side_stereo(self, left_rgb, right_rgb, left_lid, right_lid):
        left = self._white_side_candidates(left_rgb, left_lid)
        right = self._white_side_candidates(right_rgb, right_lid)
        matches = []
        for l in left:
            for r in right:
                disparity = l["pixel"][0] - r["pixel"][0]
                epipolar = abs(l["pixel"][1] - r["pixel"][1])
                if disparity <= 1.25 or epipolar > 3 or abs(math.log(l["area"] / r["area"])) > .4:
                    continue
                geometry = self._quad_stereo(l, r)
                if geometry is not None:
                    matches.append({"camera_point": self._triangulate(l["pixel"], r["pixel"]).tolist(),
                                    "left_pixel": l["pixel"], "right_pixel": r["pixel"],
                                    "bbox": l["bbox"], "right_bbox": r["bbox"],
                                    "epipolar_error_px": epipolar, **geometry})
        if len(matches) == 1:
            return {"status": "detected", **matches[0]}
        return {"status": "missing_or_ambiguous_white_side", "match_count": len(matches)}

    def detect(self, left_rgb, right_rgb, task="button") -> dict[str, Any]:
        if task not in ("button", "push_box"):
            raise ValueError("task must be 'button' or 'push_box'")
        images = [np.asarray(left_rgb), np.asarray(right_rgb)]
        for image in images:
            if image.shape != (self.height, self.width, 3) or image.dtype != np.uint8:
                raise ValueError(f"Expected RGB uint8 images shaped {(self.height, self.width, 3)}")
        left, left_rejected = self._candidates(images[0], task)
        right, right_rejected = self._candidates(images[1], task)
        diagnostics = {"camera_frame": "left_camera_x_right_y_down_z_forward",
                       "task": task, "left_candidates": left, "right_candidates": right,
                       "left_rejected": left_rejected, "right_rejected": right_rejected,
                       "calibration": {"fx": self.fx, "fy": self.fy, "cx": self.cx,
                                       "cy": self.cy, "baseline_m": self.baseline},
                       "confidence_is_calibrated_probability": False}
        result = {"camera_point": None, "left_pixel": None, "right_pixel": None,
                  "confidence": 0., "bbox": None, "diagnostics": diagnostics}
        pairs = []
        for l in left:
            for r in right:
                disparity = l["pixel"][0] - r["pixel"][0]
                epipolar = abs(l["pixel"][1] - r["pixel"][1])
                area_log = abs(math.log(l["area"] / r["area"]))
                hue_error = abs(l["hue"] - r["hue"])
                if (disparity <= 1.25 or disparity > .9 * self.width or epipolar > 4
                        or area_log > .55 or hue_error > 10):
                    continue
                score = min(l["score"], r["score"]) * math.exp(-.12 * epipolar - .25 * area_log)
                pairs.append((score, l, r, disparity, epipolar))
        pairs.sort(key=lambda item: item[0], reverse=True)
        diagnostics["stereo_pair_count"] = len(pairs)
        if not pairs:
            diagnostics["status"] = "target_missing_occluded_or_no_stereo_match"
            return result
        if len(pairs) > 1 and pairs[0][0] - pairs[1][0] < .12:
            diagnostics["status"] = "ambiguous_multiple_similar_targets"
            diagnostics["pair_scores"] = [pair[0] for pair in pairs]
            return result
        score, l, r, disparity, epipolar = pairs[0]
        point = self._triangulate(l["pixel"], r["pixel"])
        diagnostics.update(status="detected", disparity_px=disparity, epipolar_error_px=epipolar,
                           depth_sensitivity_m_per_disparity_px=float(point[2] / disparity))
        if task == "push_box":
            corners = self._quad_stereo(l, r)
            if corners is not None:
                diagnostics.update(corners)
            diagnostics["white_side_face"] = self._white_side_stereo(images[0], images[1], l, r)
        result.update(camera_point=point.tolist(), left_pixel=l["pixel"], right_pixel=r["pixel"],
                      confidence=float(score), bbox=l["bbox"], right_bbox=r["bbox"])
        return result
