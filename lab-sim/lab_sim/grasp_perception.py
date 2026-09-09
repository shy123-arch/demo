"""RGB-only stereo localization of a blue bottle with a white label band.

The detector uses appearance/shape priors and camera calibration only. Bottle
dimensions are measured from pixels and stereo, not read from a scene model.
"""
import math

import cv2
import numpy as np

from .stereo_perception import StereoPerception


class BottleStereo(StereoPerception):
    def _bottle_candidates(self, rgb):
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        mask = cv2.inRange(hsv, np.array((87, 135, 65)), np.array((123, 255, 255)))
        components = []
        for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]:
            area = cv2.contourArea(contour)
            if area < 40:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if min(x, y) < 2 or x + w >= self.width - 2 or y + h >= self.height - 2:
                continue
            components.append((x, y, w, h, contour, area))
        candidates = []
        for upper in components:
            x, y, w, h, contour, area = upper
            for lower in components:
                lx, ly, lw, lh, lc, la = lower
                width = .5 * (w + lw)
                gap = ly - (y + h)
                if (not 0 < gap < width or abs((x + .5*w) - (lx + .5*lw)) > .35*width
                        or min(w, lw) / max(w, lw) < .65):
                    continue
                bx, by = min(x,lx), y
                bw, bh = max(x+w,lx+lw)-bx, ly+lh-by
                if not 1.8 < bh / width < 6:
                    continue
                band = hsv[y+h:ly, max(x,lx):min(x+w,lx+lw)]
                if not band.size:
                    continue
                white_fraction = float(np.mean((band[:,:,1] < 55) & (band[:,:,2] > 130)))
                if white_fraction < .65:
                    continue
                # A slanted projected cylinder has a wider enclosing box than
                # its actual silhouette cross-section. Measure row spans.
                spans=[]
                for row in range(by+3,by+bh-3):
                    pixels=np.flatnonzero(mask[row,bx:bx+bw])
                    if len(pixels)>5:
                        spans.append(pixels[-1]-pixels[0]+1)
                row_width=float(np.median(spans)) if spans else width
                candidates.append({
                    'pixel': [bx + .5*(bw-1), by + .5*(bh-1)],
                    'bbox': [bx,by,bw,bh], 'area': float(area+la),
                    'width_px': row_width, 'height_px': float(bh),
                    'white_band_fraction': white_fraction,
                })
        return candidates

    def detect(self, left_rgb, right_rgb, task='grasp_lift'):
        images = [np.asarray(left_rgb), np.asarray(right_rgb)]
        for image in images:
            if image.shape != (self.height,self.width,3) or image.dtype != np.uint8:
                raise ValueError('Expected calibrated RGB uint8 stereo images')
        left,right = [self._bottle_candidates(im) for im in images]
        diagnostics = {'status':'missing_or_ambiguous_bottle', 'left_candidates':left,
                       'right_candidates':right, 'appearance_prior':'blue cylinder with white band',
                       'camera_frame':'left_camera_x_right_y_down_z_forward'}
        result = {'camera_point':None,'left_pixel':None,'right_pixel':None,
                  'bbox':None,'confidence':0.,'diagnostics':diagnostics}
        matches=[]
        for l in left:
            for r in right:
                disparity=l['pixel'][0]-r['pixel'][0]
                epipolar=abs(l['pixel'][1]-r['pixel'][1])
                if (disparity <= 2 or epipolar > 4
                        or abs(math.log(l['area']/r['area'])) > .5):
                    continue
                point=self._triangulate(l['pixel'],r['pixel'])
                # Perspective magnification of a circular cross-section away
                # from the optical axis; uses only the triangulated viewing ray.
                diameter=l['width_px']*point[2]/self.fx / np.sqrt(1+(point[0]/point[2])**2)
                height=l['height_px']*point[2]/self.fy
                matches.append((l,r,point,diameter,height,epipolar))
        if len(matches) != 1:
            diagnostics['match_count']=len(matches)
            return result
        l,r,point,diameter,height,epipolar=matches[0]
        diagnostics.update(status='detected',diameter_estimate_m=float(diameter),
                           projected_height_estimate_m=float(height),
                           epipolar_error_px=float(epipolar),right_bbox=r['bbox'])
        result.update(camera_point=point.tolist(),left_pixel=l['pixel'],right_pixel=r['pixel'],
                      bbox=l['bbox'],confidence=float(min(l['white_band_fraction'],r['white_band_fraction'])))
        return result
