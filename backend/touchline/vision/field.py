"""Field-region segmentation and pitch-line detection.

The field mask is a heuristic: grass hue range in HSV, morphological closing to
absorb players and lines, then the largest connected component. It is used to
(1) restrict detector tiles to the playing surface, (2) reject detections whose
ground-contact point is not on grass (photographers, stewards, crowd), and
(3) exclude moving players from camera-motion feature tracking (we only track
features on static scene structure).
"""
from __future__ import annotations

import cv2
import numpy as np

GRASS_LO = (30, 50, 35)
GRASS_HI = (90, 255, 255)


def field_mask(frame: np.ndarray) -> np.ndarray:
    small = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, GRASS_LO, GRASS_HI)
    green = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    n, lab, st, _ = cv2.connectedComponentsWithStats(green)
    if n <= 1:
        return np.zeros(frame.shape[:2], np.uint8)
    big = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    m = (lab == big).astype(np.uint8) * 255
    # Fill holes left by players/lines: close strongly, then fill from the convex structure row-wise.
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return cv2.resize(m, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)


def field_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def field_fraction(mask: np.ndarray) -> float:
    return float(np.count_nonzero(mask)) / mask.size


def white_line_mask(frame: np.ndarray, fmask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    inner = cv2.erode(fmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))
    return ((tophat > 28) & (inner > 0) & (hsv[..., 1] < 110)).astype(np.uint8) * 255


def detect_line_segments(frame: np.ndarray, fmask: np.ndarray | None = None) -> np.ndarray:
    """Pitch-marking segments (N,4) as x1,y1,x2,y2 via top-hat + probabilistic Hough."""
    if fmask is None:
        fmask = field_mask(frame)
    lines = cv2.HoughLinesP(white_line_mask(frame, fmask), 1, np.pi / 360, 50, minLineLength=60, maxLineGap=12)
    if lines is None:
        return np.zeros((0, 4), np.float32)
    # OpenCV 5 returns (N,4); OpenCV 4 returned (N,1,4). Normalise.
    return np.asarray(lines, np.float32).reshape(-1, 4)


def snap_to_intersection(point: tuple[float, float], segments: np.ndarray, radius: float = 25.0):
    """Return the nearest intersection of two non-parallel detected line segments within `radius` px.

    Used as an optional calibration aid: a user's click near a marking corner is snapped
    to the corner implied by the detected lines. Returns None when nothing qualifies.
    """
    p = np.asarray(point, np.float64)
    best, best_d = None, radius
    segs = np.asarray(segments, np.float64)
    near = [s for s in segs if _dist_point_segment(p, s) < radius * 2]
    for i in range(len(near)):
        for j in range(i + 1, len(near)):
            a, b = near[i], near[j]
            da = np.arctan2(a[3] - a[1], a[2] - a[0])
            db = np.arctan2(b[3] - b[1], b[2] - b[0])
            ang = abs((np.degrees(da - db) + 90) % 180 - 90)
            if ang < 12:
                continue
            q = _intersect(a, b)
            if q is None:
                continue
            d = float(np.linalg.norm(q - p))
            if d < best_d:
                best, best_d = q, d
    return None if best is None else (float(best[0]), float(best[1]))


def _intersect(a, b):
    x1, y1, x2, y2 = a
    x3, y3, x4, y4 = b
    den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(den) < 1e-9:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den
    return np.array([x1 + t * (x2 - x1), y1 + t * (y2 - y1)])


def _dist_point_segment(p, s):
    a, b = s[:2], s[2:]
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / max(np.dot(ab, ab), 1e-9), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))
