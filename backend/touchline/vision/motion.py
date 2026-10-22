"""Camera-motion estimation between frames.

Handheld and panning cameras rotate far more than they translate, so the image-to-image
motion is well modelled by a homography over the whole scene (pure rotation induces a
homography regardless of depth). We estimate it two ways:

* ``match_to_reference``: ORB features + Hamming brute-force matching + Lowe ratio test,
  then ``cv2.findHomography(..., cv2.USAC_MAGSAC)``. Anchoring every frame to the
  calibration (reference) frame avoids accumulating drift.
* ``track_pair``: Shi-Tomasi corners + pyramidal Lucas-Kanade optical flow with a
  forward-backward consistency check, then MAGSAC. Used to chain frame-to-frame when
  the direct reference match fails.

Players are masked out of feature extraction (they move independently of the camera).
Each estimate reports an inlier ratio and median inlier reprojection error so the agent
can decide whether the motion model is trustworthy enough to update the calibration.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np

# Thresholds for a "validated" motion model (see docs/QUALITY_SIGNALS.md).
MIN_INLIERS = 40
MIN_INLIER_RATIO = 0.45
MAX_REPROJ_PX = 2.5
WORK_WIDTH = 960  # estimate at half resolution for speed, then rescale


@dataclass
class MotionEstimate:
    M: list[list[float]] | None  # maps current-frame pixels -> target-frame pixels (full resolution)
    method: str
    matches: int
    inliers: int
    inlier_ratio: float
    reproj_error_px: float
    validated: bool

    def to_dict(self) -> dict:
        return asdict(self)


def _scale_mats(s: float):
    S = np.diag([s, s, 1.0])
    return S, np.linalg.inv(S)


def _gray_small(frame: np.ndarray) -> tuple[np.ndarray, float]:
    s = WORK_WIDTH / frame.shape[1]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_AREA), s


def static_mask(shape: tuple[int, int], boxes: list[np.ndarray], scale: float) -> np.ndarray:
    h, w = shape
    m = np.full((h, w), 255, np.uint8)
    for b in boxes:
        x1, y1, x2, y2 = (np.asarray(b) * scale).astype(int)
        pad = int(0.3 * max(1, y2 - y1))
        cv2.rectangle(m, (x1 - pad, y1 - pad), (x2 + pad, y2 + pad), 0, -1)
    return m


def _fit(src: np.ndarray, dst: np.ndarray, method: str, n_matches: int, scale: float) -> MotionEstimate:
    if len(src) < 8:
        return MotionEstimate(None, method, n_matches, 0, 0.0, float("inf"), False)
    M, inl = cv2.findHomography(src, dst, cv2.USAC_MAGSAC, 3.0, maxIters=2000, confidence=0.999)
    if M is None or inl is None:
        return MotionEstimate(None, method, n_matches, 0, 0.0, float("inf"), False)
    inl = inl.reshape(-1).astype(bool)
    proj = cv2.perspectiveTransform(src[inl].reshape(-1, 1, 2), M).reshape(-1, 2)
    err = float(np.median(np.linalg.norm(proj - dst[inl], axis=1))) / scale if inl.any() else float("inf")
    S, Si = _scale_mats(scale)
    M_full = Si @ M @ S
    M_full /= M_full[2, 2]
    n_in = int(inl.sum())
    ratio = n_in / max(1, len(src))
    ok = n_in >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO and err <= MAX_REPROJ_PX and _sane(M_full)
    return MotionEstimate(M_full.tolist(), method, n_matches, n_in, ratio, err, bool(ok))


def _sane(M: np.ndarray) -> bool:
    """Reject homographies that imply a mirror, a collapse, or an extreme zoom between frames."""
    A = M[:2, :2]
    det = float(np.linalg.det(A))
    return 0.5 < det < 2.0 and abs(M[2, 0]) < 2e-3 and abs(M[2, 1]) < 2e-3


class ReferenceMatcher:
    def __init__(self, ref_frame: np.ndarray, ref_boxes: list[np.ndarray] | None = None, nfeatures: int = 3000):
        self.orb = cv2.ORB_create(nfeatures=nfeatures, fastThreshold=10)
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        g, self.s = _gray_small(ref_frame)
        self.ref_shape = g.shape
        self.kp, self.des = self.orb.detectAndCompute(g, static_mask(g.shape, ref_boxes or [], self.s))

    def match_to_reference(self, frame: np.ndarray, boxes: list[np.ndarray]) -> MotionEstimate:
        g, s = _gray_small(frame)
        kp, des = self.orb.detectAndCompute(g, static_mask(g.shape, boxes, s))
        if des is None or self.des is None or len(kp) < 8:
            return MotionEstimate(None, "orb_ref", 0, 0, 0.0, float("inf"), False)
        pairs = self.bf.knnMatch(des, self.des, k=2)
        good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.75 * p[1].distance]
        src = np.float32([kp[m.queryIdx].pt for m in good]).reshape(-1, 2)
        dst = np.float32([self.kp[m.trainIdx].pt for m in good]).reshape(-1, 2)
        return _fit(src, dst, "orb_ref", len(good), s)


def track_pair(prev: np.ndarray, cur: np.ndarray, prev_boxes: list[np.ndarray]) -> MotionEstimate:
    """Homography mapping `cur` pixels to `prev` pixels from LK optical flow."""
    gp, s = _gray_small(prev)
    gc, _ = _gray_small(cur)
    pts = cv2.goodFeaturesToTrack(gp, maxCorners=1500, qualityLevel=0.01, minDistance=7,
                                  mask=static_mask(gp.shape, prev_boxes, s))
    if pts is None or len(pts) < 8:
        return MotionEstimate(None, "lk_pair", 0, 0, 0.0, float("inf"), False)
    lk = dict(winSize=(21, 21), maxLevel=4, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))
    nxt, st, _ = cv2.calcOpticalFlowPyrLK(gp, gc, pts, None, **lk)
    back, st2, _ = cv2.calcOpticalFlowPyrLK(gc, gp, nxt, None, **lk)
    fb = np.linalg.norm((back - pts).reshape(-1, 2), axis=1)
    ok = (st.reshape(-1) == 1) & (st2.reshape(-1) == 1) & (fb < 1.0)
    src = nxt.reshape(-1, 2)[ok].astype(np.float32)  # in cur
    dst = pts.reshape(-1, 2)[ok].astype(np.float32)  # in prev
    return _fit(src, dst, "lk_pair", int(ok.sum()), s)


def quad_drift_px(M_to_ref: np.ndarray, quad_ref: np.ndarray) -> float:
    """How far (px) the calibration quad has moved in the current frame relative to the reference frame."""
    Minv = np.linalg.inv(np.asarray(M_to_ref, np.float64))
    cur = cv2.perspectiveTransform(np.asarray(quad_ref, np.float64).reshape(-1, 1, 2), Minv).reshape(-1, 2)
    return float(np.max(np.linalg.norm(cur - np.asarray(quad_ref), axis=1)))
