"""Four-point field calibration: validation, homography, and image<->pitch transforms.

Pitch frame (metres): x along the touchline from the left goal line (0) to the
right goal line (L, default 105); y across the pitch from the far touchline (0)
to the near touchline (W, default 68). Viewed from a sideline camera this frame
has the same handedness as image coordinates (x right, y down), so a correctly
ordered calibration never produces a mirrored mapping.

A calibration is four image points matched, in order, to four pitch points. The
usual case is a marked rectangle (penalty area, goal area, centre-circle box...)
whose dimensions the user enters; ``rectangle_pitch_points`` builds its corners
in the required cyclic order. Four known landmarks forming any convex
quadrilateral are also accepted.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

DEFAULT_PITCH = (105.0, 68.0)

# Standard IFAB markings, used by presets and held-out checks. Coordinates are for the
# LEFT end (x=0 goal line); mirror x -> L - x for the right end.
PRESET_RECTANGLES = {
    "penalty_area": {"width_m": 16.5, "height_m": 40.32, "label": "Penalty area (16.5 x 40.32 m)"},
    "goal_area": {"width_m": 5.5, "height_m": 18.32, "label": "Goal area (5.5 x 18.32 m)"},
}


def standard_landmarks(L: float = 105.0, W: float = 68.0) -> dict[str, tuple[float, float]]:
    c = W / 2
    lm = {}
    for end, gx, s in (("left", 0.0, 1.0), ("right", L, -1.0)):
        lm[f"{end}_goal_far_post"] = (gx, c - 3.66)
        lm[f"{end}_goal_near_post"] = (gx, c + 3.66)
        lm[f"{end}_goal_area_far_goalline"] = (gx, c - 9.16)
        lm[f"{end}_goal_area_near_goalline"] = (gx, c + 9.16)
        lm[f"{end}_goal_area_far_front"] = (gx + s * 5.5, c - 9.16)
        lm[f"{end}_goal_area_near_front"] = (gx + s * 5.5, c + 9.16)
        lm[f"{end}_penalty_area_far_goalline"] = (gx, c - 20.16)
        lm[f"{end}_penalty_area_near_goalline"] = (gx, c + 20.16)
        lm[f"{end}_penalty_area_far_front"] = (gx + s * 16.5, c - 20.16)
        lm[f"{end}_penalty_area_near_front"] = (gx + s * 16.5, c + 20.16)
        lm[f"{end}_penalty_spot"] = (gx + s * 11.0, c)
        lm[f"{end}_corner_far"] = (gx, 0.0)
        lm[f"{end}_corner_near"] = (gx, W)
    lm["halfway_far"] = (L / 2, 0.0)
    lm["halfway_near"] = (L / 2, W)
    lm["centre_spot"] = (L / 2, c)
    return lm


def rectangle_pitch_points(x0: float, y0: float, width_m: float, height_m: float) -> np.ndarray:
    """Corners in calibration order: (x0,y0) -> (x0+w,y0) -> (x0+w,y0+h) -> (x0,y0+h)."""
    return np.array([[x0, y0], [x0 + width_m, y0], [x0 + width_m, y0 + height_m], [x0, y0 + height_m]], np.float64)


@dataclass
class CalibrationReport:
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    homography: list[list[float]] | None = None  # image -> pitch
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def _signed_area(q: np.ndarray) -> float:
    x, y = q[:, 0], q[:, 1]
    return 0.5 * float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def _segments_cross(p1, p2, p3, p4) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, p3) * orient(p1, p2, p4) < 0) and (orient(p3, p4, p1) * orient(p3, p4, p2) < 0)


def quad_problems(q: np.ndarray, name: str, min_side: float, min_area: float) -> list[str]:
    """Degeneracy and ordering checks for an ordered quadrilateral."""
    errs = []
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return [f"{name}: exactly four finite points are required"]
    d = np.linalg.norm(q[:, None] - q[None], axis=2)[np.triu_indices(4, 1)]
    if d.min() < min_side:
        errs.append(f"{name}: two points are only {d.min():.1f} apart (min {min_side})")
    area = abs(_signed_area(q))
    if area < min_area:
        errs.append(f"{name}: quadrilateral area {area:.1f} is too small (min {min_area})")
    # Any three points nearly collinear -> homography is ill-conditioned.
    for i in range(4):
        tri = np.delete(q, i, axis=0)
        ta = abs(_signed_area(tri))
        if area > 0 and ta / area < 0.04:
            errs.append(f"{name}: three points are nearly collinear")
            break
    if _segments_cross(q[0], q[1], q[2], q[3]) or _segments_cross(q[1], q[2], q[3], q[0]):
        errs.append(f"{name}: edges cross (points are out of order; click them in the order shown)")
    else:
        cross = []
        for i in range(4):
            a, b, c = q[i], q[(i + 1) % 4], q[(i + 2) % 4]
            cross.append(np.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
        if len(set(cross)) > 1:
            errs.append(f"{name}: quadrilateral is not convex (check point order)")
    return errs


def to_pitch(H: np.ndarray, pts) -> np.ndarray:
    pts = np.asarray(pts, np.float64).reshape(-1, 1, 2)
    if len(pts) == 0:
        return np.zeros((0, 2))
    return cv2.perspectiveTransform(pts, np.asarray(H, np.float64)).reshape(-1, 2)


def to_image(H: np.ndarray, pts) -> np.ndarray:
    return to_pitch(np.linalg.inv(np.asarray(H, np.float64)), pts)


def homogeneous_w(H: np.ndarray, pts) -> np.ndarray:
    p = np.c_[np.asarray(pts, np.float64).reshape(-1, 2), np.ones(len(pts))]
    return p @ np.asarray(H, np.float64)[2]


def validate_calibration(
    image_points,
    pitch_points,
    frame_shape: tuple[int, int],
    field_mask: np.ndarray | None = None,
    pitch_size: tuple[float, float] = DEFAULT_PITCH,
    holdout: list[dict] | None = None,
) -> CalibrationReport:
    """Validate four ordered correspondences and return a report with the image->pitch homography.

    holdout: optional [{"name", "image": [u,v], "pitch": [x,y]}] landmarks NOT used for fitting,
    reported as reprojection error in metres.
    """
    img = np.asarray(image_points, np.float64)
    pit = np.asarray(pitch_points, np.float64)
    h, w = frame_shape[:2]
    rep = CalibrationReport(valid=False)
    rep.errors += quad_problems(img, "image points", min_side=12.0, min_area=0.002 * w * h)
    rep.errors += quad_problems(pit, "pitch points", min_side=1.0, min_area=4.0)
    if img.shape == (4, 2):
        if np.any(img[:, 0] < -0.05 * w) or np.any(img[:, 0] > 1.05 * w) or np.any(img[:, 1] < -0.05 * h) or np.any(img[:, 1] > 1.05 * h):
            rep.errors.append("image points: a point lies outside the frame")
    if rep.errors:
        return rep
    if np.sign(_signed_area(img)) != np.sign(_signed_area(pit)):
        rep.errors.append(
            "orientation: image points are in mirrored order relative to the pitch rectangle "
            "(go around the rectangle in the order shown: P1 -> P2 -> P3 -> P4)")
        return rep

    H, _ = cv2.findHomography(img.astype(np.float32), pit.astype(np.float32), 0)
    if H is None or not np.all(np.isfinite(H)):
        rep.errors.append("homography: OpenCV could not compute a finite homography")
        return rep
    H = H / H[2, 2]
    cond = float(np.linalg.cond(H))
    fit_err = float(np.max(np.linalg.norm(to_pitch(H, img) - pit, axis=1)))
    rep.metrics.update(condition_number=cond, fit_error_m=fit_err)

    # Plausibility 1: the vanishing line (image of the line at infinity) must not cut through the
    # calibration quad or the visible field, otherwise part of the "field" projects behind the camera.
    sample = img
    if field_mask is not None and np.count_nonzero(field_mask):
        ys, xs = np.nonzero(field_mask[::8, ::8])
        sample = np.c_[xs * 8, ys * 8].astype(np.float64)
    wq = homogeneous_w(H, img)
    ws = homogeneous_w(H, sample)
    sgn = np.sign(wq[0])
    if not np.all(np.sign(wq) == sgn):
        rep.errors.append("plausibility: horizon passes through the calibration rectangle")
        return rep
    behind = float(np.mean(np.sign(ws) != sgn))
    rep.metrics["field_behind_horizon_fraction"] = behind
    if behind > 0.02:
        rep.warnings.append(f"plausibility: {behind:.0%} of the visible grass maps beyond the horizon")

    # Plausibility 2: local scale (px per metre) at the quad centre, and its anisotropy.
    c = pit.mean(axis=0)
    du = to_image(H, [c + [0.5, 0], c - [0.5, 0], c + [0, 0.5], c - [0, 0.5]])
    sx = float(np.linalg.norm(du[0] - du[1]))
    sy = float(np.linalg.norm(du[2] - du[3]))
    rep.metrics.update(px_per_m_x=sx, px_per_m_y=sy)
    if min(sx, sy) < 0.3:
        rep.errors.append(f"plausibility: implied scale {min(sx, sy):.2f} px/m is implausibly small")
    if max(sx, sy) / max(min(sx, sy), 1e-6) > 40:
        rep.warnings.append("plausibility: extreme foreshortening; distances along one axis are unreliable")

    # Visibility: which part of the pitch the visible grass covers.
    L, W = pitch_size
    vis = visible_pitch_region(H, field_mask, frame_shape, pitch_size)
    rep.metrics.update(vis)
    if vis["visible_pitch_fraction"] < 0.05:
        rep.warnings.append("visibility: under 5% of the pitch is visible; team shape metrics will be withheld")
    if vis["visible_width_m"] < 0.6 * W:
        rep.warnings.append(
            f"visibility: only {vis['visible_width_m']:.0f} m of the {W:.0f} m pitch width is visible; "
            "team width is a partial-view measurement")

    if holdout:
        errs = []
        for hp in holdout:
            q = to_pitch(H, [hp["image"]])[0]
            errs.append({"name": hp.get("name", ""), "error_m": float(np.linalg.norm(q - np.asarray(hp["pitch"])))})
        rep.metrics["holdout"] = errs
        rep.metrics["holdout_rmse_m"] = float(np.sqrt(np.mean([e["error_m"] ** 2 for e in errs])))
        if rep.metrics["holdout_rmse_m"] > 3.0:
            rep.warnings.append(f"held-out landmarks reproject with RMSE {rep.metrics['holdout_rmse_m']:.1f} m")

    rep.homography = H.tolist()
    rep.metrics["calibration_quad_pitch"] = pit.tolist()
    rep.metrics["calibration_quad_image"] = img.tolist()
    rep.valid = not rep.errors
    return rep


def visible_pitch_region(H, field_mask, frame_shape, pitch_size=DEFAULT_PITCH) -> dict:
    L, W = pitch_size
    h, w = frame_shape[:2]
    if field_mask is not None and np.count_nonzero(field_mask):
        ys, xs = np.nonzero(field_mask[::10, ::10])
        pts = np.c_[xs * 10, ys * 10].astype(np.float64)
    else:
        gx, gy = np.meshgrid(np.linspace(0, w - 1, 64), np.linspace(h * 0.3, h - 1, 36))
        pts = np.c_[gx.ravel(), gy.ravel()]
    wv = homogeneous_w(H, pts)
    ref = np.sign(np.median(wv)) if len(wv) else 1.0
    pts = pts[np.sign(wv) == ref]
    P = to_pitch(H, pts) if len(pts) else np.zeros((0, 2))
    inside = (P[:, 0] >= 0) & (P[:, 0] <= L) & (P[:, 1] >= 0) & (P[:, 1] <= W) if len(P) else np.zeros(0, bool)
    Pi = P[inside]
    if len(Pi) == 0:
        return {"visible_pitch_fraction": 0.0, "visible_x_range_m": [0, 0], "visible_y_range_m": [0, 0],
                "visible_length_m": 0.0, "visible_width_m": 0.0}
    cells = np.zeros((34, 53), bool)
    cx = np.clip((Pi[:, 0] / L * 53).astype(int), 0, 52)
    cy = np.clip((Pi[:, 1] / W * 34).astype(int), 0, 33)
    cells[cy, cx] = True
    # Fill gaps between sampled rows (far-field samples are sparse after projection).
    cells = cv2.morphologyEx(cells.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)).astype(bool)
    xr = [float(np.percentile(Pi[:, 0], 1)), float(np.percentile(Pi[:, 0], 99))]
    yr = [float(np.percentile(Pi[:, 1], 1)), float(np.percentile(Pi[:, 1], 99))]
    return {"visible_pitch_fraction": float(cells.mean()), "visible_x_range_m": xr, "visible_y_range_m": yr,
            "visible_length_m": xr[1] - xr[0], "visible_width_m": yr[1] - yr[0]}


def distance_outside_polygon(points: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Distance (same units) from each point to a convex polygon; 0 inside."""
    poly32 = np.asarray(poly, np.float32).reshape(-1, 1, 2)
    out = np.zeros(len(points))
    for i, p in enumerate(np.asarray(points, np.float64)):
        d = cv2.pointPolygonTest(poly32, (float(p[0]), float(p[1])), True)
        out[i] = max(0.0, -d)
    return out
