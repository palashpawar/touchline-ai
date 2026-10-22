"""Segment analysis: detection -> filtering -> (optional) motion compensation -> projection -> tracking.

`analyze_segment` is the unit the agent calls (and re-calls with a different configuration).
It never mutates the run context; it returns a `SegmentResult` that the agent can compare
against a previous result before deciding which one to keep.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .vision import calibration as cal
from .vision import field as fieldmod
from .vision import motion as motionmod
from .vision.detector import YoloxPersonDetector, filter_players
from .vision.teams import torso_lab
from .vision.tracker import Tracker, torso_hist
from .vision.video import VideoMeta, read_frames, sample_indices

PITCH_MARGIN_M = 3.0
LINE_TOL_PX = 4.0
INTERIOR_ERODE_PX = 41


@dataclass
class SegmentConfig:
    stride: int = 6
    motion_compensation: bool = False
    # Per-frame image->reference homographies supplied by update_calibration (frame index as str).
    motion_model: dict | None = None
    detector_conf: float = 0.30

    def to_dict(self) -> dict:
        d = asdict(self)
        if d.get("motion_model"):
            d["motion_model"] = {"frames": len(d["motion_model"]), "source": "update_calibration"}
        return d


@dataclass
class SegmentResult:
    segment_id: str
    start: int
    end: int
    config: dict
    frames: list[dict] = field(default_factory=list)
    tracks: list[dict] = field(default_factory=list)
    tracker_stats: dict = field(default_factory=dict)
    elapsed_s: float = 0.0
    attempt: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


class RunContext:
    """Everything shared by all segments of one run: video, calibration, detector, references."""

    def __init__(self, video_path: Path, meta: VideoMeta, calibration: dict, detector: YoloxPersonDetector):
        self.video_path = Path(video_path)
        self.meta = meta
        self.calibration = calibration
        self.valid_calibration = bool(calibration.get("valid"))
        self.H_ref = np.asarray(calibration["homography"], np.float64) if self.valid_calibration else None
        self.ref_frame_idx = int(calibration.get("frame_index", 0))
        self.quad_img = np.asarray(calibration.get("image_points", []), np.float64)
        self.quad_pitch = np.asarray(calibration.get("pitch_points", []), np.float64)
        self.pitch_size = tuple(calibration.get("pitch_size", cal.DEFAULT_PITCH))
        self.detector = detector
        ref = next(f for _, f in read_frames(video_path, [self.ref_frame_idx]))
        self.ref_frame = ref
        fm = fieldmod.field_mask(ref)
        roi = fieldmod.field_bbox(fm)
        ref_dets, _ = filter_players(detector.detect(ref, roi), fm, frame=ref)
        self.ref_boxes = [d.box for d in ref_dets]
        self.matcher = motionmod.ReferenceMatcher(ref, self.ref_boxes)
        lines = fieldmod.white_line_mask(ref, fm)
        for b in self.ref_boxes:
            x1, y1, x2, y2 = b.astype(int)
            cv2.rectangle(lines, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
        self.ref_lines = lines
        self.ref_line_dt = cv2.distanceTransform(cv2.bitwise_not(lines), cv2.DIST_L2, 3)

    def segments(self, seconds: float = 4.0) -> list[tuple[str, int, int]]:
        n = self.meta.frame_count
        step = max(1, int(round(seconds * self.meta.fps)))
        out = []
        for k, s in enumerate(range(0, n, step)):
            e = min(n, s + step)
            if e - s < step // 3 and out:  # merge a short tail into the previous segment
                sid, ps, _ = out[-1]
                out[-1] = (sid, ps, e)
            else:
                out.append((f"seg{k + 1}", s, e))
        return out


def line_alignment(ctx: RunContext, frame: np.ndarray, fmask: np.ndarray, boxes, M_to_ref: np.ndarray | None) -> float | None:
    """Share of the frame's pitch-line pixels that land within LINE_TOL_PX of a reference-frame line
    pixel after mapping into the reference frame (identity when M_to_ref is None).

    Independent of player tracking: it checks whether the geometry used for projection still
    matches the painted lines. 1.0 = perfect overlap. None when too few line pixels are visible.
    """
    lines = fieldmod.white_line_mask(frame, fmask)
    for b in boxes:
        x1, y1, x2, y2 = np.asarray(b).astype(int)
        cv2.rectangle(lines, (x1 - 3, y1 - 3), (x2 + 3, y2 + 3), 0, -1)
    if M_to_ref is not None:
        lines = cv2.warpPerspective(lines, np.asarray(M_to_ref, np.float64), (lines.shape[1], lines.shape[0]),
                                    flags=cv2.INTER_NEAREST)
    ys, xs = np.nonzero(lines)
    if len(xs) < 200 or np.count_nonzero(ctx.ref_lines) < 200:
        return None
    return float(np.mean(ctx.ref_line_dt[ys, xs] <= LINE_TOL_PX))


def frame_homography(ctx: RunContext, M_to_ref: np.ndarray | None) -> np.ndarray | None:
    """Image->pitch homography for a frame: H_ref composed with the frame->reference motion."""
    if ctx.H_ref is None:
        return None
    return ctx.H_ref if M_to_ref is None else ctx.H_ref @ np.asarray(M_to_ref, np.float64)


def analyze_segment(ctx: RunContext, segment_id: str, start: int, end: int, config: SegmentConfig,
                    attempt: int = 0, progress=None) -> SegmentResult:
    t0 = time.time()
    fps = ctx.meta.fps
    L, W = ctx.pitch_size
    idx = sample_indices(start, end, config.stride)
    res = SegmentResult(segment_id, start, end, config.to_dict(), attempt=attempt)
    tracker = Tracker()
    last_frame, last_boxes, last_M = None, None, None
    for k, (fi, frame) in enumerate(read_frames(ctx.video_path, idx)):
        fm = fieldmod.field_mask(frame)
        roi = fieldmod.field_bbox(fm)
        raw = ctx.detector.detect(frame, roi) if roi else []
        on_grass, reasons = filter_players(raw, fm, frame=frame)
        boxes = [d.box for d in on_grass]

        # Camera motion relative to the reference frame (always measured; only *used* when compensating).
        est = ctx.matcher.match_to_reference(frame, boxes)
        if not est.validated and last_frame is not None and last_M is not None:
            pair = motionmod.track_pair(last_frame, frame, last_boxes)
            if pair.validated:
                chained = np.asarray(last_M) @ np.asarray(pair.M)
                est = motionmod.MotionEstimate(chained.tolist(), "lk_chain", pair.matches, pair.inliers,
                                               pair.inlier_ratio, pair.reproj_error_px, True)
        M_meas = np.asarray(est.M) if est.validated else None
        drift = motionmod.quad_drift_px(M_meas, ctx.quad_img) if (M_meas is not None and len(ctx.quad_img) == 4) else None

        M_use = None
        if config.motion_compensation:
            if config.motion_model and str(fi) in config.motion_model:
                M_use = np.asarray(config.motion_model[str(fi)], np.float64)
            elif M_meas is not None:
                M_use = M_meas
        H_t = frame_homography(ctx, M_use)
        align = line_alignment(ctx, frame, fm, boxes, M_use)

        dets, n_out, n_int, n_out_int = [], 0, 0, 0
        # Interior = foot well inside the grass region. Used for the off-pitch quality signal so that people at
        # the grass edge (photographers, stewards) that the projection test correctly removes do not count as
        # calibration errors.
        interior = cv2.erode(fm, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (INTERIOR_ERODE_PX, INTERIOR_ERODE_PX)))
        for d in on_grass:
            fx, fy = int(np.clip(d.foot[0], 0, fm.shape[1] - 1)), int(np.clip(d.foot[1] - 2, 0, fm.shape[0] - 1))
            is_int = interior[fy, fx] > 0
            n_int += is_int
            foot = d.foot
            foot_ref = foot if M_use is None else cv2.perspectiveTransform(
                foot.reshape(1, 1, 2).astype(np.float64), M_use).reshape(2)
            pitch = None
            if H_t is not None:
                wv = cal.homogeneous_w(H_t, [foot])[0]
                p = cal.to_pitch(H_t, [foot])[0]
                inside = (wv * cal.homogeneous_w(ctx.H_ref, [ctx.quad_img[0]])[0] > 0) and \
                    -PITCH_MARGIN_M <= p[0] <= L + PITCH_MARGIN_M and -PITCH_MARGIN_M <= p[1] <= W + PITCH_MARGIN_M
                if not inside:
                    n_out += 1
                    n_out_int += is_int
                    continue  # projected off the pitch: excluded from tracking and metrics
                pitch = p
            dets.append({"box": d.box, "foot_ref": foot_ref, "score": d.score, "height": d.height,
                         "hist": torso_hist(frame, d.box), "team_feat": torso_lab(frame, d.box),
                         "pitch": pitch})
        tracker.step(fi, fi / fps, dets)
        res.frames.append({
            "frame": fi, "t": fi / fps, "n_raw": len(raw), "n_on_grass": len(on_grass), "n_kept": len(dets),
            "n_out_of_pitch": n_out,
            "n_interior": int(n_int), "n_out_of_pitch_interior": int(n_out_int), "rejected": reasons,
            "motion": {k2: v for k2, v in est.to_dict().items() if k2 != "M"},
            "M_to_ref": est.M if est.validated else None,
            "drift_px": drift, "line_alignment": align, "compensated": M_use is not None,
            "H": None if H_t is None else np.asarray(H_t).tolist(),
        })
        last_frame, last_boxes = frame, boxes
        last_M = M_meas if M_meas is not None else last_M
        if progress:
            progress((k + 1) / len(idx))
    for tr in tracker.finished():
        res.tracks.append({"tid": f"{segment_id}-{attempt}-{tr.tid}", "obs": [asdict(o) for o in tr.obs],
                           "coasted_frames": tr.coasted_frames})
    st = dict(tracker.stats)
    r = st.pop("assoc_residual_h")
    st["assoc_residual_median_h"] = float(np.median(r)) if r else None
    res.tracker_stats = st
    res.elapsed_s = time.time() - t0
    return res
