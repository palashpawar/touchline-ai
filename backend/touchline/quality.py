"""Quality signals computed from a SegmentResult.

Every signal is a measured quantity with units and a threshold. None of them is a
probability; they are heuristic indicators whose thresholds were set by inspecting the
development clip and the synthetic stress tests (see docs/QUALITY_SIGNALS.md for the
rationale and known failure modes). ``None`` means "not measurable for this segment"
(for example, metric signals without a valid calibration).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

MAX_PLAYER_SPEED_MS = 12.0  # faster than an elite sprint (~10.5 m/s): treated as a projection/ID error
SHORT_TRACK_S = 1.0


@dataclass(frozen=True)
class SignalSpec:
    key: str
    label: str
    unit: str
    threshold: float
    bad_when: str  # "above" | "below"
    definition: str
    limitations: str


SIGNALS: dict[str, SignalSpec] = {s.key: s for s in [
    SignalSpec("calibration_valid", "Calibration valid", "bool", 1, "below",
               "Four-point calibration passed degeneracy, ordering, horizon and scale checks.",
               "A valid homography can still be inaccurate far from the calibrated rectangle."),
    SignalSpec("camera_drift_px", "Camera drift vs calibration frame", "px", 12.0, "above",
               "Max displacement of the calibration rectangle's corners between the reference frame and any "
               "sampled frame, from the validated frame-to-reference homography.",
               "Only measurable when the motion model validates; zoom changes appear as drift too."),
    SignalSpec("line_alignment", "Pitch-line alignment", "fraction", 0.60, "below",
               "Median over sampled frames of the share of detected pitch-line pixels that fall within 4 px of "
               "a reference-frame line pixel after mapping with the geometry used for projection.",
               "White kit, shadows and line wear add noise; needs >=200 visible line pixels per frame."),
    SignalSpec("motion_inlier_ratio", "Camera-motion inlier ratio", "fraction", 0.45, "below",
               "Median MAGSAC inlier ratio of the frame-to-reference homography over sampled frames.",
               "High ratios on a near-static camera are uninformative; crowd motion lowers it."),
    SignalSpec("motion_reproj_px", "Camera-motion reprojection error", "px", 2.5, "above",
               "Median (over frames) of the median inlier reprojection error of the motion homography.",
               "Measured at half resolution and rescaled."),
    SignalSpec("motion_validated_fraction", "Frames with validated motion", "fraction", 0.80, "below",
               "Share of sampled frames whose motion estimate (to the calibration frame, or chained) met inlier-count, "
               "ratio, error and sanity limits. Low = the calibration can no longer be verified for this segment.",
               "A static camera over a texture-poor scene can also fail to validate; the agent then asks rather than assumes."),
    SignalSpec("out_of_pitch_fraction", "Projected locations outside pitch", "fraction", 0.12, "above",
               "Share of detections whose foot is well inside the grass region (>=20 px from its edge) but whose "
               "projected ground point falls >3 m outside the pitch.",
               "Assistant referees stand outside the touchline legitimately; grass-mask leakage can still admit "
               "edge spectators. Rises when calibration or camera geometry is wrong."),
    SignalSpec("implausible_jump_rate", "Implausible projected jumps", "fraction", 0.08, "above",
               f"Share of consecutive same-track observations implying a ground speed >{MAX_PLAYER_SPEED_MS:.0f} m/s.",
               "Far-field projection noise can produce false jumps; needs valid calibration."),
    SignalSpec("short_track_ratio", "Short (fragmented) tracks", "fraction", 0.30, "above",
               f"Share of tracks spanning < {SHORT_TRACK_S:.0f} s of the segment.",
               "Players entering/leaving the view legitimately create short tracks."),
    SignalSpec("fragmentation_index", "Track fragmentation index", "tracks per visible player", 1.60, "above",
               "Number of tracks divided by the median number of kept detections per sampled frame. ~1.0 means one "
               "track per visible player.",
               "Inflated by view changes; deflated if the detector misses players consistently."),
    SignalSpec("assoc_residual_h", "Association residual", "box heights", 0.60, "above",
               "Median distance between a track's Kalman prediction and its matched detection, in box heights.",
               "Low values do not prove correct identity: two players can swap with low residual."),
    SignalSpec("detection_gap_fraction", "Detection gaps", "fraction", 0.15, "above",
               "Share of sampled frames with fewer than half the segment's median kept detections.",
               "Occlusion by spectators in front of the camera is a natural cause the agent cannot fix."),
    SignalSpec("extrapolated_fraction", "Points far from calibration", "fraction", 0.50, "above",
               "Share of projected player points more than 30 m from the calibrated rectangle (informational).",
               "Homography error grows with distance from the calibrated area."),
]}

TRIGGERS = ("camera_drift_px", "line_alignment", "motion_validated_fraction", "out_of_pitch_fraction", "implausible_jump_rate",
            "short_track_ratio", "fragmentation_index", "assoc_residual_h", "detection_gap_fraction")


def _breach(key: str, value) -> bool:
    if value is None:
        return False
    spec = SIGNALS[key]
    return value > spec.threshold if spec.bad_when == "above" else value < spec.threshold


@dataclass
class QualityReport:
    signals: dict
    breaches: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def assess(seg: dict, ctx_quad_pitch: np.ndarray | None, calibration_valid: bool) -> QualityReport:
    """seg: SegmentResult.to_dict()."""
    frames = seg["frames"]
    tracks = seg["tracks"]
    sig: dict = {"calibration_valid": 1.0 if calibration_valid else 0.0}

    drift = [f["drift_px"] for f in frames if f["drift_px"] is not None]
    sig["camera_drift_px"] = float(max(drift)) if drift else None
    al = [f["line_alignment"] for f in frames if f["line_alignment"] is not None]
    sig["line_alignment"] = float(np.median(al)) if al else None
    mo = [f["motion"] for f in frames]
    sig["motion_inlier_ratio"] = float(np.median([m["inlier_ratio"] for m in mo])) if mo else None
    errs = [m["reproj_error_px"] for m in mo if np.isfinite(m["reproj_error_px"])]
    sig["motion_reproj_px"] = float(np.median(errs)) if errs else None
    sig["motion_validated_fraction"] = float(np.mean([m["validated"] for m in mo])) if mo else None

    interior = sum(f.get("n_interior", f["n_on_grass"]) for f in frames)
    out_int = sum(f.get("n_out_of_pitch_interior", f["n_out_of_pitch"]) for f in frames)
    sig["out_of_pitch_fraction"] = (out_int / interior) if (interior and calibration_valid) else None

    jumps = pairs = 0
    pts_all = []
    spans = []
    for tr in tracks:
        obs = tr["obs"]
        spans.append(obs[-1]["t"] - obs[0]["t"])
        for a, b in zip(obs, obs[1:]):
            if a["pitch"] is None or b["pitch"] is None:
                continue
            dt = b["t"] - a["t"]
            if dt <= 0:
                continue
            pairs += 1
            if np.linalg.norm(np.subtract(b["pitch"], a["pitch"])) / dt > MAX_PLAYER_SPEED_MS:
                jumps += 1
        pts_all += [o["pitch"] for o in obs if o["pitch"] is not None]
    sig["implausible_jump_rate"] = (jumps / pairs) if pairs else None
    sig["short_track_ratio"] = float(np.mean([s < SHORT_TRACK_S for s in spans])) if spans else None
    kept = [f["n_kept"] for f in frames]
    med = float(np.median(kept)) if kept else 0.0
    sig["fragmentation_index"] = (len(tracks) / med) if med >= 1 else None
    sig["assoc_residual_h"] = seg["tracker_stats"].get("assoc_residual_median_h")
    sig["detection_gap_fraction"] = float(np.mean([k < 0.5 * med for k in kept])) if med >= 1 else 1.0
    if ctx_quad_pitch is not None and len(ctx_quad_pitch) == 4 and pts_all:
        from .vision.calibration import distance_outside_polygon
        d = distance_outside_polygon(np.asarray(pts_all), ctx_quad_pitch)
        sig["extrapolated_fraction"] = float(np.mean(d > 30.0))
    else:
        sig["extrapolated_fraction"] = None
    sig["mean_kept_detections"] = float(np.mean(kept)) if kept else 0.0
    sig["sampled_frames"] = len(frames)
    sig["stride"] = seg["config"]["stride"]
    sig["motion_compensated"] = bool(seg["config"]["motion_compensation"])

    breaches = [k for k in TRIGGERS if _breach(k, sig.get(k))]
    if sig["motion_compensated"] and "camera_drift_px" in breaches:
        # Drift is expected once the geometry follows the camera; line_alignment checks the result instead.
        breaches.remove("camera_drift_px")
    if not calibration_valid:
        breaches.insert(0, "calibration_valid")
    return QualityReport(sig, breaches)


def spec_table() -> list[dict]:
    return [asdict(s) for s in SIGNALS.values()]
