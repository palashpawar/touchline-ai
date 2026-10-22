"""Rule-based recovery policy and run comparison.

Pure functions of measured quality signals: no learned model, no LLM. The same inputs always
give the same plan, which makes every decision in the trace reproducible and testable.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from ..quality import SIGNALS

MAX_ATTEMPTS = 2
STRIDE_SCHEDULE = (6, 3, 2, 1)  # every stride divides the default (6) so original frames stay in the denser run
CAMERA_SIGNALS = ("camera_drift_px", "line_alignment", "motion_validated_fraction")
TRACKING_SIGNALS = ("short_track_ratio", "fragmentation_index", "assoc_residual_h", "implausible_jump_rate",
                    "detection_gap_fraction")
MIN_REL_IMPROVEMENT = 0.10


@dataclass
class Plan:
    action: str  # accept | camera_motion_update | densify_sampling | request_human_review
    reason: str
    target_signals: list[str] = field(default_factory=list)
    proposed_config: dict | None = None
    review_kind: str | None = None  # recalibrate | approve_rerun | inspect

    def to_dict(self) -> dict:
        return asdict(self)


def next_stride(stride: int) -> int | None:
    denser = [s for s in STRIDE_SCHEDULE if s < stride]
    return denser[0] if denser else None


def motion_model_ok(sig: dict) -> bool:
    return (sig.get("motion_validated_fraction") or 0) >= SIGNALS["motion_validated_fraction"].threshold and \
        (sig.get("motion_inlier_ratio") or 0) >= SIGNALS["motion_inlier_ratio"].threshold and \
        (sig.get("motion_reproj_px") if sig.get("motion_reproj_px") is not None else 99) <= SIGNALS["motion_reproj_px"].threshold


def plan_recovery(signals: dict, breaches: list[str], config: dict, attempts_used: int) -> Plan:
    if "calibration_valid" in breaches:
        return Plan("request_human_review", "Calibration is invalid; metric outputs are withheld until the field "
                    "points are corrected.", review_kind="recalibrate")
    if not breaches:
        return Plan("accept", "All trigger signals are within thresholds.")

    def fmt(keys):
        return ", ".join(f"{k}={_fmt(signals.get(k))} (threshold {SIGNALS[k].threshold})" for k in keys)

    cam = [k for k in CAMERA_SIGNALS if k in breaches]
    trk = [k for k in TRACKING_SIGNALS if k in breaches]

    if attempts_used >= MAX_ATTEMPTS:
        nxt = next_stride(config["stride"])
        proposal = dict(config, stride=nxt) if (trk and nxt) else None
        return Plan("request_human_review",
                    f"Automatic retry limit ({MAX_ATTEMPTS}) reached with breaches remaining: {fmt(breaches)}.",
                    target_signals=breaches, proposed_config=proposal,
                    review_kind="approve_rerun" if proposal else "recalibrate")

    if cam and not config.get("motion_compensation"):
        if motion_model_ok(signals):
            return Plan("camera_motion_update",
                        f"Camera moved away from the calibration frame ({fmt(cam)}) and the motion model is validated "
                        f"(inlier ratio {_fmt(signals.get('motion_inlier_ratio'))}, reprojection "
                        f"{_fmt(signals.get('motion_reproj_px'))} px). Re-estimate motion and update calibration.",
                        # Geometric evidence only: tracking signals (e.g. jump rate) are re-assessed after the fix
                        # and handled by densify_sampling, rather than vetoing a validated geometry correction.
                        target_signals=["line_alignment", "out_of_pitch_fraction"],
                        proposed_config=dict(config, motion_compensation=True))
        return Plan("request_human_review",
                    f"Camera geometry changed or cannot be verified ({fmt(cam)}) and the motion model is not validated "
                    f"(validated frames {_fmt(signals.get('motion_validated_fraction'))}); calibration cannot be "
                    "carried forward automatically.", target_signals=cam, review_kind="recalibrate")

    if cam and config.get("motion_compensation"):
        return Plan("request_human_review",
                    f"Geometry check still failing after motion compensation ({fmt(cam)}); the calibration itself is "
                    "probably off for this view.", target_signals=cam, review_kind="recalibrate")

    if trk:
        nxt = next_stride(config["stride"])
        if nxt is None:
            return Plan("request_human_review", f"Tracking breaches ({fmt(trk)}) at full frame rate; denser sampling "
                        "is not possible.", target_signals=trk, review_kind="inspect")
        return Plan("densify_sampling",
                    f"Tracking quality breaches ({fmt(trk)}) at stride {config['stride']}; rerun at stride {nxt} so "
                    "players move less between samples.",
                    target_signals=trk, proposed_config=dict(config, stride=nxt))

    if "out_of_pitch_fraction" in breaches:
        return Plan("request_human_review", f"Many projections fall off the pitch ({fmt(['out_of_pitch_fraction'])}) "
                    "with camera motion already handled; check calibration points.",
                    target_signals=["out_of_pitch_fraction"], review_kind="recalibrate")
    return Plan("request_human_review", f"Unhandled breaches: {fmt(breaches)}", target_signals=breaches,
                review_kind="inspect")


def _fmt(v):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.3g}"
    return str(v)


def _breaches(key: str, value) -> bool:
    if value is None:
        return False
    spec = SIGNALS[key]
    return value > spec.threshold if spec.bad_when == "above" else value < spec.threshold


def _better(key: str, before, after) -> float | None:
    """Signed relative improvement (>0 is better), or None if not comparable."""
    if before is None or after is None:
        return None
    spec = SIGNALS[key]
    if spec.bad_when == "above":
        delta = before - after
    else:
        delta = after - before
    scale = max(abs(before), 1e-3)
    return delta / scale


def compare_signals(before: dict, after: dict, target_signals: list[str]) -> dict:
    """Decide whether a corrected run should replace the previous one.

    Accept when at least one target signal improves by >=10% (relative), no target signal gets >=10% worse
    while ending beyond its threshold, and no guard signal (line alignment, off-pitch share, detections per
    frame) degrades materially.
    """
    rows = {}
    improved, worsened = [], []
    for k in target_signals:
        r = _better(k, before.get(k), after.get(k))
        rows[k] = {"before": before.get(k), "after": after.get(k), "relative_improvement": r}
        if r is None:
            continue
        after_breaches = _breaches(k, after.get(k))
        rows[k]["after_breaches_threshold"] = after_breaches
        if r >= MIN_REL_IMPROVEMENT:
            improved.append(k)
        elif r <= -MIN_REL_IMPROVEMENT and after_breaches:
            # Movement that stays inside the acceptable range is reported but is not a regression.
            worsened.append(k)
    guards = []
    la_b, la_a = before.get("line_alignment"), after.get("line_alignment")
    if la_b is not None and la_a is not None and la_a < la_b - 0.05:
        guards.append("line_alignment dropped")
    op_b, op_a = before.get("out_of_pitch_fraction"), after.get("out_of_pitch_fraction")
    if op_b is not None and op_a is not None and op_a > op_b + 0.05:
        guards.append("out_of_pitch_fraction rose")
    kb, ka = before.get("mean_kept_detections") or 0, after.get("mean_kept_detections") or 0
    if kb > 0 and ka < 0.8 * kb:
        guards.append("detections per frame fell >20%")
    accepted = bool(improved) and not worsened and not guards
    if accepted:
        reason = f"Improved {', '.join(improved)}; no target or guard signal degraded."
    elif not improved:
        reason = "No target signal improved by at least 10%."
    else:
        reason = "Rejected: " + "; ".join([f"{k} worsened" for k in worsened] + guards)
    return {"accepted": accepted, "reason": reason, "signals": rows, "improved": improved,
            "worsened": worsened, "guard_failures": guards}
