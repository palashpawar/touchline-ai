import json

import pytest

from touchline.agent import policy
from touchline.agent.trace import AgentStep, Trace

GOOD_MOTION = {"motion_validated_fraction": 1.0, "motion_inlier_ratio": 0.9, "motion_reproj_px": 1.5}
CFG = {"stride": 6, "motion_compensation": False, "motion_model": None, "detector_conf": 0.3}


def sig(**kw):
    base = {"calibration_valid": 1.0, "camera_drift_px": 3.0, "line_alignment": 0.9, "out_of_pitch_fraction": 0.02,
            "implausible_jump_rate": 0.01, "short_track_ratio": 0.1, "fragmentation_index": 1.1,
            "assoc_residual_h": 0.1, "detection_gap_fraction": 0.0, "mean_kept_detections": 20.0, **GOOD_MOTION}
    base.update(kw)
    return base


def test_no_breaches_accepts():
    assert policy.plan_recovery(sig(), [], CFG, 0).action == "accept"


def test_invalid_calibration_requests_recalibration():
    p = policy.plan_recovery(sig(calibration_valid=0.0), ["calibration_valid"], CFG, 0)
    assert p.action == "request_human_review" and p.review_kind == "recalibrate"


def test_camera_drift_with_validated_motion_triggers_calibration_update():
    s = sig(camera_drift_px=80, line_alignment=0.2)
    p = policy.plan_recovery(s, ["camera_drift_px", "line_alignment"], CFG, 0)
    assert p.action == "camera_motion_update"
    assert p.proposed_config["motion_compensation"] is True
    assert "line_alignment" in p.target_signals


def test_camera_drift_with_unvalidated_motion_escalates():
    s = sig(camera_drift_px=80, line_alignment=0.2, motion_validated_fraction=0.3, motion_inlier_ratio=0.2)
    p = policy.plan_recovery(s, ["camera_drift_px", "line_alignment"], CFG, 0)
    assert p.action == "request_human_review" and p.review_kind == "recalibrate"


def test_misalignment_after_compensation_escalates_not_loops():
    cfg = dict(CFG, motion_compensation=True)
    p = policy.plan_recovery(sig(line_alignment=0.3), ["line_alignment"], cfg, 1)
    assert p.action == "request_human_review"


def test_tracking_breach_densifies_sampling():
    p = policy.plan_recovery(sig(short_track_ratio=0.5), ["short_track_ratio"], CFG, 0)
    assert p.action == "densify_sampling" and p.proposed_config["stride"] == 3


def test_stride_schedule_keeps_original_frames():
    # Every stride divides the default (6), so the initial run's frames are present in any denser rerun.
    assert all(policy.STRIDE_SCHEDULE[0] % s == 0 for s in policy.STRIDE_SCHEDULE)
    assert policy.next_stride(6) == 3 and policy.next_stride(3) == 2 and policy.next_stride(1) is None


def test_tracking_breach_at_full_rate_escalates():
    p = policy.plan_recovery(sig(short_track_ratio=0.5), ["short_track_ratio"], dict(CFG, stride=1), 0)
    assert p.action == "request_human_review"


@pytest.mark.parametrize("attempts", [2, 3, 10])
def test_retry_limit_escalates_with_proposal(attempts):
    p = policy.plan_recovery(sig(short_track_ratio=0.5), ["short_track_ratio"], dict(CFG, stride=3), attempts)
    assert p.action == "request_human_review"
    assert p.review_kind == "approve_rerun" and p.proposed_config["stride"] == 2


def test_compare_accepts_improvement():
    b = sig(line_alignment=0.2, out_of_pitch_fraction=0.08)
    a = sig(line_alignment=0.8, out_of_pitch_fraction=0.05)
    r = policy.compare_signals(b, a, ["line_alignment", "out_of_pitch_fraction"])
    assert r["accepted"] and "line_alignment" in r["improved"]


def test_compare_rejects_no_improvement():
    r = policy.compare_signals(sig(short_track_ratio=0.40), sig(short_track_ratio=0.39), ["short_track_ratio"])
    assert not r["accepted"]


def test_compare_rejects_target_that_worsens_past_threshold():
    b = sig(short_track_ratio=0.5, fragmentation_index=1.5)
    a = sig(short_track_ratio=0.3, fragmentation_index=2.5)
    r = policy.compare_signals(b, a, ["short_track_ratio", "fragmentation_index"])
    assert not r["accepted"] and "fragmentation_index" in r["worsened"]


def test_compare_tolerates_change_within_acceptable_range():
    b = sig(line_alignment=0.2, implausible_jump_rate=0.03)
    a = sig(line_alignment=0.8, implausible_jump_rate=0.055)  # worse, but still under the 0.08 threshold
    r = policy.compare_signals(b, a, ["line_alignment", "implausible_jump_rate"])
    assert r["accepted"]


def test_compare_guard_rejects_detection_collapse():
    b = sig(short_track_ratio=0.5, mean_kept_detections=20)
    a = sig(short_track_ratio=0.2, mean_kept_detections=10)
    r = policy.compare_signals(b, a, ["short_track_ratio"])
    assert not r["accepted"] and r["guard_failures"]


def test_trace_roundtrip_and_required_fields():
    tr = Trace("run1")
    tr.log(state="PLAN_RECOVERY", segment_id="seg1", decision="camera_motion_update", reason="drift",
           tool="estimate_camera_motion", tool_args={"stride": 6}, observed_signals={"camera_drift_px": 80.0},
           previous_config=CFG, proposed_config=dict(CFG, motion_compensation=True),
           quality_before={"signals": {"line_alignment": 0.2}, "breaches": ["line_alignment"]},
           quality_after={"signals": {"line_alignment": 0.8}, "breaches": []}, outcome="ok", accepted=True)
    text = tr.to_jsonl()
    row = json.loads(text.splitlines()[0])
    for k in ("timestamp", "run_id", "segment_id", "observed_signals", "decision", "reason", "tool", "tool_args",
              "previous_config", "proposed_config", "quality_before", "quality_after", "outcome", "accepted"):
        assert k in row
    back = Trace.from_jsonl(text)
    assert back.steps[0].to_dict() == tr.steps[0].to_dict()


def test_unknown_state_rejected():
    with pytest.raises(ValueError):
        AgentStep(run_id="r", segment_id=None, state="DANCE", decision="x", reason="y")
