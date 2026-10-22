"""State-machine tests with a scripted toolbox: retry limits, acceptance/rejection, escalation."""
from types import SimpleNamespace

from touchline.agent.agent import SegmentAgent
from touchline.agent.trace import Trace
from touchline.pipeline import SegmentConfig
from touchline.agent import policy

BASE = {"calibration_valid": 1.0, "camera_drift_px": 2.0, "line_alignment": 0.9, "out_of_pitch_fraction": 0.02,
        "implausible_jump_rate": 0.01, "short_track_ratio": 0.1, "fragmentation_index": 1.1, "assoc_residual_h": 0.1,
        "detection_gap_fraction": 0.0, "mean_kept_detections": 20.0, "motion_validated_fraction": 1.0,
        "motion_inlier_ratio": 0.9, "motion_reproj_px": 1.2}


def q(**kw):
    s = dict(BASE, **kw)
    breaches = [k for k, thr, above in [("camera_drift_px", 12, True), ("line_alignment", 0.6, False),
                                        ("short_track_ratio", 0.3, True)]
                if (s[k] > thr if above else s[k] < thr)]
    if s.get("motion_compensated") and "camera_drift_px" in breaches:
        breaches.remove("camera_drift_px")
    return {"signals": s, "breaches": breaches}


class FakeTools:
    """Returns scripted quality reports per (attempt) and records every call."""

    def __init__(self, qualities, compare_ok=True, update_ok=True):
        self.qualities = qualities  # list indexed by attempt
        self.compare_ok = compare_ok
        self.update_ok = update_ok
        self.results, self.reviews, self.calls = {}, [], []

    def analyze_segment(self, sid, s, e, cfg, attempt=0):
        self.calls.append(("analyze", attempt, cfg.stride, cfg.motion_compensation))
        rid = f"{sid}.a{attempt}.x"
        self.results[rid] = {"config": cfg.to_dict(), "attempt": attempt}
        return {"result_id": rid, "tracks": 10, "sampled_frames": 20, "elapsed_s": 0.1}

    rerun_segment = analyze_segment

    def assess_tracking_quality(self, rid, frames=None):
        return self.qualities[min(int(rid.split(".a")[1].split(".")[0]), len(self.qualities) - 1)]

    def estimate_camera_motion(self, sid, s, e, stride):
        self.calls.append(("estimate_motion", stride))
        return {"motion_id": "m1", "stats": {"validated_fraction": 1.0}}

    def update_calibration(self, mid):
        self.calls.append(("update_calibration",))
        return {"accepted": self.update_ok, "reason": "ok" if self.update_ok else "alignment did not improve",
                "motion_model": {"0": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]} if self.update_ok else None, "probe": []}

    def compare_runs(self, before, after, targets):
        self.calls.append(("compare", before, after))
        qb, qa = self.assess_tracking_quality(before), self.assess_tracking_quality(after)
        res = policy.compare_signals(qb["signals"], qa["signals"], targets)
        if not self.compare_ok:
            res.update(accepted=False, reason="forced rejection")
        res.update(evaluation_frames=20, quality_before=qb, quality_after=qa)
        return res

    def request_human_review(self, sid, reason, kind, proposed, rid=None):
        self.reviews.append({"review_id": f"review-{len(self.reviews) + 1}", "segment_id": sid, "kind": kind,
                             "proposed_config": proposed})
        return {"review_id": self.reviews[-1]["review_id"]}


def make_agent(tools):
    meta = SimpleNamespace(width=1920, height=1080, fps=30.0, frame_count=120, duration_s=4.0,
                           to_dict=lambda: {"fps": 30.0})
    ctx = SimpleNamespace(meta=meta, calibration={"valid": True, "metrics": {}},
                          segments=lambda seconds=4.0: [("seg1", 0, 120)])
    a = SegmentAgent(ctx, Trace("t"), SegmentConfig(stride=6))
    a.tools = tools
    return a


def test_retry_limit_is_two_and_escalates():
    # Tracking stays bad but improves each time -> accepted twice, then retry budget is exhausted.
    tools = FakeTools([q(short_track_ratio=0.6), q(short_track_ratio=0.5), q(short_track_ratio=0.4)])
    a = make_agent(tools)
    out = a.run()[0]
    reruns = [c for c in tools.calls if c[0] == "analyze" and c[1] > 0]
    assert len(reruns) == policy.MAX_ATTEMPTS == 2
    assert [c[2] for c in reruns] == [3, 2]  # stride 6 -> 3 -> 2
    assert out.status == "uncertain" and out.attempts == 2
    assert tools.reviews and tools.reviews[-1]["kind"] == "approve_rerun"
    assert tools.reviews[-1]["proposed_config"]["stride"] == 1
    states = [s.state for s in a.trace.steps]
    assert states[0] == "INGEST" and states[-1] == "COMPLETE" and "REQUEST_HUMAN_REVIEW" in states


def test_rejected_correction_preserves_previous_result():
    tools = FakeTools([q(short_track_ratio=0.6), q(short_track_ratio=0.6)], compare_ok=False)
    out = make_agent(tools).run()[0]
    assert out.status == "uncertain"
    assert out.final_result == out.initial_result
    assert len([c for c in tools.calls if c[0] == "analyze"]) == 2  # initial + one rejected rerun


def test_camera_drift_runs_motion_update_then_accepts():
    tools = FakeTools([q(camera_drift_px=90, line_alignment=0.2),
                       q(camera_drift_px=90, line_alignment=0.85, motion_compensated=True)])
    a = make_agent(tools)
    out = a.run()[0]
    names = [c[0] for c in tools.calls]
    assert names.index("estimate_motion") < names.index("update_calibration") < names.index("compare")
    rerun = [c for c in tools.calls if c[0] == "analyze" and c[1] == 1][0]
    assert rerun[3] is True  # motion compensation on in the rerun
    assert out.status == "recovered" and out.final_result != out.initial_result


def test_failed_calibration_update_escalates_without_rerun():
    tools = FakeTools([q(camera_drift_px=90, line_alignment=0.2)], update_ok=False)
    out = make_agent(tools).run()[0]
    assert out.status == "uncertain"
    assert not [c for c in tools.calls if c[0] == "analyze" and c[1] > 0]
    assert tools.reviews[-1]["kind"] == "recalibrate"


def test_decision_depends_on_signals_not_fixed_sequence():
    good = FakeTools([q()])
    make_agent(good).run()
    assert [c[0] for c in good.calls] == ["analyze"]  # nothing to fix -> no extra tool calls
    drift = FakeTools([q(camera_drift_px=90, line_alignment=0.2), q(line_alignment=0.9, motion_compensated=True)])
    make_agent(drift).run()
    frag = FakeTools([q(short_track_ratio=0.6), q(short_track_ratio=0.1)])
    make_agent(frag).run()
    assert "estimate_motion" in [c[0] for c in drift.calls]
    assert "estimate_motion" not in [c[0] for c in frag.calls]
    assert [c[2] for c in frag.calls if c[0] == "analyze"] == [6, 3]
