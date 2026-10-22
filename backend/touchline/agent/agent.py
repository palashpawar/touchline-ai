"""Explicit perception -> decision -> action state machine over clip segments.

States: INGEST, CALIBRATE, ANALYZE, ASSESS_QUALITY, PLAN_RECOVERY, RERUN_SEGMENT, COMPARE,
REQUEST_HUMAN_REVIEW, COMPLETE. The plan chosen in PLAN_RECOVERY is a pure function of the
signals measured in ASSESS_QUALITY (see policy.plan_recovery), so the tool called next — and its
arguments — are determined by the OpenCV measurements of that specific segment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..pipeline import RunContext, SegmentConfig
from . import policy
from .tools import Toolbox
from .trace import Trace


@dataclass
class SegmentOutcome:
    segment_id: str
    start: int
    end: int
    initial_result: str
    final_result: str
    status: str  # accepted | recovered | uncertain
    attempts: int
    initial_quality: dict
    final_quality: dict
    comparisons: list[dict] = field(default_factory=list)
    review_id: str | None = None


def _slim_q(q: dict) -> dict:
    return {"signals": q["signals"], "breaches": q["breaches"]}


def _cfg(c: SegmentConfig | dict) -> dict:
    d = c.to_dict() if isinstance(c, SegmentConfig) else dict(c)
    if d.get("motion_model"):
        d["motion_model"] = {"frames": len(d["motion_model"]) if isinstance(d["motion_model"], dict) else d["motion_model"].get("frames"),
                             "source": "update_calibration"}
    return d


class SegmentAgent:
    def __init__(self, ctx: RunContext, trace: Trace, initial: SegmentConfig | None = None,
                 max_attempts: int = policy.MAX_ATTEMPTS, segment_seconds: float = 4.0, progress=None):
        self.ctx = ctx
        self.trace = trace
        self.initial = initial or SegmentConfig()
        self.max_attempts = max_attempts
        self.segment_seconds = segment_seconds
        self.tools = Toolbox(ctx, progress=None)
        self.progress = progress
        self.outcomes: list[SegmentOutcome] = []

    def run(self) -> list[SegmentOutcome]:
        meta = self.ctx.meta
        self.trace.log(state="INGEST", segment_id=None, decision="ingest",
                       reason=f"{meta.width}x{meta.height} @ {meta.fps:.2f} fps, {meta.frame_count} frames "
                              f"({meta.duration_s:.1f} s)", tool=None, observed_signals=meta.to_dict(),
                       outcome="ok", accepted=True)
        cal = self.ctx.calibration
        self.trace.log(state="CALIBRATE", segment_id=None, decision="validate_calibration",
                       reason="valid" if cal.get("valid") else "; ".join(cal.get("errors", [])),
                       observed_signals={k: v for k, v in cal.get("metrics", {}).items() if "quad" not in k},
                       outcome="valid" if cal.get("valid") else "invalid", accepted=bool(cal.get("valid")))
        segs = self.ctx.segments(self.segment_seconds)
        for i, (sid, s, e) in enumerate(segs):
            self.outcomes.append(self._segment(sid, s, e))
            if self.progress:
                self.progress((i + 1) / len(segs))
        n_unc = sum(o.status == "uncertain" for o in self.outcomes)
        self.trace.log(state="COMPLETE", segment_id=None, decision="complete",
                       reason=f"{len(self.outcomes)} segments: "
                              f"{sum(o.status == 'accepted' for o in self.outcomes)} accepted, "
                              f"{sum(o.status == 'recovered' for o in self.outcomes)} recovered, {n_unc} need review",
                       outcome="needs_review" if n_unc else "complete", accepted=n_unc == 0)
        return self.outcomes

    def _segment(self, sid: str, s: int, e: int) -> SegmentOutcome:
        T, tools = self.trace, self.tools
        cfg = SegmentConfig(**{k: v for k, v in self.initial.__dict__.items()})
        summ = tools.analyze_segment(sid, s, e, cfg, attempt=0)
        T.log(state="ANALYZE", segment_id=sid, decision="analyze_segment", reason="initial schedule",
              tool="analyze_segment", tool_args={"segment_id": sid, "frames": [s, e], "config": _cfg(cfg)},
              proposed_config=_cfg(cfg), outcome=f"{summ['tracks']} tracks over {summ['sampled_frames']} frames "
                                                f"in {summ['elapsed_s']} s", accepted=True)
        cur = summ["result_id"]
        q = tools.assess_tracking_quality(cur)
        T.log(state="ASSESS_QUALITY", segment_id=sid, decision="assess_tracking_quality",
              reason=("breaches: " + ", ".join(q["breaches"])) if q["breaches"] else "no breaches",
              tool="assess_tracking_quality", tool_args={"result_id": cur}, observed_signals=q["signals"],
              quality_before=_slim_q(q), outcome="assessed")
        initial_q = q
        attempts, comparisons, status, review_id = 0, [], "accepted", None
        recovered = False
        while True:
            plan = policy.plan_recovery(q["signals"], q["breaches"], _cfg(cfg), attempts)
            T.log(state="PLAN_RECOVERY", segment_id=sid, decision=plan.action, reason=plan.reason,
                  observed_signals={k: q["signals"].get(k) for k in (plan.target_signals or q["breaches"])},
                  previous_config=_cfg(cfg), proposed_config=plan.proposed_config and _cfg(plan.proposed_config),
                  attempt=attempts, outcome="planned")
            if plan.action == "accept":
                status = "recovered" if recovered else "accepted"
                break
            if plan.action == "request_human_review":
                rv = tools.request_human_review(sid, plan.reason, plan.review_kind or "inspect",
                                                plan.proposed_config and _cfg(plan.proposed_config), cur)
                review_id = rv["review_id"]
                T.log(state="REQUEST_HUMAN_REVIEW", segment_id=sid, decision="request_human_review",
                      reason=plan.reason, tool="request_human_review",
                      tool_args={"segment_id": sid, "kind": plan.review_kind, "review_id": review_id},
                      proposed_config=plan.proposed_config and _cfg(plan.proposed_config),
                      quality_before=_slim_q(q), attempt=attempts, outcome="segment marked uncertain; previous "
                      "result preserved", accepted=False)
                status = "uncertain"
                break

            attempts += 1
            new_cfg = SegmentConfig(**{k: v for k, v in cfg.__dict__.items()})
            if plan.action == "camera_motion_update":
                new_cfg.motion_compensation = True
                mo = tools.estimate_camera_motion(sid, s, e, new_cfg.stride)
                T.log(state="RERUN_SEGMENT", segment_id=sid, decision="estimate_camera_motion",
                      reason="camera drift / line misalignment observed", tool="estimate_camera_motion",
                      tool_args={"segment_id": sid, "stride": new_cfg.stride}, observed_signals=mo["stats"],
                      attempt=attempts, outcome=f"validated {mo['stats']['validated_fraction']:.0%} of frames")
                up = tools.update_calibration(mo["motion_id"])
                T.log(state="RERUN_SEGMENT", segment_id=sid, decision="update_calibration", reason=up["reason"],
                      tool="update_calibration", tool_args={"motion_id": mo["motion_id"]},
                      observed_signals={"probe": up.get("probe")}, attempt=attempts,
                      outcome="calibration updated per frame" if up["accepted"] else "update rejected",
                      accepted=up["accepted"])
                if not up["accepted"]:
                    rv = tools.request_human_review(sid, "Camera-motion calibration update failed validation: "
                                                    + up["reason"], "recalibrate", None, cur)
                    review_id = rv["review_id"]
                    T.log(state="REQUEST_HUMAN_REVIEW", segment_id=sid, decision="request_human_review",
                          reason=up["reason"], tool="request_human_review",
                          tool_args={"segment_id": sid, "kind": "recalibrate", "review_id": review_id},
                          quality_before=_slim_q(q), attempt=attempts,
                          outcome="previous result preserved; segment uncertain", accepted=False)
                    status = "uncertain"
                    break
                new_cfg.motion_model = up["motion_model"]
            elif plan.action == "densify_sampling":
                new_cfg.stride = plan.proposed_config["stride"]

            summ = tools.rerun_segment(sid, s, e, new_cfg, attempt=attempts)
            T.log(state="RERUN_SEGMENT", segment_id=sid, decision="rerun_segment", reason=plan.action,
                  tool="rerun_segment", tool_args={"segment_id": sid, "attempt": attempts, "config": _cfg(new_cfg)},
                  previous_config=_cfg(cfg), proposed_config=_cfg(new_cfg), attempt=attempts,
                  outcome=f"{summ['tracks']} tracks over {summ['sampled_frames']} frames in {summ['elapsed_s']} s")
            cmp = tools.compare_runs(cur, summ["result_id"], plan.target_signals)
            comparisons.append({"attempt": attempts, "action": plan.action, "before": cur,
                                "after": summ["result_id"], **{k: v for k, v in cmp.items()
                                                               if k not in ("quality_before", "quality_after")}})
            T.log(state="COMPARE", segment_id=sid, decision="compare_runs", reason=cmp["reason"], tool="compare_runs",
                  tool_args={"before_id": cur, "after_id": summ["result_id"], "target_signals": plan.target_signals,
                             "evaluation_frames": cmp["evaluation_frames"]},
                  previous_config=_cfg(cfg), proposed_config=_cfg(new_cfg),
                  quality_before=_slim_q(cmp["quality_before"]), quality_after=_slim_q(cmp["quality_after"]),
                  attempt=attempts, outcome="accepted" if cmp["accepted"] else "rejected", accepted=cmp["accepted"])
            if not cmp["accepted"]:
                rv = tools.request_human_review(sid, f"Correction '{plan.action}' did not validate: {cmp['reason']}",
                                                "approve_rerun" if policy.next_stride(new_cfg.stride) else "inspect",
                                                _cfg(dict(new_cfg.__dict__, stride=policy.next_stride(new_cfg.stride)))
                                                if policy.next_stride(new_cfg.stride) else None, cur)
                review_id = rv["review_id"]
                T.log(state="REQUEST_HUMAN_REVIEW", segment_id=sid, decision="request_human_review",
                      reason=cmp["reason"], tool="request_human_review",
                      tool_args={"segment_id": sid, "review_id": review_id}, quality_before=_slim_q(q),
                      attempt=attempts, outcome="previous result preserved; segment uncertain", accepted=False)
                status = "uncertain"
                break
            cur, cfg, recovered = summ["result_id"], new_cfg, True
            q = tools.assess_tracking_quality(cur)
            T.log(state="ASSESS_QUALITY", segment_id=sid, decision="assess_tracking_quality",
                  reason=("breaches: " + ", ".join(q["breaches"])) if q["breaches"] else "no breaches",
                  tool="assess_tracking_quality", tool_args={"result_id": cur}, observed_signals=q["signals"],
                  quality_before=_slim_q(q), attempt=attempts, outcome="assessed")
        return SegmentOutcome(sid, s, e, initial_result=_initial_id(tools, sid), final_result=cur, status=status,
                              attempts=attempts, initial_quality=_slim_q(initial_q), final_quality=_slim_q(q),
                              comparisons=comparisons, review_id=review_id)


def _initial_id(tools: Toolbox, sid: str) -> str:
    return next(k for k in tools.results if k.startswith(f"{sid}.a0."))
