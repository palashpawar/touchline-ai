"""Structured tools the agent calls. Each takes/returns JSON-serialisable data and is listed in TOOL_SPECS."""
from __future__ import annotations

import uuid

import numpy as np

from .. import quality
from ..pipeline import RunContext, SegmentConfig, analyze_segment, line_alignment
from ..vision import field as fieldmod
from ..vision import motion as motionmod
from ..vision.detector import filter_players
from ..vision.video import read_frames, sample_indices
from . import policy

TOOL_SPECS = {
    "analyze_segment": {"args": {"segment_id": "str", "config": "SegmentConfig"},
                        "returns": "result_id + summary", "doc": "Detect, filter, project and track one segment."},
    "assess_tracking_quality": {"args": {"result_id": "str", "frames": "list[int] | None"},
                                "returns": "QualityReport", "doc": "Compute quality signals, optionally on a fixed frame set."},
    "estimate_camera_motion": {"args": {"segment_id": "str", "stride": "int"},
                               "returns": "motion_id + stats", "doc": "Per-frame frame->reference homographies (ORB to reference, LK chaining fallback)."},
    "update_calibration": {"args": {"motion_id": "str"},
                           "returns": "accepted + per-frame homographies", "doc": "Validate a motion model and derive per-frame calibration."},
    "rerun_segment": {"args": {"segment_id": "str", "config": "SegmentConfig", "attempt": "int"},
                      "returns": "result_id + summary", "doc": "Re-analyse a segment with a corrected configuration."},
    "compare_runs": {"args": {"before_id": "str", "after_id": "str", "target_signals": "list[str]"},
                     "returns": "accepted + per-signal deltas", "doc": "Compare runs on identical evaluation frames."},
    "request_human_review": {"args": {"segment_id": "str", "reason": "str", "kind": "str", "proposed_config": "dict | None"},
                             "returns": "review_id", "doc": "Escalate to the coach/analyst."},
}


def restrict(seg: dict, frames: set[int]) -> dict:
    """Copy of a segment result keeping only the given frame indices (frames and track observations)."""
    out = dict(seg)
    out["frames"] = [f for f in seg["frames"] if f["frame"] in frames]
    tracks = []
    for tr in seg["tracks"]:
        obs = [o for o in tr["obs"] if o["frame"] in frames]
        if obs:
            tracks.append(dict(tr, obs=obs))
    out["tracks"] = tracks
    return out


class Toolbox:
    def __init__(self, ctx: RunContext, progress=None):
        self.ctx = ctx
        self.results: dict[str, dict] = {}
        self.motions: dict[str, dict] = {}
        self.reviews: list[dict] = []
        self.progress = progress

    def _summary(self, rid: str) -> dict:
        r = self.results[rid]
        return {"result_id": rid, "segment_id": r["segment_id"], "sampled_frames": len(r["frames"]),
                "tracks": len(r["tracks"]), "elapsed_s": round(r["elapsed_s"], 2), "config": r["config"]}

    def analyze_segment(self, segment_id: str, start: int, end: int, config: SegmentConfig, attempt: int = 0) -> dict:
        res = analyze_segment(self.ctx, segment_id, start, end, config, attempt=attempt, progress=self.progress)
        rid = f"{segment_id}.a{attempt}.{uuid.uuid4().hex[:6]}"
        self.results[rid] = res.to_dict()
        return self._summary(rid)

    rerun_segment = analyze_segment

    def assess_tracking_quality(self, result_id: str, frames: list[int] | None = None) -> dict:
        seg = self.results[result_id]
        if frames is not None:
            seg = restrict(seg, set(frames))
        q = quality.assess(seg, self.ctx.quad_pitch if self.ctx.valid_calibration else None, self.ctx.valid_calibration)
        return q.to_dict()

    def estimate_camera_motion(self, segment_id: str, start: int, end: int, stride: int) -> dict:
        ctx = self.ctx
        wanted = set(sample_indices(start, end, stride))
        model, rows = {}, []
        prev_frame, prev_boxes, prev_M = None, [], None
        # Decode every frame so the LK chain has small inter-frame motion; ORB-anchor the sampled frames.
        for fi, frame in read_frames(ctx.video_path, list(range(start, end))):
            fm = fieldmod.field_mask(frame)
            if fi in wanted or prev_M is None:
                boxes = [d.box for d in filter_players(ctx.detector.detect(frame, fieldmod.field_bbox(fm)), fm, frame=frame)[0]] \
                    if fi in wanted else []
                est = ctx.matcher.match_to_reference(frame, boxes)
            else:
                boxes, est = [], motionmod.MotionEstimate(None, "skip", 0, 0, 0.0, float("inf"), False)
            M = np.asarray(est.M) if est.validated else None
            if M is None and prev_frame is not None and prev_M is not None:
                pair = motionmod.track_pair(prev_frame, frame, prev_boxes)
                if pair.validated:
                    M = prev_M @ np.asarray(pair.M)
                    est = motionmod.MotionEstimate(M.tolist(), "lk_chain", pair.matches, pair.inliers,
                                                   pair.inlier_ratio, pair.reproj_error_px, True)
            if fi in wanted:
                rows.append({"frame": fi, **{k: v for k, v in est.to_dict().items() if k != "M"}})
                if M is not None:
                    model[str(fi)] = M.tolist()
            prev_frame, prev_boxes = frame, boxes
            prev_M = M if M is not None else prev_M
        n = max(1, len(rows))
        stats = {
            "frames": len(rows), "validated_fraction": len(model) / n,
            "median_inlier_ratio": float(np.median([r["inlier_ratio"] for r in rows])) if rows else 0.0,
            "median_reproj_px": float(np.median([r["reproj_error_px"] for r in rows if np.isfinite(r["reproj_error_px"])] or [np.inf])),
            "methods": {m: sum(r["method"] == m for r in rows) for m in {r["method"] for r in rows}},
        }
        mid = f"motion.{segment_id}.{uuid.uuid4().hex[:6]}"
        self.motions[mid] = {"model": model, "stats": stats, "segment_id": segment_id, "start": start, "end": end}
        return {"motion_id": mid, "stats": stats}

    def update_calibration(self, motion_id: str, probe_frames: int = 5) -> dict:
        """Accept the motion model only if it validates AND improves pitch-line alignment on probe frames."""
        mo = self.motions[motion_id]
        st = mo["stats"]
        if st["validated_fraction"] < quality.SIGNALS["motion_validated_fraction"].threshold or \
                st["median_reproj_px"] > quality.SIGNALS["motion_reproj_px"].threshold:
            return {"accepted": False, "reason": f"motion model not validated: {st}", "probe": []}
        keys = sorted(mo["model"], key=int)
        picks = [keys[i] for i in np.linspace(0, len(keys) - 1, min(probe_frames, len(keys))).astype(int)]
        probe = []
        for fi, frame in read_frames(self.ctx.video_path, [int(k) for k in picks]):
            fm = fieldmod.field_mask(frame)
            boxes = [d.box for d in filter_players(self.ctx.detector.detect(frame, fieldmod.field_bbox(fm)), fm, frame=frame)[0]]
            a0 = line_alignment(self.ctx, frame, fm, boxes, None)
            a1 = line_alignment(self.ctx, frame, fm, boxes, np.asarray(mo["model"][str(fi)]))
            probe.append({"frame": fi, "alignment_static": a0, "alignment_updated": a1})
        s = [p["alignment_static"] for p in probe if p["alignment_static"] is not None]
        u = [p["alignment_updated"] for p in probe if p["alignment_updated"] is not None]
        if not u:
            return {"accepted": False, "reason": "no pitch lines visible to validate the update", "probe": probe}
        ms, mu = float(np.median(s)) if s else 0.0, float(np.median(u))
        ok = mu >= quality.SIGNALS["line_alignment"].threshold and mu >= ms
        return {"accepted": bool(ok), "probe": probe, "median_alignment_static": ms, "median_alignment_updated": mu,
                "reason": (f"line alignment {ms:.2f} -> {mu:.2f} on {len(probe)} probe frames" +
                           ("" if ok else f"; below threshold {quality.SIGNALS['line_alignment'].threshold}")),
                "motion_model": mo["model"] if ok else None}

    def compare_runs(self, before_id: str, after_id: str, target_signals: list[str]) -> dict:
        fb = {f["frame"] for f in self.results[before_id]["frames"]}
        fa = {f["frame"] for f in self.results[after_id]["frames"]}
        common = sorted(fb & fa)
        qb = self.assess_tracking_quality(before_id, common)
        qa = self.assess_tracking_quality(after_id, common)
        cmp = policy.compare_signals(qb["signals"], qa["signals"], target_signals)
        cmp.update(evaluation_frames=len(common), quality_before=qb, quality_after=qa)
        return cmp

    def request_human_review(self, segment_id: str | None, reason: str, kind: str, proposed_config: dict | None,
                             result_id: str | None = None) -> dict:
        rid = f"review-{len(self.reviews) + 1}"
        self.reviews.append({"review_id": rid, "segment_id": segment_id, "reason": reason, "kind": kind,
                             "proposed_config": proposed_config, "result_id": result_id, "status": "open"})
        return {"review_id": rid}
