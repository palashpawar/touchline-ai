"""Job lifecycle: upload validation, calibration, agent runs, human review actions, report export."""
from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

from . import quality, results
from .agent import policy
from .agent.agent import SegmentAgent
from .agent.tools import Toolbox
from .agent.trace import Trace
from .config import Settings
from .db import DB
from .pipeline import RunContext, SegmentConfig
from .storage import Storage, put_json
from .vision import calibration as cal
from .vision import field as fieldmod
from .vision.detector import MODEL_SHA256, YoloxPersonDetector
from .vision.video import VideoError, probe, read_frame

log = logging.getLogger("touchline.jobs")

ALLOWED_EXT = {".mp4", ".mov", ".m4v", ".webm", ".mkv", ".avi", ".ogv"}
MIN_W, MIN_H = 640, 360
_detector: YoloxPersonDetector | None = None


class UploadError(ValueError):
    pass


def detector(settings: Settings) -> YoloxPersonDetector:
    global _detector
    if _detector is None:
        _detector = YoloxPersonDetector(settings.model_path)
    return _detector


def validate_upload(filename: str, size_bytes: int, settings: Settings) -> str:
    ext = Path(filename or "").suffix.lower()
    if ext not in ALLOWED_EXT:
        raise UploadError(f"Unsupported file type '{ext or 'none'}'. Upload one of: {', '.join(sorted(ALLOWED_EXT))}.")
    if size_bytes <= 0:
        raise UploadError("The uploaded file is empty.")
    if size_bytes > settings.max_upload_mb * 1024 * 1024:
        raise UploadError(f"File is {size_bytes / 1e6:.0f} MB; the limit is {settings.max_upload_mb} MB.")
    return ext


def validate_video(path: Path, settings: Settings) -> dict:
    try:
        meta = probe(path)
    except VideoError as e:
        raise UploadError(str(e)) from e
    if meta.width < MIN_W or meta.height < MIN_H:
        raise UploadError(f"Resolution {meta.width}x{meta.height} is below the {MIN_W}x{MIN_H} minimum.")
    tol = 0.5
    if meta.duration_s < settings.min_duration_s - tol or meta.duration_s > settings.max_duration_s + tol:
        raise UploadError(f"Clip is {meta.duration_s:.1f} s long; upload a clip between "
                          f"{settings.min_duration_s:.0f} and {settings.max_duration_s:.0f} seconds (trim it first).")
    return meta.to_dict()


def ingest(db: DB, storage: Storage, settings: Settings, filename: str, src: Path) -> dict:
    ext = validate_upload(filename, src.stat().st_size, settings)
    meta = validate_video(src, settings)
    job = db.create_job(Path(filename).name, "", meta)
    key = f"jobs/{job['id']}/input{ext}"
    storage.put_file(key, src, "video/" + ext.lstrip("."))
    db.update_job(job["id"], input_key=key)
    log.info("job ingested", extra={"job_id": job["id"], "event": "ingest"})
    return db.get_job(job["id"])


def input_path(storage: Storage, job: dict) -> Path:
    return storage.fetch(job["input_key"])


def frame_jpeg(storage: Storage, job: dict, index: int, max_width: int = 1920) -> bytes:
    frame = read_frame(input_path(storage, job), index)
    if frame.shape[1] > max_width:
        frame = cv2.resize(frame, None, fx=max_width / frame.shape[1], fy=max_width / frame.shape[1])
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    return buf.tobytes()


def line_segments(storage: Storage, job: dict, index: int) -> list[list[float]]:
    frame = read_frame(input_path(storage, job), index)
    return fieldmod.detect_line_segments(frame).round(1).tolist()


def calibrate(db: DB, storage: Storage, job: dict, body: dict) -> dict:
    """body: {frame_index, image_points[4], pitch_points[4] | rectangle{x0,y0,width_m,height_m}, pitch_size?, holdout?}"""
    idx = int(body.get("frame_index", 0))
    if not 0 <= idx < job["meta"]["frame_count"]:
        raise UploadError(f"frame_index must be in [0, {job['meta']['frame_count'] - 1}]")
    frame = read_frame(input_path(storage, job), idx)
    pitch_size = tuple(body.get("pitch_size") or cal.DEFAULT_PITCH)
    if body.get("rectangle"):
        r = body["rectangle"]
        pitch_pts = cal.rectangle_pitch_points(float(r["x0"]), float(r["y0"]), float(r["width_m"]), float(r["height_m"]))
    else:
        pitch_pts = np.asarray(body.get("pitch_points"), np.float64)
    rep = cal.validate_calibration(body.get("image_points"), pitch_pts, frame.shape, fieldmod.field_mask(frame),
                                   pitch_size=pitch_size, holdout=body.get("holdout"))
    out = rep.to_dict()
    out.update(frame_index=idx, image_points=body.get("image_points"), pitch_points=np.asarray(pitch_pts).tolist(),
               pitch_size=list(pitch_size), rectangle=body.get("rectangle"), label=body.get("label"))
    db.update_job(job["id"], calibration=out, status="calibrated" if rep.valid else "uploaded",
                  stage="ready to analyze" if rep.valid else "calibration invalid")
    return out


# ---------------------------------------------------------------- runs

def _run_prefix(job_id: str, run_no: int) -> str:
    return f"jobs/{job_id}/runs/{run_no}"


def run_analysis(db: DB, storage: Storage, settings: Settings, job_id: str, payload: dict) -> None:
    job = db.get_job(job_id)
    run_no = int(job["run_no"] or 0) + 1
    db.update_job(job_id, run_no=run_no, status="running", stage="preparing", progress=0.0, error=None,
                  options=payload)
    t0 = time.time()
    video = input_path(storage, job)
    meta = probe(video)
    calib = job["calibration"] or {"valid": False, "errors": ["no calibration submitted"]}
    det = detector(settings)
    ctx = RunContext(video, meta, calib, det)

    def sink(step):
        db.add_step(job_id, run_no, step.to_dict())

    trace = Trace(f"{job_id}.run{run_no}", sink=sink)
    init = SegmentConfig(stride=int(payload.get("initial_stride", 6)))

    def prog(p):
        db.update_job(job_id, progress=round(0.8 * p, 3), stage=f"agent analysing segments ({p:.0%})")

    agent = SegmentAgent(ctx, trace, init, progress=prog)
    if not ctx.valid_calibration:
        # Without a valid calibration the agent stops at CALIBRATE and asks for points; no metric output.
        trace.log(state="INGEST", segment_id=None, decision="ingest", reason="clip accepted", outcome="ok", accepted=True)
        trace.log(state="CALIBRATE", segment_id=None, decision="validate_calibration",
                  reason="; ".join(calib.get("errors", [])) or "invalid", outcome="invalid", accepted=False)
        rv = {"review_id": "review-1", "segment_id": None, "kind": "recalibrate",
              "reason": "Calibration invalid: " + "; ".join(calib.get("errors", [])), "proposed_config": None}
        trace.log(state="REQUEST_HUMAN_REVIEW", segment_id=None, decision="request_human_review", reason=rv["reason"],
                  tool="request_human_review", tool_args={"kind": "recalibrate"}, outcome="awaiting user", accepted=False)
        db.add_review(job_id, run_no, rv)
        db.update_job(job_id, status="needs_review", stage="calibration required", progress=1.0)
        return
    outcomes = agent.run()
    _finalise(db, storage, settings, job_id, run_no, ctx, agent.tools, [o.__dict__ for o in outcomes], trace,
              payload, t0, initial_stride=init.stride)


def _finalise(db, storage, settings, job_id, run_no, ctx, tools: Toolbox, outcomes: list[dict], trace: Trace,
              options: dict, t0: float, initial_stride: int, extra_steps: list[dict] | None = None) -> None:
    prefix = _run_prefix(job_id, run_no)
    statuses = {o["segment_id"]: o["status"] for o in outcomes}
    final = [tools.results[o["final_result"]] for o in outcomes]
    initial = [tools.results[o["initial_result"]] for o in outcomes]
    team_colors = options.get("team_colors")
    overrides = {int(k): v for k, v in (options.get("team_overrides") or {}).items()}
    db.update_job(job_id, stage="assembling tactical outputs", progress=0.82)
    A_final = results.assemble(ctx, final, statuses, team_colors, overrides)
    A_init = results.assemble(ctx, initial, {o["segment_id"]: "initial" for o in outcomes}, team_colors, overrides)
    tmp = Path(tempfile.mkdtemp(prefix="touchline-"))
    try:
        db.update_job(job_id, stage="rendering annotated video (final)", progress=0.86)
        vf = results.render_video(ctx, A_final, final, tmp / "annotated_final", "Touchline AI - corrected", statuses=statuses)
        db.update_job(job_id, stage="rendering annotated video (initial)", progress=0.93)
        vi = results.render_video(ctx, A_init, initial, tmp / "annotated_initial", "Touchline AI - initial run")
        artifacts = {"annotated_final": vf.name, "annotated_initial": vi.name}
        for p in (vf, vi):
            storage.put_file(f"{prefix}/{p.name}", p, "video/webm" if p.suffix == ".webm" else "video/mp4")
        comps = []
        for o in outcomes:
            if o["final_result"] != o["initial_result"] or o["comparisons"]:
                after = tools.results[o["comparisons"][-1]["after"]] if o["comparisons"] else tools.results[o["final_result"]]
                c = results.comparison_frame(ctx, tools.results[o["initial_result"]], after, tmp / f"compare_{o['segment_id']}.jpg")
                if c:
                    storage.put_file(f"{prefix}/compare_{o['segment_id']}.jpg", Path(c["path"]), "image/jpeg")
                    comps.append({"segment_id": o["segment_id"], "frame": c["frame"], "artifact": f"compare_{o['segment_id']}.jpg"})
        artifacts["comparisons"] = comps
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    for rv in tools.reviews:
        if rv.get("status") == "open":
            db.add_review(job_id, run_no, rv)
    report = build_report(db.get_job(job_id), run_no, ctx, outcomes, A_final, A_init, tools, artifacts, t0,
                          initial_stride)
    put_json(storage, f"{prefix}/report.json", report)
    storage.put_bytes(f"{prefix}/trace.jsonl", trace.to_jsonl().encode(), "application/x-ndjson")
    state = {"outcomes": outcomes, "results": tools.results, "reviews": tools.reviews, "options": options,
             "initial_stride": initial_stride}
    storage.put_bytes(f"{prefix}/state.json", json.dumps(state, default=_np).encode(), "application/json")
    needs = any(o["status"] == "uncertain" for o in outcomes)
    db.update_job(job_id, status="needs_review" if needs else "complete",
                  stage="human review requested" if needs else "complete", progress=1.0)
    log.info("run finished", extra={"job_id": job_id, "event": "run_finished", "elapsed_s": round(time.time() - t0, 1),
                                     "status": "needs_review" if needs else "complete"})


def _np(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def _summary_metrics(A: dict) -> dict:
    out = {}
    for t in ("A", "B"):
        s = A["team_shape"][t]
        out[t] = {k: s.get(k) for k in ("available", "width_m_median", "depth_m_median", "visible_players_median",
                                         "extrapolated_share_median", "frames", "reason")}
    out["tracks"] = len(A["tracks"])
    out["tracks_ge_2s"] = sum(1 for t in A["tracks"] if t["obs"][-1]["t"] - t["obs"][0]["t"] >= 2.0)
    return out


def build_report(job, run_no, ctx: RunContext, outcomes, A_final, A_init, tools: Toolbox, artifacts, t0, initial_stride) -> dict:
    import platform
    return {
        "schema": "touchline.report/v1",
        "job": {k: job[k] for k in ("id", "filename", "created_at", "meta")},
        "run_no": run_no,
        "provenance": {"opencv": cv2.__version__, "numpy": np.__version__, "python": platform.python_version(),
                       "machine": platform.machine(), "platform": platform.platform(),
                       "detector": {"model": "YOLOX-S (OpenCV Zoo object_detection_yolox_2022nov.onnx)",
                                    "sha256": MODEL_SHA256, "license": "Apache-2.0", "runtime": "cv2.dnn"},
                       "agent": "rule-based state machine (no LLM in the control path)",
                       "processing_seconds": round(time.time() - t0, 1)},
        "calibration": {k: v for k, v in (job["calibration"] or {}).items()},
        "quality_signal_definitions": quality.spec_table(),
        "policy": {"max_attempts_per_segment": policy.MAX_ATTEMPTS, "stride_schedule": list(policy.STRIDE_SCHEDULE),
                   "initial_stride": initial_stride, "min_relative_improvement": policy.MIN_REL_IMPROVEMENT},
        "segments": [{k: o[k] for k in ("segment_id", "start", "end", "status", "attempts", "initial_quality",
                                        "final_quality", "comparisons", "review_id")} |
                     {"final_config": tools.results[o["final_result"]]["config"],
                      "initial_config": tools.results[o["initial_result"]]["config"]} for o in outcomes],
        "summary": {"initial": _summary_metrics(A_init), "final": _summary_metrics(A_final)},
        "results": {k: v for k, v in A_final.items() if not k.startswith("_")},
        "initial_results": {"team_shape": {k: v for k, v in A_init["team_shape"].items() if k != "per_frame"},
                            "tracks": len(A_init["tracks"])},
        "reviews": tools.reviews,
        "artifacts": artifacts,
        "measurement_classes": {
            "measured": ["player detections", "projected ground positions (m)", "trails", "occupancy heatmap",
                         "team width/depth of visible players"],
            "heuristic_quality_indicators": [s["key"] for s in quality.spec_table()] + ["team confidence"],
            "unavailable": list(A_final["unavailable"].keys())},
    }


def load_state(storage: Storage, job_id: str, run_no: int) -> dict:
    return json.loads(storage.read_bytes(f"{_run_prefix(job_id, run_no)}/state.json"))


def review_action(db: DB, storage: Storage, settings: Settings, job_id: str, payload: dict) -> None:
    """Human decision on an open review. Actions:
    approve_rerun: run the agent-proposed configuration for that segment (beyond the automatic retry budget),
                   then compare on identical frames; the correction is kept only if it validates.
    accept_uncertain: keep the preserved result and include it in metrics, flagged as reviewer-accepted.
    team_update: change team colours/overrides and re-assemble without re-running vision.
    """
    job = db.get_job(job_id)
    run_no = int(job["run_no"])
    st = load_state(storage, job_id, run_no)
    video = input_path(storage, job)
    ctx = RunContext(video, probe(video), job["calibration"], detector(settings))
    tools = Toolbox(ctx)
    tools.results = st["results"]
    tools.reviews = st["reviews"]
    outcomes = st["outcomes"]
    options = dict(st.get("options") or {})
    prior = db.steps(job_id, run_no)
    trace = Trace(f"{job_id}.run{run_no}", sink=lambda s: db.add_step(job_id, run_no, s.to_dict()))
    from .agent.trace import AgentStep
    for s in prior:
        s = {k: v for k, v in s.items() if k != "run_no"}
        trace.steps.append(AgentStep(**s))
    t0 = time.time()
    action = payload["action"]
    db.update_job(job_id, status="running", stage=f"applying review: {action}", progress=0.05)
    rv_row = db.get_review(payload["review_id"]) if payload.get("review_id") else None
    seg_id = rv_row["segment_id"] if rv_row else None
    oc = next((o for o in outcomes if o["segment_id"] == seg_id), None)
    if rv_row:
        local_id = rv_row["id"].split(f"-r{run_no}-", 1)[-1]
        for r in tools.reviews:
            if r["review_id"] == local_id:
                r["status"] = "resolved"

    if action == "approve_rerun" and oc:
        cfg_d = dict(rv_row["proposed_config"] or {})
        prev_cfg = tools.results[oc["final_result"]]["config"]
        cfg_d.setdefault("motion_compensation", prev_cfg["motion_compensation"])
        cfg = SegmentConfig(stride=int(cfg_d.get("stride") or 1), motion_compensation=bool(cfg_d["motion_compensation"]))
        if cfg.motion_compensation:
            mo = tools.estimate_camera_motion(seg_id, oc["start"], oc["end"], cfg.stride)
            up = tools.update_calibration(mo["motion_id"])
            cfg.motion_model = up.get("motion_model")
        trace.log(state="RERUN_SEGMENT", segment_id=seg_id, decision="human_approved_rerun",
                  reason="Reviewer approved the proposed rerun", tool="rerun_segment", actor="human",
                  tool_args={"segment_id": seg_id, "config": cfg.to_dict()}, previous_config=prev_cfg,
                  proposed_config=cfg.to_dict(), attempt=oc["attempts"] + 1, outcome="queued")
        summ = tools.rerun_segment(seg_id, oc["start"], oc["end"], cfg, attempt=oc["attempts"] + 1)
        targets = [k for k in policy.TRACKING_SIGNALS + policy.CAMERA_SIGNALS if k != "camera_drift_px"]
        cmp = tools.compare_runs(oc["final_result"], summ["result_id"], targets)
        trace.log(state="COMPARE", segment_id=seg_id, decision="compare_runs", reason=cmp["reason"], tool="compare_runs",
                  tool_args={"before_id": oc["final_result"], "after_id": summ["result_id"],
                             "evaluation_frames": cmp["evaluation_frames"]},
                  quality_before={"signals": cmp["quality_before"]["signals"], "breaches": cmp["quality_before"]["breaches"]},
                  quality_after={"signals": cmp["quality_after"]["signals"], "breaches": cmp["quality_after"]["breaches"]},
                  attempt=oc["attempts"] + 1, outcome="accepted" if cmp["accepted"] else "rejected",
                  accepted=cmp["accepted"])
        oc["comparisons"].append({"attempt": oc["attempts"] + 1, "action": "human_approved_rerun",
                                  "before": oc["final_result"], "after": summ["result_id"],
                                  **{k: v for k, v in cmp.items() if k not in ("quality_before", "quality_after")}})
        oc["attempts"] += 1
        if cmp["accepted"]:
            oc["final_result"] = summ["result_id"]
            q = tools.assess_tracking_quality(summ["result_id"])
            oc["final_quality"] = {"signals": q["signals"], "breaches": q["breaches"]}
            oc["status"] = "recovered" if not q["breaches"] else "uncertain"
        db.resolve_review(rv_row["id"], "approved", {"accepted": cmp["accepted"], "reason": cmp["reason"]})
    elif action == "accept_uncertain" and oc:
        oc["status"] = "reviewer_accepted"
        trace.log(state="REQUEST_HUMAN_REVIEW", segment_id=seg_id, decision="reviewer_accepted",
                  reason=payload.get("note") or "Reviewer accepted the preserved result", actor="human",
                  outcome="segment included with reviewer flag", accepted=True)
        db.resolve_review(rv_row["id"], "accepted", {"note": payload.get("note")})
    elif action == "team_update":
        options["team_colors"] = payload.get("team_colors") or options.get("team_colors")
        options["team_overrides"] = payload.get("team_overrides") or options.get("team_overrides")
        trace.log(state="COMPLETE", segment_id=None, decision="team_assignment_updated", actor="human",
                  reason=f"colours={options.get('team_colors')}, overrides={options.get('team_overrides')}",
                  outcome="re-assembled", accepted=True)
    else:
        raise ValueError(f"unsupported review action {action!r}")
    n_unc = sum(o["status"] == "uncertain" for o in outcomes)
    trace.log(state="COMPLETE", segment_id=None, decision="complete", reason=f"{n_unc} segments still need review",
              outcome="needs_review" if n_unc else "complete", accepted=n_unc == 0)
    _finalise(db, storage, settings, job_id, run_no, ctx, tools, outcomes, trace, options, t0,
              int(st.get("initial_stride", 6)))
