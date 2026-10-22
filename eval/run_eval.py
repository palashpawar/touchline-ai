"""Reproducible evaluation harness for Touchline AI.

    .venv/bin/python eval/run_eval.py            # all cases (~15 min on an Apple M3)
    .venv/bin/python eval/run_eval.py --only natural,identity

Cases
  natural      Real CC BY-SA clip, real (manual) calibration, full agent run. NATURAL camera motion.
  identity     Identity consistency vs manual labels on segment 1 (initial vs corrected run, same frames).
  calibration  Held-out landmark reprojection error on frame 0 and on three moved frames (static vs updated).
  synthetic_*  Constructed clips (clearly SYNTHETIC) with an expected agent action each:
      control_static      frame 0 repeated + sensor noise           -> expect: accept (no recovery)
      synthetic_pan       frame 0 warped by a known pan/rotation    -> expect: camera_motion_update
      time_compressed     every 3rd real frame (3x player speed)    -> expect: densify_sampling somewhere
      textureless_pan     synthetic_pan with texture destroyed      -> expect: request_human_review
      scene_cut           6 s France clip spliced to 6 s CC0 Latvia clip (real footage, synthetic edit)
                          -> expect: first segment fixed/accepted, post-cut segment escalated (reference lost)
      swapped_points      calibration P2/P3 swapped                 -> expect: calibration rejected (review)
      shifted_points      all four points shifted 40 px             -> expect: request_human_review (likely MISSED)
Outputs: eval/out/results.json, eval/RESULTS.md, eval/out/<case>/trace.jsonl
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from touchline.agent.agent import SegmentAgent  # noqa: E402
from touchline.agent.trace import Trace  # noqa: E402
from touchline.pipeline import RunContext, SegmentConfig  # noqa: E402
from touchline.vision import calibration as cal  # noqa: E402
from touchline.vision import field as fieldmod  # noqa: E402
from touchline.vision import motion as motionmod  # noqa: E402
from touchline.vision.detector import YoloxPersonDetector  # noqa: E402
from touchline.vision.video import probe, read_frame, read_frames  # noqa: E402

CLIP = ROOT / "data/clips/france_germany_2018_phase1_ccbysa.ogv"
OUT = ROOT / "eval/out"
LM = json.loads((ROOT / "eval/labels/fg_landmarks.json").read_text())
IDL = json.loads((ROOT / "eval/labels/fg_identity_seg1.json").read_text())
CAL_IMG = [p["image"] for p in LM["calibration"]["points"]]
CAL_PIT = [p["pitch"] for p in LM["calibration"]["points"]]
_det = None


def detector():
    global _det
    if _det is None:
        _det = YoloxPersonDetector(ROOT / "models/object_detection_yolox_2022nov.onnx")
    return _det


def calibration_for(video: Path, img=CAL_IMG, pit=CAL_PIT, frame=0) -> dict:
    f0 = read_frame(video, frame)
    rep = cal.validate_calibration(img, pit, f0.shape, fieldmod.field_mask(f0), holdout=LM["calibration"]["holdout"])
    return dict(rep.to_dict(), frame_index=frame, image_points=img, pitch_points=pit, pitch_size=[105, 68])


def run_agent(name: str, video: Path, calib: dict, stride: int = 6) -> dict:
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    meta = probe(video)
    t0 = time.time()
    trace = Trace(f"eval.{name}")
    if not calib["valid"]:
        trace.log(state="CALIBRATE", segment_id=None, decision="validate_calibration", reason="; ".join(calib["errors"]),
                  outcome="invalid", accepted=False)
        trace.log(state="REQUEST_HUMAN_REVIEW", segment_id=None, decision="request_human_review",
                  reason="calibration invalid", tool="request_human_review", outcome="awaiting user", accepted=False)
        (out / "trace.jsonl").write_text(trace.to_jsonl())
        return {"name": name, "calibration_valid": False, "calibration_errors": calib["errors"], "segments": [],
                "actions": ["request_human_review"], "elapsed_s": round(time.time() - t0, 1), "frames": meta.frame_count}
    ctx = RunContext(video, meta, calib, detector())
    agent = SegmentAgent(ctx, trace, SegmentConfig(stride=stride))
    outcomes = agent.run()
    (out / "trace.jsonl").write_text(trace.to_jsonl())
    segs = []
    for o in outcomes:
        segs.append({"segment_id": o.segment_id, "status": o.status, "attempts": o.attempts,
                     "initial_breaches": o.initial_quality["breaches"], "final_breaches": o.final_quality["breaches"],
                     "comparisons": [{k: c[k] for k in ("attempt", "action", "accepted", "reason", "signals", "evaluation_frames")}
                                     for c in o.comparisons],
                     "review": o.review_id is not None})
    plans = [s.decision for s in trace.steps if s.state == "PLAN_RECOVERY"]
    elapsed = time.time() - t0
    n_analysed = sum(len(r["frames"]) for r in agent.tools.results.values())
    return {"name": name, "calibration_valid": True, "segments": segs, "actions": plans,
            "max_attempts": max([s["attempts"] for s in segs] or [0]), "elapsed_s": round(elapsed, 1),
            "frames": meta.frame_count, "analysed_frames": n_analysed,
            "s_per_analysed_frame": round(elapsed / max(1, n_analysed), 3),
            "_agent": agent, "_ctx": ctx}


# --------------------------------------------------------------------- identity

def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)


def identity_eval(result: dict) -> dict:
    """For each labelled person: which predicted track covers it on each labelled frame (IoU >= 0.5)."""
    obs = {}
    for tr in result["tracks"]:
        for o in tr["obs"]:
            obs.setdefault(o["frame"], []).append((tr["tid"], o["box"]))
    per_gt, matched, total = {}, 0, 0
    for f, rows in sorted(IDL["frames"].items(), key=lambda kv: int(kv[0])):
        for r in rows:
            total += 1
            best = max(obs.get(int(f), []), key=lambda tb: iou(tb[1], r["box"]), default=None)
            tid = best[0] if best and iou(best[1], r["box"]) >= 0.5 else None
            matched += tid is not None
            per_gt.setdefault(r["gt_id"], []).append(tid)
    switches = frags = 0
    for seq in per_gt.values():
        ids = [t for t in seq if t is not None]
        switches += sum(1 for a, b in zip(ids, ids[1:]) if a != b)
        frags += max(0, len(set(ids)) - 1)
    # purity: does any predicted track cover two different labelled people?
    owners = {}
    for g, seq in per_gt.items():
        for t in seq:
            if t is not None:
                owners.setdefault(t, set()).add(g)
    merged = sum(1 for s in owners.values() if len(s) > 1)
    return {"labelled_boxes": total, "coverage": round(matched / total, 3), "id_switches": switches,
            "extra_ids_per_person": round(frags / len(per_gt), 3), "tracks_covering_multiple_people": merged,
            "per_person": {g: [t.split("-")[-1] if t else None for t in seq] for g, seq in per_gt.items()}}


# --------------------------------------------------------------------- calibration

def calibration_eval(ctx: RunContext, agent: SegmentAgent | None) -> dict:
    H0 = ctx.H_ref
    rows = []
    for h in LM["calibration"]["holdout"]:
        e = float(np.linalg.norm(cal.to_pitch(H0, [h["image"]])[0] - h["pitch"]))
        rows.append({"frame": 0, "name": h["name"], "static_error_m": round(e, 2), "updated_error_m": round(e, 2)})
    model = {}
    if agent is not None:
        for m in agent.tools.motions.values():
            model.update(m["model"])
    for fi_s, pts in LM["moved_frames"].items():
        fi = int(fi_s)
        frame = read_frame(ctx.video_path, fi)
        est = ctx.matcher.match_to_reference(frame, [])
        M = np.asarray(model[fi_s]) if fi_s in model else (np.asarray(est.M) if est.validated else None)
        for p in pts:
            es = float(np.linalg.norm(cal.to_pitch(H0, [p["image"]])[0] - p["pitch"]))
            eu = float(np.linalg.norm(cal.to_pitch(H0 @ M, [p["image"]])[0] - p["pitch"])) if M is not None else None
            rows.append({"frame": fi, "name": p["name"], "static_error_m": round(es, 2),
                         "updated_error_m": None if eu is None else round(eu, 2),
                         "motion_source": "agent_motion_model" if fi_s in model else est.method})
    moved = [r for r in rows if r["frame"] != 0]

    def rmse(k):
        v = [r[k] for r in moved if r[k] is not None]
        return round(float(np.sqrt(np.mean(np.square(v)))), 2) if v else None
    return {"points": rows, "frame0_holdout_rmse_m": round(float(np.sqrt(np.mean([r["static_error_m"] ** 2 for r in rows if r["frame"] == 0]))), 2),
            "moved_frames_rmse_static_m": rmse("static_error_m"), "moved_frames_rmse_updated_m": rmse("updated_error_m"),
            "note": "Moved-frame points include the four calibration landmarks re-annotated on later frames, so the "
                    "'updated' error isolates camera-motion error plus annotation noise; frame-0 rows are true held-out points."}


def motion_gt_eval(video: Path, gt: dict[int, np.ndarray], ctx: RunContext) -> dict:
    """Synthetic pan: compare estimated frame->reference homography with the known one (corner error, px)."""
    errs = []
    corners = np.float64([[0, 0], [1919, 0], [1919, 1079], [0, 1079], [960, 800]])
    for fi, frame in read_frames(video, sorted(gt)):
        est = ctx.matcher.match_to_reference(frame, [])
        if not est.validated:
            errs.append(None)
            continue
        a = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), np.asarray(est.M)).reshape(-1, 2)
        b = cv2.perspectiveTransform(corners.reshape(-1, 1, 2), gt[fi]).reshape(-1, 2)
        errs.append(float(np.max(np.linalg.norm(a - b, axis=1))))
    v = [e for e in errs if e is not None]
    return {"frames": len(errs), "validated": len(v), "max_corner_error_px_median": round(float(np.median(v)), 2) if v else None,
            "max_corner_error_px_p95": round(float(np.percentile(v, 95)), 2) if v else None}


# --------------------------------------------------------------------- synthetic clips

def write_clip(path: Path, frames, fps=30.0):
    path.parent.mkdir(parents=True, exist_ok=True)
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    for f in frames:
        w.write(f)
    w.release()
    return path


def pan_h(t: float) -> np.ndarray:
    """Known synthetic camera motion: maps REFERENCE pixels -> pixels of the frame at time t (s)."""
    dx, dy, rot = -12.0 * t, 3.0 * np.sin(t), np.radians(0.25 * t)
    c, s = np.cos(rot), np.sin(rot)
    A = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    T = np.array([[1, 0, dx], [0, 1, dy], [0, 0, 1]])
    C = np.array([[1, 0, -960], [0, 1, -540], [0, 0, 1]])
    return np.linalg.inv(C) @ T @ A @ C


def build_synthetic():
    rng = np.random.default_rng(7)
    f0 = read_frame(CLIP, 0)
    n = 360  # 12 s at 30 fps
    syn = OUT / "synthetic"
    clips = {}
    clips["control_static"] = write_clip(syn / "control_static.mp4",
                                         (np.clip(f0 + rng.normal(0, 2, f0.shape), 0, 255).astype(np.uint8) for _ in range(n)))
    gt = {}

    def pan_frames(textureless=False):
        fm = fieldmod.field_mask(f0)
        for i in range(n):
            Hm = pan_h(i / 30)
            gt[i] = np.linalg.inv(Hm)  # frame -> reference
            src = f0
            if textureless:
                src = f0.copy()
                src[fm == 0] = (90, 90, 90)
                src = cv2.GaussianBlur(src, (0, 0), 3)
            fr = cv2.warpPerspective(src, Hm, (1920, 1080), borderMode=cv2.BORDER_REPLICATE)
            if textureless:
                fr = np.clip(fr + rng.normal(0, 18, fr.shape), 0, 255).astype(np.uint8)
            yield fr
    clips["synthetic_pan"] = write_clip(syn / "synthetic_pan.mp4", pan_frames())
    gt_pan = dict(gt)
    clips["textureless_pan"] = write_clip(syn / "textureless_pan.mp4", pan_frames(True))
    lat = ROOT / "data/clips/latvia_gibraltar_2026-03-31_cc0.webm"

    def cut_frames():
        for _, f in read_frames(CLIP, list(range(0, 180))):
            yield f
        for _, f in read_frames(lat, list(range(0, 180))):
            yield cv2.resize(f, (1920, 1080))
    clips["scene_cut"] = write_clip(syn / "scene_cut.mp4", cut_frames())
    clips["time_compressed"] = write_clip(syn / "time_compressed.mp4", (f for i, f in read_frames(CLIP, list(range(0, 508, 3)))))
    # Added after the 3x result (densify improved fragmentation by 9%, below the 10% bar): a harsher sampling stress.
    clips["time_compressed_5x"] = write_clip(syn / "time_compressed_5x.mp4", (f for i, f in read_frames(CLIP, list(range(0, 508, 5)))))
    return clips, gt_pan


def expected_ok(name: str, r: dict) -> bool:
    acts = r["actions"]
    if name == "control_static":
        return set(acts) == {"accept"}
    if name in ("synthetic_pan", "natural"):
        return "camera_motion_update" in acts
    if name in ("time_compressed", "time_compressed_5x"):
        return "densify_sampling" in acts
    if name in ("textureless_pan", "shifted_points"):
        # Escalation can come from PLAN_RECOVERY or from a rejected correction at COMPARE.
        return "request_human_review" in acts or any(s["review"] for s in r["segments"])
    if name == "scene_cut":
        segs = r["segments"]
        return segs[0]["status"] in ("accepted", "recovered") and segs[-1]["status"] == "uncertain"
    if name == "swapped_points":
        return not r["calibration_valid"]
    return False


EXPECTED = {"natural": "camera_motion_update", "control_static": "accept only", "synthetic_pan": "camera_motion_update",
            "time_compressed": "densify_sampling", "textureless_pan": "request_human_review",
            "time_compressed_5x": "densify_sampling",
            "scene_cut": "seg1 fixed/accepted; post-cut segment escalated",
            "swapped_points": "calibration rejected", "shifted_points": "request_human_review"}


def strip(r):
    return {k: v for k, v in r.items() if not k.startswith("_")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    only = set(filter(None, a.only.split(",")))
    want = lambda k: not only or k in only  # noqa: E731
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"hardware": {"platform": platform.platform(), "machine": platform.machine(), "python": platform.python_version(),
                        "cpu": _cpu(), "opencv": cv2.__version__, "numpy": np.__version__},
           "cases": {}}
    t_all = time.time()
    if want("natural") or want("identity") or want("calibration"):
        r = run_agent("natural", CLIP, calibration_for(CLIP))
        res["cases"]["natural"] = strip(r) | {"kind": "natural", "expected": EXPECTED["natural"], "trigger_correct": expected_ok("natural", r)}
        agent, ctx = r["_agent"], r["_ctx"]
        seg1 = [o for o in agent.outcomes if o.segment_id == "seg1"][0]
        res["identity"] = {"initial": identity_eval(agent.tools.results[seg1.initial_result]),
                           "corrected": identity_eval(agent.tools.results[seg1.final_result]),
                           "labels": IDL["annotator"], "scope": IDL["scope"]}
        res["calibration"] = calibration_eval(ctx, agent)
    if any(want(k) for k in ("control_static", "synthetic_pan", "time_compressed", "textureless_pan", "scene_cut",
                             "time_compressed_5x")):
        clips, gt = build_synthetic()
        for name, path in clips.items():
            if not want(name) and only:
                continue
            r = run_agent(name, path, calibration_for(path))
            res["cases"][name] = strip(r) | {"kind": "synthetic", "expected": EXPECTED[name], "trigger_correct": expected_ok(name, r)}
            if name == "synthetic_pan":
                res["cases"][name]["motion_vs_ground_truth"] = motion_gt_eval(path, {k: v for k, v in gt.items() if k % 15 == 0}, r["_ctx"])
    if want("swapped_points"):
        img = [CAL_IMG[0], CAL_IMG[2], CAL_IMG[1], CAL_IMG[3]]
        r = run_agent("swapped_points", CLIP, calibration_for(CLIP, img=img))
        res["cases"]["swapped_points"] = r | {"kind": "synthetic", "expected": EXPECTED["swapped_points"], "trigger_correct": expected_ok("swapped_points", r)}
    if want("shifted_points"):
        img = [[x + 40, y] for x, y in CAL_IMG]
        r = run_agent("shifted_points", CLIP, calibration_for(CLIP, img=img))
        res["cases"]["shifted_points"] = strip(r) | {"kind": "synthetic", "expected": EXPECTED["shifted_points"], "trigger_correct": expected_ok("shifted_points", r)}
    prev_path = OUT / "results.json"
    if only and prev_path.exists():
        prev = json.loads(prev_path.read_text())
        for k in ("identity", "calibration"):
            if k not in res and k in prev:
                res[k] = prev[k]
        for k, v in prev.get("cases", {}).items():
            res["cases"].setdefault(k, v)
        for k, v in res["cases"].items():
            v["trigger_correct"] = expected_ok(k, v)
    cases = res["cases"]
    res["trigger_accuracy"] = {"correct": sum(c["trigger_correct"] for c in cases.values()), "total": len(cases)}
    res["retry_limit"] = {"max_attempts_observed": max([c.get("max_attempts", 0) for c in cases.values()] or [0]), "limit": 2}
    imp = []
    for name, c in cases.items():
        for s in c.get("segments", []):
            for cmp in s["comparisons"]:
                imp.append({"case": name, "segment": s["segment_id"], "action": cmp["action"], "accepted": cmp["accepted"],
                            "signals": {k: {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items()}
                                        for k, v in cmp["signals"].items()}})
    res["recoveries"] = imp
    res["total_elapsed_s"] = round(time.time() - t_all, 1)
    (OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
    (ROOT / "eval/RESULTS.md").write_text(render_md(res))
    print(render_md(res))


def _cpu():
    try:
        if platform.system() == "Darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor()


def render_md(r: dict) -> str:
    L = ["# Evaluation results", "", "Generated by `eval/run_eval.py`; numbers below are copied from `eval/out/results.json`.", "",
         f"Hardware: {r['hardware']['cpu']} · {r['hardware']['platform']} · OpenCV {r['hardware']['opencv']} · "
         f"total harness time {r.get('total_elapsed_s')} s", ""]
    L += ["## Agent trigger accuracy", "", "| Case | Kind | Expected | Plans chosen (per segment, in order) | Correct | Max attempts | Time (s) | s / analysed frame |",
          "|---|---|---|---|---|---|---|---|"]
    for n, c in r["cases"].items():
        L.append(f"| {n} | {c['kind']} | {c['expected']} | {', '.join(c['actions'])} | {'yes' if c['trigger_correct'] else '**no**'} | "
                 f"{c.get('max_attempts', '-')} | {c['elapsed_s']} | {c.get('s_per_analysed_frame', '-')} |")
    ta = r["trigger_accuracy"]
    L += ["", f"Trigger accuracy: **{ta['correct']}/{ta['total']}**. Retry limit: max attempts observed "
              f"{r['retry_limit']['max_attempts_observed']} (limit {r['retry_limit']['limit']}).", ""]
    L += ["## Did each recovery improve its target signals? (identical evaluation frames)", "",
          "| Case | Segment | Action | Accepted | Target signals before → after |", "|---|---|---|---|---|"]
    for x in r["recoveries"]:
        sig = "; ".join(f"{k}: {_f(v['before'])} → {_f(v['after'])}" for k, v in x["signals"].items())
        L.append(f"| {x['case']} | {x['segment']} | {x['action']} | {'yes' if x['accepted'] else 'no'} | {sig} |")
    if "identity" in r:
        L += ["", "## Identity consistency on manually labelled segment (seg1, frames 0–90)", "",
              f"Labels: {r['identity']['labels']}. {r['identity']['scope']}", "",
              "| Run | Coverage | ID switches | Extra IDs / person | Tracks covering >1 person |", "|---|---|---|---|---|"]
        for k in ("initial", "corrected"):
            v = r["identity"][k]
            L.append(f"| {k} | {v['coverage']} | {v['id_switches']} | {v['extra_ids_per_person']} | {v['tracks_covering_multiple_people']} |")
    if "calibration" in r:
        c = r["calibration"]
        L += ["", "## Calibration reprojection error (metres)", "",
              f"Frame-0 held-out RMSE: **{c['frame0_holdout_rmse_m']} m**. Moved frames (150/300/450): static calibration RMSE "
              f"**{c['moved_frames_rmse_static_m']} m** vs motion-updated **{c['moved_frames_rmse_updated_m']} m**.", "", c["note"], "",
              "| Frame | Landmark | Static (m) | Updated (m) |", "|---|---|---|---|"]
        for p in c["points"]:
            L.append(f"| {p['frame']} | {p['name']} | {p['static_error_m']} | {p['updated_error_m']} |")
    sp = r["cases"].get("synthetic_pan", {}).get("motion_vs_ground_truth")
    if sp:
        L += ["", "## Camera-motion estimate vs known synthetic motion", "",
              f"{sp['validated']}/{sp['frames']} probe frames validated; max corner error median {sp['max_corner_error_px_median']} px, "
              f"p95 {sp['max_corner_error_px_p95']} px (1920×1080)."]
    return "\n".join(L) + "\n"


def _f(v):
    return "n/a" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v))


if __name__ == "__main__":
    main()
