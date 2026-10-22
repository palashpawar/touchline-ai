"""Assemble per-segment results into clip-level tracks, team assignment, tactical metrics and renders."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from .pipeline import RunContext
from .vision import calibration as cal
from .vision import render
from .vision.teams import assign_teams
from .vision.video import read_frames

HEAT_CELL_M = 5.0
MIN_TEAM_PLAYERS = 4
TEAM_CONF_MIN = 0.5
EXTRAPOLATION_M = 30.0


def _foot(box):
    return np.array([(box[0] + box[2]) / 2, box[3]])


def stitch(segments: list[dict]) -> list[dict]:
    """Join tracks across segment boundaries by Hungarian matching of boundary observations.

    Candidates: tracks whose last observation is the last sampled frame of segment k, and tracks whose first
    observation is the first sampled frame of segment k+1 (adjacent frames). Cost: foot-point distance in raw
    image pixels normalised by box height; gate 0.75 box heights. Raw image coordinates are used because
    adjacent frames differ by at most one frame interval, independent of each segment's configuration.
    """
    out, open_tracks = [], {}
    gid = 0
    prev_last_frame = None
    for seg in segments:
        frames = [f["frame"] for f in seg["frames"]]
        first_frame = frames[0] if frames else None
        starters = [t for t in seg["tracks"] if t["obs"][0]["frame"] == first_frame]
        enders = [g for g, tr in open_tracks.items() if tr["obs"][-1]["frame"] == prev_last_frame]
        links = {}
        if starters and enders and prev_last_frame is not None and first_frame - prev_last_frame <= 2:
            C = np.full((len(enders), len(starters)), 1e6)
            for i, g in enumerate(enders):
                a = open_tracks[g]["obs"][-1]
                for j, t in enumerate(starters):
                    b = t["obs"][0]
                    hh = max(8.0, 0.5 * ((a["box"][3] - a["box"][1]) + (b["box"][3] - b["box"][1])))
                    d = np.linalg.norm(_foot(a["box"]) - _foot(b["box"])) / hh
                    if d < 0.75:
                        C[i, j] = d
            r, c = linear_sum_assignment(C)
            for i, j in zip(r, c):
                if C[i, j] < 1e6:
                    links[starters[j]["tid"]] = enders[i]
        new_open = {}
        for t in seg["tracks"]:
            obs = [dict(o, segment=seg["segment_id"]) for o in t["obs"]]
            if t["tid"] in links:
                g = links[t["tid"]]
                open_tracks[g]["obs"] += obs
                open_tracks[g]["source_tracks"].append(t["tid"])
                new_open[g] = open_tracks[g]
            else:
                gid += 1
                tr = {"id": gid, "obs": obs, "source_tracks": [t["tid"]]}
                out.append(tr)
                new_open[gid] = tr
        open_tracks = new_open
        prev_last_frame = frames[-1] if frames else prev_last_frame
    return out


def assemble(ctx: RunContext, segment_results: list[dict], statuses: dict[str, str],
             team_colors: list[str] | None = None, team_overrides: dict[int, str] | None = None) -> dict:
    L, W = ctx.pitch_size
    fps = ctx.meta.fps
    tracks = stitch(segment_results)
    feats = {t["id"]: [o["team_feat"] for o in t["obs"] if o.get("team_feat")] for t in tracks}
    teams = assign_teams(feats, team_colors=team_colors, overrides=team_overrides)
    seg_by_id = {s["segment_id"]: s for s in segment_results}
    dt_by_seg = {s["segment_id"]: s["config"]["stride"] / fps for s in segment_results}
    metric_ok = ctx.valid_calibration

    frames_index: dict[int, dict] = {}
    for s in segment_results:
        for f in s["frames"]:
            frames_index[f["frame"]] = {"frame": f["frame"], "t": f["t"], "segment": s["segment_id"],
                                        "status": statuses.get(s["segment_id"], "accepted"), "players": []}
    trails = []
    heat = {k: np.zeros((int(np.ceil(W / HEAT_CELL_M)), int(np.ceil(L / HEAT_CELL_M)))) for k in ("all", "A", "B")}
    for t in tracks:
        info = teams["tracks"].get(t["id"], {"team": "unknown", "confidence": 0.0})
        t["team"], t["team_confidence"] = info["team"], info["confidence"]
        pts = []
        for o in t["obs"]:
            fr = frames_index.get(o["frame"])
            entry = {"id": t["id"], "team": t["team"], "team_confidence": t["team_confidence"], "box": o["box"]}
            if o["pitch"] is not None and metric_ok:
                x, y = o["pitch"]
                entry.update(x=x, y=y)
                pts.append([x, y])
                cx, cy = int(np.clip(x // HEAT_CELL_M, 0, heat["all"].shape[1] - 1)), int(np.clip(y // HEAT_CELL_M, 0, heat["all"].shape[0] - 1))
                if 0 <= x <= L and 0 <= y <= W:
                    dt = dt_by_seg[o["segment"]]
                    heat["all"][cy, cx] += dt
                    if t["team"] in ("A", "B") and t["team_confidence"] >= TEAM_CONF_MIN:
                        heat[t["team"]][cy, cx] += dt
            if fr is not None:
                fr["players"].append(entry)
        if pts:
            trails.append({"id": t["id"], "team": t["team"], "points": pts})

    quad = ctx.quad_pitch if len(ctx.quad_pitch) == 4 else None
    shape = []
    for fi in sorted(frames_index):
        fr = frames_index[fi]
        row = {"frame": fi, "t": fr["t"], "segment": fr["segment"], "status": fr["status"]}
        for team in ("A", "B"):
            ps = [p for p in fr["players"] if p["team"] == team and p["team_confidence"] >= TEAM_CONF_MIN and "x" in p]
            row[f"{team}_n"] = len(ps)
            if metric_ok and len(ps) >= MIN_TEAM_PLAYERS:
                xy = np.array([[p["x"], p["y"]] for p in ps])
                row[f"{team}_width_m"] = float(xy[:, 1].max() - xy[:, 1].min())
                row[f"{team}_depth_m"] = float(xy[:, 0].max() - xy[:, 0].min())
                if quad is not None:
                    row[f"{team}_extrapolated_share"] = float(np.mean(cal.distance_outside_polygon(xy, quad) > EXTRAPOLATION_M))
        shape.append(row)

    def summarise(team):
        rows = [r for r in shape if f"{team}_width_m" in r and r["status"] != "uncertain"]
        if not rows:
            return {"available": False, "reason": (
                "calibration invalid" if not metric_ok else
                f"fewer than {MIN_TEAM_PLAYERS} confidently assigned, projected players in every reliable frame")}
        return {"available": True, "frames": len(rows),
                "width_m_median": float(np.median([r[f"{team}_width_m"] for r in rows])),
                "depth_m_median": float(np.median([r[f"{team}_depth_m"] for r in rows])),
                "visible_players_median": float(np.median([r[f"{team}_n"] for r in rows])),
                "extrapolated_share_median": float(np.median([r.get(f"{team}_extrapolated_share", 0) for r in rows])),
                "caveat": "Computed from players visible in this camera view only (typically fewer than 11); "
                          "a partial view understates true team width/depth. Positions >30 m from the calibrated "
                          "rectangle carry larger projection error."}

    return {
        "tracks": [{"id": t["id"], "team": t["team"], "team_confidence": t["team_confidence"],
                    "n_obs": len(t["obs"]), "first_frame": t["obs"][0]["frame"], "last_frame": t["obs"][-1]["frame"],
                    "source_tracks": t["source_tracks"],
                    "obs": [{"frame": o["frame"], "t": o["t"], "box": o["box"], "pitch": o["pitch"] if metric_ok else None}
                            for o in t["obs"]]} for t in tracks],
        "teams": {"method": teams["method"], "centres": teams["centres"],
                  "note": "Team labels come from jersey colour clustering; confidence is a heuristic agreement score, not a probability."},
        "frames": [dict(frames_index[k]) for k in sorted(frames_index)],
        "trails": trails if metric_ok else [],
        "team_shape": {"per_frame": shape, "A": summarise("A"), "B": summarise("B")},
        "heatmap": {"available": metric_ok, "cell_m": HEAT_CELL_M, "unit": "player-seconds per cell",
                    "definition": "Each observation of a player inside the pitch adds the segment's sampling interval "
                                  "(stride / fps seconds) to the 5 m x 5 m cell containing its projected ground point.",
                    "grids": {k: v.round(3).tolist() for k, v in heat.items()} if metric_ok else {}},
        "unavailable": {
            "possession": "not implemented", "passing": "not implemented", "ball_tracking": "not implemented",
            "player_identity": "not implemented (track IDs are anonymous and can switch)",
            "speed_distance": "not reported: sampled-frame projection noise is too large for per-player kinematics"},
        "_segments": {k: {"config": v["config"]} for k, v in seg_by_id.items()},
    }


def _interp_boxes(track_obs: list[dict], frame: int):
    for a, b in zip(track_obs, track_obs[1:]):
        if a["frame"] <= frame <= b["frame"] and b["frame"] - a["frame"] <= 12:
            w = (frame - a["frame"]) / max(1, b["frame"] - a["frame"])
            return (1 - w) * np.asarray(a["box"]) + w * np.asarray(b["box"]), (1 - w) * np.asarray(a["pitch"] or [np.nan, np.nan]) + w * np.asarray(b["pitch"] or [np.nan, np.nan])
    if track_obs and track_obs[-1]["frame"] == frame:
        o = track_obs[-1]
        return np.asarray(o["box"]), np.asarray(o["pitch"] or [np.nan, np.nan])
    return None, None


def render_video(ctx: RunContext, assembled: dict, segment_results: list[dict], out_path: Path,
                 title: str, out_width: int = 960, statuses: dict[str, str] | None = None, frame_step: int = 2) -> Path:
    """Render every `frame_step`-th frame at fps/frame_step (same duration). OpenCV's FFmpeg writer exposes no
    bitrate control, so resolution and frame rate are how we bound artifact size."""
    statuses = statuses or {}
    H_by_frame = {}
    seg_of = {}
    for s in segment_results:
        for f in s["frames"]:
            if f.get("H") is not None:
                H_by_frame[f["frame"]] = np.asarray(f["H"])
        for fi in range(s["start"], s["end"]):
            seg_of[fi] = s
    keys = np.array(sorted(H_by_frame)) if H_by_frame else np.array([])
    sc = out_width / ctx.meta.width
    size = (out_width, int(round(ctx.meta.height * sc)))
    writer, path = render.open_writer(out_path, ctx.meta.fps / frame_step, size)
    tracks = assembled["tracks"]
    trail_hist: dict[int, list] = {}
    for fi, frame in read_frames(ctx.video_path, list(range(0, ctx.meta.frame_count, frame_step))):
        img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        seg = seg_of.get(fi)
        if len(keys) and ctx.valid_calibration:
            k = keys[np.argmin(np.abs(keys - fi))]
            render.draw_pitch_projection(img, H_by_frame[int(k)], ctx.pitch_size, (255, 0, 255), 1, sc)
        players = []
        for t in tracks:
            if not (t["first_frame"] <= fi <= t["last_frame"]):
                continue
            box, pitch = _interp_boxes(t["obs"], fi)
            if box is None:
                continue
            col = render.TEAM_BGR.get(t["team"], (160, 160, 160))
            b = (box * sc).astype(int)
            cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), col, 1, cv2.LINE_AA)
            cv2.putText(img, str(t["id"]), (b[0], b[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.38, col, 1, cv2.LINE_AA)
            foot = (int((b[0] + b[2]) / 2), b[3])
            th = trail_hist.setdefault(t["id"], [])
            th.append(foot)
            del th[:-12]
            if len(th) > 1:
                cv2.polylines(img, [np.int32(th)], False, col, 1, cv2.LINE_AA)
            if np.all(np.isfinite(pitch)):
                players.append({"x": float(pitch[0]), "y": float(pitch[1]), "team": t["team"]})
        if ctx.valid_calibration:
            mm = render.minimap(players, ctx.pitch_size, px_per_m=1.9)
            mh, mw = mm.shape[:2]
            img[size[1] - mh - 10: size[1] - 10, size[0] - mw - 10: size[0] - 10] = mm
        st = statuses.get(seg["segment_id"], "") if seg else ""
        banner = f"{title} | {seg['segment_id'] if seg else ''} {st} | stride {seg['config']['stride'] if seg else '-'}" \
                 f"{' | motion-compensated' if seg and seg['config']['motion_compensation'] else ''}"
        cv2.rectangle(img, (0, 0), (size[0], 26), (40, 22, 12), -1)
        cv2.putText(img, banner, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (235, 235, 235), 1, cv2.LINE_AA)
        if st == "uncertain":
            cv2.putText(img, "UNCERTAIN - needs review", (size[0] - 250, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 170, 255), 1, cv2.LINE_AA)
        writer.write(img)
    writer.release()
    return path


def comparison_frame(ctx: RunContext, before: dict, after: dict, out_path: Path) -> dict | None:
    """Side-by-side render of the same frame under the before/after geometry and detections."""
    fb = {f["frame"]: f for f in before["frames"]}
    fa = {f["frame"]: f for f in after["frames"]}
    common = sorted(set(fb) & set(fa))
    if not common:
        return None
    fi = common[len(common) * 2 // 3]
    frame = next(f for _, f in read_frames(ctx.video_path, [fi]))
    panels = []
    for label, res, fr in (("BEFORE", before, fb[fi]), ("AFTER", after, fa[fi])):
        img = frame.copy()
        if fr.get("H") is not None:
            render.draw_pitch_projection(img, np.asarray(fr["H"]), ctx.pitch_size, (255, 0, 255), 3)
        for t in res["tracks"]:
            for o in t["obs"]:
                if o["frame"] == fi:
                    b = np.int32(o["box"])
                    cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (0, 255, 255), 2)
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        cfg = res["config"]
        txt = f"{label}: stride {cfg['stride']}, motion comp {'on' if cfg['motion_compensation'] else 'off'}, " \
              f"line align {fr['line_alignment']:.2f}" if fr.get("line_alignment") is not None else label
        cv2.rectangle(img, (0, 0), (960, 30), (40, 22, 12), -1)
        cv2.putText(img, txt, (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (235, 235, 235), 1, cv2.LINE_AA)
        panels.append(img)
    out = np.hstack(panels)
    cv2.imwrite(str(out_path), out, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return {"frame": fi, "path": str(out_path)}
