import { useMemo, useRef, useState } from "react";
import { api, AppConfig, artifactUrl, Job, Review, SignalSpec } from "./api";
import { FramePlayers, teamColor, TopDown } from "./Pitch";

const fmt = (v: any, d = 2) => (v === null || v === undefined ? "n/a" : typeof v === "number" ? v.toFixed(d) : String(v));

export function Results({ job, report, reviews, config, onAction, onRecalibrate }: {
  job: Job; report: any; reviews: Review[]; config: AppConfig;
  onAction: (j: Job) => void; onRecalibrate: () => void;
}) {
  const [which, setWhich] = useState<"final" | "initial">("final");
  const [frame, setFrame] = useState<FramePlayers | null>(null);
  const [showTrails, setShowTrails] = useState(true);
  const [heatKey, setHeatKey] = useState<"none" | "all" | "A" | "B">("none");
  const vref = useRef<HTMLVideoElement>(null);
  const res = report.results;
  const [L, W] = report.calibration.pitch_size ?? [105, 68];
  const frames: FramePlayers[] = res.frames;
  const frameIdx = useMemo(() => frames.map((f) => f.frame), [frames]);
  const fps = report.job.meta.fps;
  const vname = report.artifacts[which === "final" ? "annotated_final" : "annotated_initial"];

  const onTime = () => {
    const t = vref.current?.currentTime ?? 0;
    const fi = Math.round(t * fps);
    let best = 0;
    for (let k = 0; k < frameIdx.length; k++) if (Math.abs(frameIdx[k] - fi) < Math.abs(frameIdx[best] - fi)) best = k;
    setFrame(frames[best] ?? null);
  };

  const specs: Record<string, SignalSpec> = Object.fromEntries(config.signals.map((s) => [s.key, s]));
  const open = reviews.filter((r) => r.status === "open" && r.run_no === job.run_no);

  return (
    <div className="results">
      <section className="grid-2">
        <div className="card">
          <div className="card-head">
            <h3>Annotated video</h3>
            <div className="seg-ctl small">
              <button className={which === "final" ? "on" : ""} onClick={() => setWhich("final")}>Corrected</button>
              <button className={which === "initial" ? "on" : ""} onClick={() => setWhich("initial")}>Initial run</button>
            </div>
          </div>
          <video key={vname} ref={vref} src={artifactUrl(job.id, vname, job.run_no)} controls muted playsInline
            onTimeUpdate={onTime} onSeeked={onTime} onLoadedData={onTime} className="player" />
          <p className="small muted">Magenta: pitch template projected through the per-frame calibration. Boxes: tracked
            players coloured by estimated team. IDs are anonymous track numbers, not player identities.</p>
        </div>
        <div className="card">
          <div className="card-head">
            <h3>Top-down view</h3>
            <div className="row small">
              <label className="chk"><input type="checkbox" checked={showTrails} onChange={(e) => setShowTrails(e.target.checked)} /> Trails</label>
              <select value={heatKey} onChange={(e) => setHeatKey(e.target.value as any)}>
                <option value="none">No heatmap</option><option value="all">Heatmap: all players</option>
                <option value="A">Heatmap: team A</option><option value="B">Heatmap: team B</option>
              </select>
            </div>
          </div>
          {res.heatmap.available ? (
            <TopDown L={L} W={W} frame={frame} trails={res.trails} showTrails={showTrails}
              heat={heatKey === "none" ? null : res.heatmap.grids[heatKey]} heatCell={res.heatmap.cell_m}
              calibQuad={report.calibration.pitch_points} />
          ) : <div className="empty">Top-down positions are unavailable because the calibration is invalid.</div>}
          <div className="small muted">
            {frame ? <>Frame {frame.frame} · {frame.t.toFixed(2)} s · segment status <b>{frame.status}</b> · {frame.players.filter((p) => p.x !== undefined).length} projected players</> : "Play the video to sync positions."}
            {heatKey !== "none" && <> · Heatmap: {res.heatmap.definition}</>}
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>Tactical measurements</h3><span className="tag measured">measured · partial view</span></div>
        <div className="metric-grid">
          {(["A", "B"] as const).map((t) => {
            const s = res.team_shape[t];
            const si = report.initial_results.team_shape[t];
            return (
              <div key={t} className="metric-card" style={{ borderTopColor: teamColor(t) }}>
                <div className="metric-title"><span className="swatch" style={{ background: res.teams.centres?.[t] ?? teamColor(t) }} /> Team {t}</div>
                {s.available ? (
                  <>
                    <div className="metric-row"><span>Width (visible players)</span><b>{fmt(s.width_m_median, 1)} m</b>
                      <em>initial {fmt(si?.width_m_median, 1)} m</em></div>
                    <div className="metric-row"><span>Depth (visible players)</span><b>{fmt(s.depth_m_median, 1)} m</b>
                      <em>initial {fmt(si?.depth_m_median, 1)} m</em></div>
                    <div className="metric-row"><span>Visible players (median)</span><b>{fmt(s.visible_players_median, 0)} / 11</b></div>
                    <div className="metric-row"><span>Points &gt;30 m from calibration</span><b>{fmt(100 * s.extrapolated_share_median, 0)}%</b></div>
                    <p className="small muted">{s.caveat}</p>
                  </>
                ) : <div className="empty small">Unavailable: {s.reason}</div>}
              </div>
            );
          })}
          <div className="metric-card neutral">
            <div className="metric-title">Not measured</div>
            {Object.entries(res.unavailable).map(([k, v]) => <div key={k} className="metric-row"><span>{k.replace("_", " ")}</span><em>{String(v)}</em></div>)}
            <p className="small muted">Teams: {res.teams.note} Method: {res.teams.method}.</p>
          </div>
        </div>
      </section>

      <section className="card">
        <div className="card-head"><h3>Segment quality — initial vs corrected</h3><span className="tag heuristic">heuristic indicators, not probabilities</span></div>
        <div className="table-wrap">
          <table className="qtable">
            <thead>
              <tr><th>Signal</th><th>Threshold</th>{report.segments.map((s: any) => <th key={s.segment_id}>{s.segment_id}<br /><span className={`status ${s.status}`}>{s.status}</span></th>)}</tr>
            </thead>
            <tbody>
              {["line_alignment", "camera_drift_px", "out_of_pitch_fraction", "implausible_jump_rate", "short_track_ratio",
                "fragmentation_index", "assoc_residual_h", "detection_gap_fraction", "motion_inlier_ratio", "motion_reproj_px", "extrapolated_fraction"].map((k) => (
                <tr key={k}>
                  <td title={specs[k]?.definition}>{specs[k]?.label ?? k}<span className="unit">{specs[k]?.unit}</span></td>
                  <td className="mono">{specs[k]?.bad_when === "above" ? ">" : "<"} {specs[k]?.threshold}</td>
                  {report.segments.map((s: any) => {
                    const a = s.initial_quality.signals[k], b = s.final_quality.signals[k];
                    const ba = s.initial_quality.breaches.includes(k), bb = s.final_quality.breaches.includes(k);
                    return <td key={s.segment_id} className="mono"><span className={ba ? "breach" : ""}>{fmt(a)}</span>
                      {s.final_result !== s.initial_result || s.attempts ? <> → <span className={bb ? "breach" : "good"}>{fmt(b)}</span></> : null}</td>;
                  })}
                </tr>
              ))}
              <tr><td>Config</td><td />{report.segments.map((s: any) => (
                <td key={s.segment_id} className="mono small">stride {s.initial_config.stride}{s.initial_config.motion_compensation ? " +mc" : ""} → stride {s.final_config.stride}{s.final_config.motion_compensation ? " +mc" : ""}</td>))}</tr>
            </tbody>
          </table>
        </div>
        <p className="small muted">Hover a signal name for its definition. Red values breach the threshold. Corrections are compared on identical evaluation frames. Camera drift describes the camera, not the result: once motion compensation is applied it no longer counts as a breach, and pitch-line alignment checks the corrected geometry instead.</p>
      </section>

      {report.artifacts.comparisons?.length > 0 && (
        <section className="card">
          <div className="card-head"><h3>Before / after on the same frame</h3></div>
          <div className="compare-list">
            {report.artifacts.comparisons.map((c: any) => {
              const seg = report.segments.find((s: any) => s.segment_id === c.segment_id);
              const cmp = seg?.comparisons?.[seg.comparisons.length - 1];
              return (
                <figure key={c.segment_id}>
                  <img src={artifactUrl(job.id, c.artifact, job.run_no)} alt={`comparison ${c.segment_id}`} loading="lazy" />
                  <figcaption><b>{c.segment_id}</b> · frame {c.frame} · {cmp ? <>{cmp.action}: <span className={cmp.accepted ? "good" : "breach"}>{cmp.accepted ? "accepted" : "rejected"}</span> — {cmp.reason}</> : null}</figcaption>
                </figure>
              );
            })}
          </div>
        </section>
      )}

      <ReviewPanel job={job} reviews={open} resolved={reviews.filter((r) => r.status !== "open")} onAction={onAction} onRecalibrate={onRecalibrate} />
      <TeamPanel job={job} report={report} onAction={onAction} />

      <section className="card">
        <div className="card-head"><h3>Export</h3></div>
        <div className="row wrap">
          <a className="btn" href={artifactUrl(job.id, "report.json", job.run_no, true)} download>Report (JSON)</a>
          <a className="btn" href={artifactUrl(job.id, "trace.jsonl", job.run_no, true)} download>Agent trace (JSONL)</a>
          <a className="btn" href={artifactUrl(job.id, report.artifacts.annotated_final, job.run_no, true)}>Corrected video</a>
          <a className="btn ghost" href={artifactUrl(job.id, report.artifacts.annotated_initial, job.run_no, true)}>Initial video</a>
        </div>
        <p className="small muted">Processed in {report.provenance.processing_seconds} s · OpenCV {report.provenance.opencv} · {report.provenance.machine} · detector {report.provenance.detector.model}</p>
      </section>
    </div>
  );
}

function ReviewPanel({ job, reviews, resolved, onAction, onRecalibrate }: {
  job: Job; reviews: Review[]; resolved: Review[]; onAction: (j: Job) => void; onRecalibrate: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const act = async (r: Review, action: string) => {
    setBusy(r.id); setErr(null);
    try { onAction(await api.review(job.id, r.id, { action })); } catch (e: any) { setErr(e.message); } finally { setBusy(null); }
  };
  return (
    <section className={`card ${reviews.length ? "attention" : ""}`}>
      <div className="card-head"><h3>Human review</h3>{reviews.length > 0 && <span className="tag review">{reviews.length} open</span>}</div>
      {err && <div className="alert error">{err}</div>}
      {!reviews.length && <div className="empty small">No open review requests. The agent escalates here when an automatic correction cannot be validated or its retry budget is spent.</div>}
      {reviews.map((r) => (
        <div key={r.id} className="review-item">
          <div><span className="segtag">{r.segment_id ?? "clip"}</span> <b>{r.kind.replace("_", " ")}</b></div>
          <p>{r.reason}</p>
          {r.proposed_config && <p className="small mono">Proposed: stride {r.proposed_config.stride}{r.proposed_config.motion_compensation ? ", motion compensation on" : ""}</p>}
          <div className="row">
            {r.proposed_config && <button className="primary" disabled={!!busy} onClick={() => act(r, "approve_rerun")}>Approve proposed rerun</button>}
            <button className="ghost" onClick={onRecalibrate}>Correct field points</button>
            <button className="ghost" disabled={!!busy} onClick={() => act(r, "accept_uncertain")}>Accept as-is (flagged)</button>
          </div>
        </div>
      ))}
      {resolved.length > 0 && <details className="small"><summary>{resolved.length} resolved</summary>
        {resolved.map((r) => <div key={r.id} className="muted">{r.segment_id}: {r.kind} → {r.status} {r.resolution ? JSON.stringify(r.resolution) : ""}</div>)}</details>}
    </section>
  );
}

function TeamPanel({ job, report, onAction }: { job: Job; report: any; onAction: (j: Job) => void }) {
  const c = report.results.teams.centres ?? {};
  const [a, setA] = useState<string>(c.A ?? "#ffffff");
  const [b, setB] = useState<string>(c.B ?? "#1e3a8a");
  const [ov, setOv] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const submit = async () => {
    setErr(null);
    const overrides: Record<string, string> = {};
    for (const part of ov.split(",").map((s) => s.trim()).filter(Boolean)) {
      const m = part.match(/^(\d+)\s*[:=]\s*(A|B|other)$/i);
      if (!m) { setErr(`Could not parse "${part}". Use e.g. 12:A, 7:other`); return; }
      overrides[m[1]] = m[2].toUpperCase() === "OTHER" ? "other" : m[2].toUpperCase();
    }
    try { onAction(await api.review(job.id, "team", { action: "team_update", team_colors: [a, b], team_overrides: overrides })); }
    catch (e: any) { setErr(e.message); }
  };
  const low = report.results.tracks.filter((t: any) => t.team_confidence < 0.5 && t.n_obs >= 5).length;
  return (
    <section className="card">
      <div className="card-head"><h3>Team assignment</h3><span className="tag heuristic">colour heuristic</span></div>
      <p className="small muted">{report.results.tracks.length} tracks; {low} with confidence below 0.5. Set kit colours or correct individual track IDs (as shown in the video).</p>
      <div className="row wrap">
        <label>Team A kit <input type="color" value={a} onChange={(e) => setA(e.target.value)} /></label>
        <label>Team B kit <input type="color" value={b} onChange={(e) => setB(e.target.value)} /></label>
        <label className="grow">Track overrides <input placeholder="e.g. 12:A, 7:other" value={ov} onChange={(e) => setOv(e.target.value)} /></label>
        <button className="primary" onClick={submit}>Re-assign teams</button>
      </div>
      {err && <div className="alert error">{err}</div>}
    </section>
  );
}
