import { useEffect, useMemo, useRef, useState } from "react";
import { api, AppConfig, CalibrationReport, frameUrl, Job } from "./api";
import { projectTemplate } from "./geometry";
import { MiniPitch } from "./Pitch";

type Mode = "preset" | "custom" | "landmarks";
type Pt = [number, number];
type Holdout = { name: string; image: Pt };

const POINT_COLORS = ["#ff4d6d", "#ffd166", "#06d6a0", "#4cc9f0"];

function rectPoints(x0: number, y0: number, w: number, h: number): Pt[] {
  return [[x0, y0], [x0 + w, y0], [x0 + w, y0 + h], [x0, y0 + h]];
}

export function Calibrate({ job, config, onDone }: { job: Job; config: AppConfig; onDone: (j: Job) => void }) {
  const meta = job.meta;
  const prev = job.calibration;
  const [frame, setFrame] = useState(prev?.frame_index ?? 0);
  // Restore how a saved calibration was defined: named landmarks if every point matches one, else a custom rectangle.
  const savedLandmarks = prev?.pitch_points?.map((p) =>
    Object.entries(config.landmarks).find(([, q]) => Math.abs(q[0] - p[0]) < 1e-3 && Math.abs(q[1] - p[1]) < 1e-3)?.[0]);
  const savedMode: Mode = !prev ? "preset" : savedLandmarks?.every(Boolean) ? "landmarks" : "custom";
  const [mode, setMode] = useState<Mode>(savedMode);
  const [preset, setPreset] = useState("penalty_area");
  const [end, setEnd] = useState<"left" | "right">("left");
  const [pitchL, setPitchL] = useState(prev?.pitch_size?.[0] ?? 105);
  const [pitchW, setPitchW] = useState(prev?.pitch_size?.[1] ?? 68);
  const pp = prev?.pitch_points;
  const [custom, setCustom] = useState(savedMode === "custom" && pp
    ? { x0: pp[0][0], y0: pp[0][1], w: pp[1][0] - pp[0][0], h: pp[3][1] - pp[0][1] }
    : { x0: 0, y0: 13.84, w: 16.5, h: 40.32 });
  const [lmNames, setLmNames] = useState<string[]>(savedMode === "landmarks" ? (savedLandmarks as string[]) : [
    "left_goal_area_near_goalline", "left_goal_area_near_front", "left_penalty_area_near_front", "left_penalty_area_near_goalline",
  ]);
  const [pts, setPts] = useState<Pt[]>(prev?.image_points?.map((p) => [p[0], p[1]] as Pt) ?? []);
  const [holdout, setHoldout] = useState<Holdout[]>([]);
  const [holdoutName, setHoldoutName] = useState("left_goal_near_post");
  const [placing, setPlacing] = useState<"calib" | "holdout">("calib");
  const [snap, setSnap] = useState(false);
  const [segs, setSegs] = useState<number[][]>([]);
  const [report, setReport] = useState<CalibrationReport | null>(prev);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [drag, setDrag] = useState<number | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);

  useEffect(() => {
    if (!snap) return;
    api.lines(job.id, frame).then((r) => setSegs(r.segments)).catch(() => setSegs([]));
  }, [snap, frame, job.id]);

  const pitchPts: Pt[] = useMemo(() => {
    if (mode === "landmarks") return lmNames.map((n) => (config.landmarks[n] ?? [0, 0]) as Pt);
    if (mode === "custom") return rectPoints(custom.x0, custom.y0, custom.w, custom.h);
    const p = config.presets[preset];
    const x0 = end === "left" ? 0 : pitchL - p.width_m;
    return rectPoints(x0, pitchW / 2 - p.height_m / 2, p.width_m, p.height_m);
  }, [mode, lmNames, custom, preset, end, pitchL, pitchW, config]);

  const toImg = (e: React.MouseEvent): Pt => {
    const r = svgRef.current!.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * meta.width, ((e.clientY - r.top) / r.height) * meta.height];
  };

  const snapPt = (p: Pt): Pt => {
    if (!snap || !segs.length) return p;
    let best: Pt | null = null, bd = 30;
    const near = segs.filter((s) => distSeg(p, s) < 50);
    for (let i = 0; i < near.length; i++)
      for (let j = i + 1; j < near.length; j++) {
        const q = intersect(near[i], near[j]);
        if (q && Math.hypot(q[0] - p[0], q[1] - p[1]) < bd) { best = q; bd = Math.hypot(q[0] - p[0], q[1] - p[1]); }
      }
    return best ?? p;
  };

  const onClick = (e: React.MouseEvent) => {
    if (drag !== null) return;
    const p = snapPt(toImg(e));
    if (placing === "holdout") { setHoldout([...holdout, { name: holdoutName, image: p }]); return; }
    if (pts.length < 4) setPts([...pts, p]);
  };

  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      const body: any = { frame_index: frame, image_points: pts, pitch_size: [pitchL, pitchW],
        label: mode === "preset" ? `${end} ${preset}` : mode };
      body.pitch_points = pitchPts;
      if (holdout.length) body.holdout = holdout.map((h) => ({ name: h.name, image: h.image, pitch: config.landmarks[h.name] }));
      const rep = await api.calibrate(job.id, body);
      setReport(rep);
      onDone(await api.job(job.id));
    } catch (e: any) { setErr(e.message); } finally { setBusy(false); }
  };

  const overlay = report?.valid && report.homography && report.frame_index === frame
    ? projectTemplate(report.homography, report.pitch_size[0], report.pitch_size[1], meta.width, meta.height) : [];
  const nextIdx = pts.length;
  const sx = meta.width / 1000;

  return (
    <div className="calib">
      <div className="calib-stage card">
        <div className="calib-toolbar">
          <label>Frame <input type="range" min={0} max={meta.frame_count - 1} value={frame}
            onChange={(e) => { setFrame(+e.target.value); setReport(null); }} />
            <span className="mono">{frame} · {(frame / meta.fps).toFixed(2)} s</span></label>
          <label className="chk"><input type="checkbox" checked={snap} onChange={(e) => setSnap(e.target.checked)} /> Snap to detected line corners</label>
          <div className="seg-ctl">
            <button className={placing === "calib" ? "on" : ""} onClick={() => setPlacing("calib")}>Calibration points</button>
            <button className={placing === "holdout" ? "on" : ""} onClick={() => setPlacing("holdout")}>Check points</button>
          </div>
        </div>
        <div className="frame-wrap">
          <img src={frameUrl(job.id, frame)} alt={`frame ${frame}`} draggable={false} />
          <svg ref={svgRef} viewBox={`0 0 ${meta.width} ${meta.height}`} onClick={onClick}
            onMouseMove={(e) => { if (drag !== null) { const p = toImg(e); setPts(pts.map((q, i) => (i === drag ? p : q))); } }}
            onMouseUp={() => setTimeout(() => setDrag(null), 0)}>
            {snap && segs.map((s, i) => <line key={i} x1={s[0]} y1={s[1]} x2={s[2]} y2={s[3]} className="seg" strokeWidth={sx * 1.5} />)}
            {overlay.map((r, i) => <polyline key={i} points={r.map((p) => p.join(",")).join(" ")} className="tmpl" strokeWidth={sx * 2.2} />)}
            {pts.length > 1 && <polygon points={pts.map((p) => p.join(",")).join(" ")} className="quad" strokeWidth={sx * 2} />}
            {pts.map((p, i) => (
              <g key={i} onMouseDown={(e) => { e.stopPropagation(); setDrag(i); }} style={{ cursor: "grab" }}>
                <circle cx={p[0]} cy={p[1]} r={sx * 9} fill="none" stroke={POINT_COLORS[i]} strokeWidth={sx * 3} />
                <circle cx={p[0]} cy={p[1]} r={sx * 2.5} fill={POINT_COLORS[i]} />
                <text x={p[0] + sx * 12} y={p[1] - sx * 10} fill={POINT_COLORS[i]} fontSize={sx * 22} fontWeight={700}>P{i + 1}</text>
              </g>
            ))}
            {holdout.map((h, i) => (
              <g key={`h${i}`}>
                <rect x={h.image[0] - sx * 6} y={h.image[1] - sx * 6} width={sx * 12} height={sx * 12} fill="none" stroke="#fff" strokeWidth={sx * 2} />
                <text x={h.image[0] + sx * 9} y={h.image[1] + sx * 20} fill="#fff" fontSize={sx * 16}>{h.name}</text>
              </g>
            ))}
          </svg>
        </div>
        <p className="hint">
          {placing === "holdout"
            ? `Click a visible landmark to add it as a held-out check point (${holdoutName}). It is not used for fitting.`
            : nextIdx < 4
              ? <>Click <b style={{ color: POINT_COLORS[nextIdx] }}>P{nextIdx + 1}</b> — the image location of the pitch point highlighted in the diagram. Drag points to adjust.</>
              : "All four points placed. Drag to adjust, then validate."}
        </p>
      </div>

      <aside className="calib-side">
        <div className="card">
          <h3>1 · Known pitch region</h3>
          <div className="seg-ctl">
            {(["preset", "custom", "landmarks"] as Mode[]).map((m) => (
              <button key={m} className={mode === m ? "on" : ""} onClick={() => setMode(m)}>
                {m === "preset" ? "Marked rectangle" : m === "custom" ? "Custom rectangle" : "4 landmarks"}</button>
            ))}
          </div>
          {mode === "preset" && (
            <div className="form-grid">
              <label>Rectangle<select value={preset} onChange={(e) => setPreset(e.target.value)}>
                {Object.entries(config.presets).map(([k, v]) => <option key={k} value={k}>{v.label}</option>)}</select></label>
              <label>End<select value={end} onChange={(e) => setEnd(e.target.value as any)}>
                <option value="left">Left goal (x = 0)</option><option value="right">Right goal (x = L)</option></select></label>
            </div>
          )}
          {mode === "custom" && (
            <div className="form-grid four">
              {(["x0", "y0", "w", "h"] as const).map((k) => (
                <label key={k}>{k === "w" ? "width (m, along x)" : k === "h" ? "height (m, along y)" : `${k} (m)`}
                  <input type="number" step="0.01" value={custom[k]} onChange={(e) => setCustom({ ...custom, [k]: +e.target.value })} /></label>
              ))}
            </div>
          )}
          {mode === "landmarks" && (
            <div className="form-grid">
              {lmNames.map((n, i) => (
                <label key={i} style={{ color: POINT_COLORS[i] }}>P{i + 1}
                  <select value={n} onChange={(e) => setLmNames(lmNames.map((q, j) => (j === i ? e.target.value : q)))}>
                    {Object.keys(config.landmarks).map((k) => <option key={k} value={k}>{k}</option>)}</select></label>
              ))}
            </div>
          )}
          <div className="form-grid">
            <label>Pitch length L (m)<input type="number" value={pitchL} onChange={(e) => setPitchL(+e.target.value)} /></label>
            <label>Pitch width W (m)<input type="number" value={pitchW} onChange={(e) => setPitchW(+e.target.value)} /></label>
          </div>
          <MiniPitch L={pitchL} W={pitchW} points={pitchPts} colors={POINT_COLORS} next={nextIdx} />
          <p className="small muted">Pitch frame: x along the touchline from the left goal line, y from the far touchline
            to the near touchline. Go around the region in the order P1→P4.</p>
        </div>

        <div className="card">
          <h3>2 · Held-out check points <span className="tag">optional</span></h3>
          <label>Landmark<select value={holdoutName} onChange={(e) => setHoldoutName(e.target.value)}>
            {Object.keys(config.landmarks).map((k) => <option key={k} value={k}>{k}</option>)}</select></label>
          <p className="small muted">Switch to “Check points” and click the landmark. Reprojection error is reported in metres.</p>
          {holdout.length > 0 && <button className="ghost" onClick={() => setHoldout([])}>Clear {holdout.length} check point(s)</button>}
        </div>

        <div className="card">
          <div className="row">
            <button className="ghost" onClick={() => { setPts([]); setReport(null); }}>Reset points</button>
            <button className="primary" disabled={pts.length !== 4 || busy} onClick={submit}>{busy ? "Validating…" : "Validate calibration"}</button>
          </div>
          {err && <div className="alert error">{err}</div>}
          {report && <CalibReportView r={report} />}
        </div>
      </aside>
    </div>
  );
}

function CalibReportView({ r }: { r: CalibrationReport }) {
  const m = r.metrics ?? {};
  return (
    <div className="calib-report">
      <div className={`status-pill ${r.valid ? "ok" : "bad"}`}>{r.valid ? "Calibration valid" : "Calibration rejected"}</div>
      {r.errors.map((e, i) => <div key={i} className="alert error">{e}</div>)}
      {r.warnings.map((e, i) => <div key={i} className="alert warn">{e}</div>)}
      {r.valid && (
        <dl className="kv">
          <dt>Visible pitch</dt><dd>{(100 * (m.visible_pitch_fraction ?? 0)).toFixed(0)}% of area</dd>
          <dt>Visible extent</dt><dd>{(m.visible_length_m ?? 0).toFixed(0)} m × {(m.visible_width_m ?? 0).toFixed(0)} m</dd>
          <dt>Scale at region</dt><dd>{(m.px_per_m_x ?? 0).toFixed(1)} / {(m.px_per_m_y ?? 0).toFixed(1)} px/m</dd>
          {m.holdout_rmse_m !== undefined && <><dt>Held-out RMSE</dt><dd>{m.holdout_rmse_m.toFixed(2)} m ({m.holdout.length} pts)</dd></>}
        </dl>
      )}
      {r.valid && <p className="small muted">Magenta overlay: the full pitch template projected through this calibration. Check it against painted lines — error grows with distance from the calibrated region.</p>}
    </div>
  );
}

function distSeg(p: Pt, s: number[]) {
  const [ax, ay, bx, by] = s;
  const dx = bx - ax, dy = by - ay;
  const t = Math.max(0, Math.min(1, ((p[0] - ax) * dx + (p[1] - ay) * dy) / Math.max(1e-9, dx * dx + dy * dy)));
  return Math.hypot(ax + t * dx - p[0], ay + t * dy - p[1]);
}

function intersect(a: number[], b: number[]): Pt | null {
  const [x1, y1, x2, y2] = a, [x3, y3, x4, y4] = b;
  const ang = Math.abs(((Math.atan2(y2 - y1, x2 - x1) - Math.atan2(y4 - y3, x4 - x3)) * 180) / Math.PI) % 180;
  if (Math.min(ang, 180 - ang) < 12) return null;
  const den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4);
  if (Math.abs(den) < 1e-9) return null;
  const t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / den;
  return [x1 + t * (x2 - x1), y1 + t * (y2 - y1)];
}
