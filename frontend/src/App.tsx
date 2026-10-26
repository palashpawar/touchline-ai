import { useCallback, useEffect, useRef, useState } from "react";
import { AgentStep, api, ApiError, AppConfig, DEMO, Job, Review, setToken } from "./api";
import { Calibrate } from "./Calibrate";
import { Results } from "./Results";
import { Timeline } from "./Timeline";

type Step = "upload" | "calibrate" | "analyze" | "results";
const RUNNING = new Set(["queued", "running"]);

export default function App() {
  const [config, setConfig] = useState<AppConfig | null>(null);
  const [health, setHealth] = useState<any>(null);
  const [needAuth, setNeedAuth] = useState(false);
  const [job, setJob] = useState<Job | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [step, setStep] = useState<Step>("upload");
  const [trace, setTrace] = useState<AgentStep[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [report, setReport] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);

  const openJobRef = useRef<((j: Job) => Promise<void>) | null>(null);
  const boot = useCallback(async () => {
    try {
      setHealth(await api.health());
      setConfig(await api.config());
      const js = await api.jobs();
      setJobs(js);
      setNeedAuth(false);
      if (DEMO && js[0]) await openJobRef.current?.(js[0]);
    } catch (e) {
      if (e instanceof ApiError && e.status === 401) setNeedAuth(true);
      else setErr((e as Error).message);
    }
  }, []);
  useEffect(() => { boot(); }, [boot]);

  const openJob = useCallback(async (j: Job) => {
    setJob(j); setReport(null); setTrace([]); setReviews([]);
    if (j.run_no > 0) {
      setTrace(await api.trace(j.id).catch(() => []));
      setReviews(await api.reviews(j.id).catch(() => []));
      if (!RUNNING.has(j.status)) setReport(await api.report(j.id).catch(() => null));
    }
    setStep(j.run_no > 0 ? (RUNNING.has(j.status) ? "analyze" : "results") : j.calibration?.valid ? "analyze" : "calibrate");
  }, []);
  openJobRef.current = openJob;

  // Poll while a task is running: job status, live agent steps, then the report once finished.
  useEffect(() => {
    if (!job || !RUNNING.has(job.status)) return;
    const h = setInterval(async () => {
      try {
        const j = await api.job(job.id);
        setTrace(await api.trace(j.id));
        setJob(j);
        if (!RUNNING.has(j.status)) {
          setReviews(await api.reviews(j.id));
          if (j.status !== "failed") { setReport(await api.report(j.id)); setStep("results"); }
        }
      } catch (e) { setErr((e as Error).message); }
    }, 2000);
    return () => clearInterval(h);
  }, [job]);

  if (needAuth) return <Login onOk={boot} />;
  if (!config) return <div className="boot">{err ? <div className="alert error">{err}</div> : "Loading Touchline AI…"}</div>;

  const steps: [Step, string, boolean][] = [
    ["upload", "Upload", true],
    ["calibrate", "Calibrate", !!job],
    ["analyze", "Analyze", !!job?.calibration?.valid],
    ["results", "Results & review", !!report],
  ];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand"><span className="logo">◢</span> Touchline <b>AI</b></div>
        <nav className="stepper">
          {steps.map(([k, label, enabled], i) => (
            <button key={k} className={`stp ${step === k ? "on" : ""}`} disabled={!enabled} onClick={() => setStep(k)}>
              <span className="n">{i + 1}</span>{label}
            </button>
          ))}
        </nav>
        <div className="health">
          {health && <><span className={`dot ${health.status === "ok" ? "ok" : "bad"}`} /> OpenCV {health.opencv} · {health.storage}</>}
        </div>
      </header>

      {DEMO && (
        <div className="demo-banner">
          <b>Live demo · recorded run.</b> This is a real analysis processed on AWS (EC2 + S3,
          OpenCV {health?.opencv}) of a CC BY-SA clip. Explore every step; uploading and re-running are disabled here.
          <a href="https://github.com/palashpawar/touchline-ai" target="_blank" rel="noreferrer">Run it yourself →</a>
        </div>
      )}
      {err && <div className="alert error banner" onClick={() => setErr(null)}>{err} <span className="muted">(dismiss)</span></div>}

      <main>
        {job && (
          <div className="jobbar">
            <span className="mono">{job.filename}</span>
            <span className="muted">{job.meta.width}×{job.meta.height} · {job.meta.fps.toFixed(2)} fps · {job.meta.duration_s.toFixed(1)} s</span>
            <span className={`status ${job.status}`}>{job.status.replace("_", " ")}</span>
            {job.error && <span className="alert error small">{job.error.split("\n")[0]}</span>}
          </div>
        )}

        {step === "upload" && <Upload config={config} jobs={jobs} onJob={(j) => { setJobs([j, ...jobs]); openJob(j); }} onOpen={openJob} />}
        {step === "calibrate" && job && <Calibrate job={job} config={config} onDone={(j) => setJob(j)} />}
        {step === "analyze" && job && <Analyze job={job} trace={trace} onJob={setJob} onRecalibrate={() => setStep("calibrate")} />}
        {step === "results" && job && report && (
          <div className="results-layout">
            <Results job={job} report={report} reviews={reviews} config={config}
              onAction={(j) => { setJob(j); setStep("analyze"); }} onRecalibrate={() => setStep("calibrate")} />
            <aside className="card timeline-card"><h3>Agent activity</h3><Timeline steps={trace} live={false} /></aside>
          </div>
        )}
      </main>
      <footer className="foot muted small">
        Rule-based agent · OpenCV {health?.opencv} for decoding, segmentation, homography, optical flow, DNN inference and rendering ·
        Detector: YOLOX-S (Apache-2.0, OpenCV Zoo). Outputs are approximate; review before coaching decisions.
      </footer>
    </div>
  );
}

function Login({ onOk }: { onOk: () => void }) {
  const [t, setT] = useState("");
  const [e, setE] = useState<string | null>(null);
  return (
    <div className="boot">
      <form className="card login" onSubmit={async (ev) => { ev.preventDefault(); try { setToken(t); await api.login(t); onOk(); } catch (x: any) { setE(x.message); } }}>
        <h2>Touchline AI</h2>
        <p className="muted">This deployment is protected. Enter the access token from the deployment output.</p>
        <input type="password" value={t} onChange={(x) => setT(x.target.value)} placeholder="Access token" autoFocus />
        <button className="primary">Continue</button>
        {e && <div className="alert error">{e}</div>}
      </form>
    </div>
  );
}

function Upload({ config, jobs, onJob, onOpen }: { config: AppConfig; jobs: Job[]; onJob: (j: Job) => void; onOpen: (j: Job) => void }) {
  const [prog, setProg] = useState<number | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [over, setOver] = useState(false);
  const lim = config.limits;
  const send = async (f: File) => {
    setErr(null); setProg(0);
    try { onJob(await api.upload(f, setProg)); } catch (e: any) { setErr(e.message); } finally { setProg(null); }
  };
  return (
    <div className="upload-page">
      <section className="hero">
        <h1>Trustworthy tactical analysis from short match clips</h1>
        <p className="muted">Upload a {lim.min_duration_s}–{lim.max_duration_s} s clip, calibrate a known pitch rectangle, and
          watch the agent check its own tracking and camera geometry — re-running segments when the evidence says the first pass is unreliable.</p>
      </section>
      {DEMO ? (
        <div className="dropzone demo">
          <div><b>Uploads are disabled in this demo.</b></div>
          <div className="small muted">Open the recorded run below, or clone the repo to analyse your own clips.</div>
        </div>
      ) : <label className={`dropzone ${over ? "over" : ""}`} onDragOver={(e) => { e.preventDefault(); setOver(true); }}
        onDragLeave={() => setOver(false)} onDrop={(e) => { e.preventDefault(); setOver(false); const f = e.dataTransfer.files[0]; if (f) send(f); }}>
        <input type="file" accept={lim.allowed_ext.join(",")} onChange={(e) => e.target.files?.[0] && send(e.target.files[0])} hidden />
        {prog === null ? <>
          <div className="dz-icon">⬆</div>
          <div><b>Drop a clip</b> or click to choose</div>
          <div className="small muted">{lim.allowed_ext.join(" ")} · up to {lim.max_upload_mb} MB · {lim.min_duration_s}–{lim.max_duration_s} s · ≥640×360</div>
        </> : <>
          <div>{prog < 1 ? `Uploading… ${(prog * 100).toFixed(0)}%` : "Validating video (decoding all frames)…"}</div>
          <div className="bar"><div style={{ width: `${prog * 100}%` }} /></div>
        </>}
      </label>}
      {err && <div className="alert error">{err}</div>}
      {jobs.length > 0 && (
        <section className="card">
          <h3>Recent clips</h3>
          <table className="jobs">
            <tbody>
              {jobs.map((j) => (
                <tr key={j.id} onClick={() => onOpen(j)}>
                  <td className="mono">{j.filename}</td><td>{j.meta.duration_s.toFixed(1)} s</td>
                  <td><span className={`status ${j.status}`}>{j.status.replace("_", " ")}</span></td>
                  <td className="muted small">{j.created_at.replace("T", " ").slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}
    </div>
  );
}

function Analyze({ job, trace, onJob, onRecalibrate }: { job: Job; trace: AgentStep[]; onJob: (j: Job) => void; onRecalibrate: () => void }) {
  const [stride, setStride] = useState(6);
  const [useColors, setUseColors] = useState(false);
  const [a, setA] = useState("#ffffff");
  const [b, setB] = useState("#1e3a8a");
  const [err, setErr] = useState<string | null>(null);
  const running = RUNNING.has(job.status);
  const start = async () => {
    setErr(null);
    try { onJob(await api.analyze(job.id, { initial_stride: stride, team_colors: useColors ? [a, b] : null })); } catch (e: any) { setErr(e.message); }
  };
  const c = job.calibration;
  return (
    <div className="analyze">
      <div className="card">
        <h3>Analysis</h3>
        {c?.valid ? <p className="small muted">Calibration on frame {c.frame_index} ({c.label ?? "custom"}). {c.warnings.length} warning(s).
          <button className="link" onClick={onRecalibrate}>Edit calibration</button></p>
          : <div className="alert warn">No valid calibration yet. <button className="link" onClick={onRecalibrate}>Calibrate first</button></div>}
        <div className="form-grid">
          <label>Initial sampling
            <select value={stride} onChange={(e) => setStride(+e.target.value)} disabled={running}>
              {[6, 3, 2, 1].map((s) => <option key={s} value={s}>every {s} frame{s > 1 ? "s" : ""} ({(job.meta.fps / s).toFixed(1)} fps)</option>)}
            </select></label>
          <label className="chk"><input type="checkbox" checked={useColors} onChange={(e) => setUseColors(e.target.checked)} disabled={running} /> Set kit colours manually</label>
        </div>
        {useColors && <div className="row"><label>Team A <input type="color" value={a} onChange={(e) => setA(e.target.value)} /></label>
          <label>Team B <input type="color" value={b} onChange={(e) => setB(e.target.value)} /></label></div>}
        <button className="primary big" disabled={DEMO || running || !c?.valid} onClick={start}>{running ? "Running…" : job.run_no ? "Run analysis again" : "Run analysis"}</button>
        {err && <div className="alert error">{err}</div>}
        {(running || job.progress > 0) && (
          <div className="progress">
            <div className="bar"><div style={{ width: `${job.progress * 100}%` }} /></div>
            <div className="small muted">{job.stage}</div>
          </div>
        )}
        <div className="explain small muted">
          <p><b>How the agent works.</b> The clip is split into ~4 s segments. For each segment the agent analyses, measures
            quality signals (camera drift, pitch-line alignment, off-pitch projections, track fragmentation, jumps, gaps), and
            picks a recovery by rule: <i>re-estimate camera motion &amp; update calibration</i> when the camera moved and the
            motion model validates, or <i>re-run with denser sampling</i> when tracking breaks up. Each correction is compared on
            identical frames and kept only if it improves; at most two automatic attempts, then it asks you.</p>
        </div>
      </div>
      <div className="card timeline-card"><h3>Agent activity</h3><Timeline steps={trace} live={running} /></div>
    </div>
  );
}
