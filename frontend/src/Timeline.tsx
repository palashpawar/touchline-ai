import { useState } from "react";
import { AgentStep } from "./api";

const STATE_CLASS: Record<string, string> = {
  INGEST: "s-neutral", CALIBRATE: "s-neutral", ANALYZE: "s-info", ASSESS_QUALITY: "s-info",
  PLAN_RECOVERY: "s-plan", RERUN_SEGMENT: "s-action", COMPARE: "s-compare",
  REQUEST_HUMAN_REVIEW: "s-review", COMPLETE: "s-done",
};

export function Timeline({ steps, live }: { steps: AgentStep[]; live: boolean }) {
  const [open, setOpen] = useState<number | null>(null);
  const [seg, setSeg] = useState<string>("all");
  if (!steps.length)
    return <div className="empty">{live ? "Waiting for the agent's first step…" : "No agent activity yet. Run an analysis to see each perception → decision → action step."}</div>;
  const segs = Array.from(new Set(steps.map((s) => s.segment_id).filter(Boolean))) as string[];
  const shown = steps.filter((s) => seg === "all" || s.segment_id === seg || s.segment_id === null);
  return (
    <div className="timeline">
      <div className="seg-ctl small">
        <button className={seg === "all" ? "on" : ""} onClick={() => setSeg("all")}>All</button>
        {segs.map((s) => <button key={s} className={seg === s ? "on" : ""} onClick={() => setSeg(s)}>{s}</button>)}
      </div>
      <ol>
        {shown.map((s) => (
          <li key={s.step} className={`tl-item ${open === s.step ? "open" : ""}`}>
            <button className="tl-head" onClick={() => setOpen(open === s.step ? null : s.step)}>
              <span className={`state ${STATE_CLASS[s.state] ?? ""}`}>{s.state.replace("_", " ")}</span>
              {s.segment_id && <span className="segtag">{s.segment_id}</span>}
              <span className="decision">{s.decision}</span>
              {s.actor === "human" && <span className="tag human">human</span>}
              {s.accepted === true && <span className="ok-dot" title="accepted">✓</span>}
              {s.accepted === false && <span className="bad-dot" title="not accepted">✕</span>}
              <span className="ts mono">{s.timestamp.slice(11, 19)}</span>
            </button>
            <div className="tl-reason">{s.reason}</div>
            {open === s.step && (
              <div className="tl-detail">
                {s.tool && <p><b>Tool</b> <code>{s.tool}</code></p>}
                {Object.keys(s.tool_args ?? {}).length > 0 && <Json label="Arguments" v={s.tool_args} />}
                {Object.keys(s.observed_signals ?? {}).length > 0 && <Json label="Observed signals" v={s.observed_signals} />}
                {s.previous_config && <Json label="Previous config" v={s.previous_config} />}
                {s.proposed_config && <Json label="Proposed config" v={s.proposed_config} />}
                {s.quality_before && <Json label="Quality before" v={s.quality_before} />}
                {s.quality_after && <Json label="Quality after" v={s.quality_after} />}
                <p><b>Outcome</b> {s.outcome} · attempt {s.attempt}</p>
              </div>
            )}
          </li>
        ))}
      </ol>
      {live && <div className="live-dot">agent running</div>}
    </div>
  );
}

function Json({ label, v }: { label: string; v: unknown }) {
  return (
    <details open>
      <summary>{label}</summary>
      <pre>{JSON.stringify(v, (_k, x) => (typeof x === "number" ? Math.round(x * 1000) / 1000 : x), 2)}</pre>
    </details>
  );
}
