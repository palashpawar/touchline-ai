export type Job = {
  id: string;
  created_at: string;
  status: "uploaded" | "calibrated" | "queued" | "running" | "needs_review" | "complete" | "failed";
  stage: string;
  progress: number;
  filename: string;
  meta: { fps: number; width: number; height: number; frame_count: number; duration_s: number; codec: string };
  calibration: CalibrationReport | null;
  error: string | null;
  run_no: number;
  active_tasks: number;
};

export type CalibrationReport = {
  valid: boolean;
  errors: string[];
  warnings: string[];
  homography: number[][] | null;
  metrics: Record<string, any>;
  frame_index: number;
  image_points: number[][];
  pitch_points: number[][];
  pitch_size: [number, number];
  label?: string;
};

export type AgentStep = {
  step: number;
  timestamp: string;
  run_id: string;
  segment_id: string | null;
  state: string;
  decision: string;
  reason: string;
  tool: string | null;
  tool_args: Record<string, any>;
  observed_signals: Record<string, any>;
  previous_config: Record<string, any> | null;
  proposed_config: Record<string, any> | null;
  quality_before: { signals: Record<string, any>; breaches: string[] } | null;
  quality_after: { signals: Record<string, any>; breaches: string[] } | null;
  outcome: string;
  accepted: boolean | null;
  attempt: number;
  actor: string;
};

export type Review = {
  id: string;
  segment_id: string | null;
  kind: string;
  reason: string;
  proposed_config: Record<string, any> | null;
  status: string;
  resolution: Record<string, any> | null;
  run_no: number;
};

export type SignalSpec = {
  key: string;
  label: string;
  unit: string;
  threshold: number;
  bad_when: "above" | "below";
  definition: string;
  limitations: string;
};

export type AppConfig = {
  limits: { max_upload_mb: number; min_duration_s: number; max_duration_s: number; allowed_ext: string[] };
  presets: Record<string, { width_m: number; height_m: number; label: string }>;
  landmarks: Record<string, [number, number]>;
  signals: SignalSpec[];
};

/** Static, read-only build for GitHub Pages: serves one recorded run from /demo/ instead of the API. */
export const DEMO = import.meta.env.VITE_DEMO === "1";
const BASE = import.meta.env.BASE_URL;
const demoFile = (name: string) => `${BASE}demo/${name}`;
const DEMO_MSG = "This is a read-only demo of a recorded run. Run the app locally or deploy it to upload and analyse clips.";

async function demoJson<T>(name: string): Promise<T> {
  const r = await fetch(demoFile(name));
  if (!r.ok) throw new ApiError(r.status, `Demo data missing: ${name}`);
  return r.json() as Promise<T>;
}

let token = "";
try {
  token = sessionStorage.getItem("touchline_token") ?? "";
} catch {
  /* storage may be unavailable */
}
export const setToken = (t: string) => {
  token = t;
  try {
    sessionStorage.setItem("touchline_token", t);
  } catch {
    /* storage may be unavailable */
  }
};

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = { ...(init.headers as Record<string, string>) };
  if (token) headers["X-Touchline-Token"] = token;
  const r = await fetch(path, { ...init, headers, credentials: "same-origin" });
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail);
    } catch {
      /* not json */
    }
    throw new ApiError(r.status, msg);
  }
  return r.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

const liveApi = {
  health: () => req<any>("/api/health"),
  config: () => req<AppConfig>("/api/config"),
  login: (t: string) => req<any>("/api/auth", json({ token: t })),
  jobs: () => req<Job[]>("/api/jobs"),
  job: (id: string) => req<Job>(`/api/jobs/${id}`),
  upload: (file: File, onProgress?: (p: number) => void) =>
    new Promise<Job>((resolve, reject) => {
      const x = new XMLHttpRequest();
      x.open("POST", "/api/jobs");
      if (token) x.setRequestHeader("X-Touchline-Token", token);
      x.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
      x.onload = () => {
        let body: any = {};
        try {
          body = JSON.parse(x.responseText);
        } catch {
          /* ignore */
        }
        if (x.status >= 200 && x.status < 300) resolve(body as Job);
        else reject(new ApiError(x.status, body.detail ?? `Upload failed (${x.status})`));
      };
      x.onerror = () => reject(new ApiError(0, "Network error during upload"));
      const fd = new FormData();
      fd.append("file", file);
      x.send(fd);
    }),
  lines: (id: string, frame: number) => req<{ segments: number[][] }>(`/api/jobs/${id}/lines/${frame}`),
  calibrate: (id: string, body: unknown) => req<CalibrationReport>(`/api/jobs/${id}/calibration`, json(body)),
  analyze: (id: string, body: unknown) => req<Job>(`/api/jobs/${id}/analyze`, json(body)),
  trace: (id: string) => req<AgentStep[]>(`/api/jobs/${id}/trace`),
  reviews: (id: string) => req<Review[]>(`/api/jobs/${id}/reviews`),
  review: (id: string, rid: string, body: unknown) => req<Job>(`/api/jobs/${id}/reviews/${rid}`, json(body)),
  report: (id: string) => req<any>(`/api/jobs/${id}/report`),
};

const readOnly = () => Promise.reject(new ApiError(403, DEMO_MSG));
const demoApi: typeof liveApi = {
  health: () => demoJson("health.json"),
  config: () => demoJson<AppConfig>("config.json"),
  login: () => Promise.resolve({ ok: true }),
  jobs: async () => [await demoJson<Job>("job.json")],
  job: () => demoJson<Job>("job.json"),
  upload: () => readOnly(),
  lines: () => Promise.resolve({ segments: [] }),
  calibrate: () => readOnly(),
  analyze: () => readOnly(),
  trace: () => demoJson<AgentStep[]>("trace.json"),
  reviews: () => Promise.resolve([]),
  review: () => readOnly(),
  report: () => demoJson("report.json"),
};

export const api = DEMO ? demoApi : liveApi;

export const frameUrl = (id: string, i: number) => (DEMO ? demoFile("frame0.jpg") : `/api/jobs/${id}/frame/${i}`);
export const artifactUrl = (id: string, name: string, run: number, download = false) =>
  DEMO ? demoFile(name) : `/api/jobs/${id}/artifacts/${name}?run=${run}${download ? "&download=true" : ""}`;
