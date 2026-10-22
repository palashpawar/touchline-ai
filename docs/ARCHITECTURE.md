# Architecture

## System

```mermaid
flowchart LR
  subgraph Browser
    UI["React + TypeScript (Vite)<br/>upload · calibration · player · pitch view · agent timeline · review"]
  end
  subgraph EC2["EC2 (Docker Compose) — or local"]
    API["FastAPI api<br/>validation · auth token · artifacts"]
    DB[("SQLite<br/>jobs · tasks · steps · reviews")]
    W["worker (N=1 threads)<br/>rule-based agent + OpenCV 5 pipeline"]
    CACHE[("/data volume<br/>local artifact cache")]
  end
  S3[("S3 bucket (private, SSE-S3)<br/>touchline/jobs/&lt;id&gt;/…<br/>lifecycle: 7 days")]
  SSM["SSM Parameter Store<br/>API token (SecureString)"]
  UI -- "HTTPS/HTTP + X-Touchline-Token" --> API
  API -- enqueue / status --> DB
  W -- claim task / log steps --> DB
  API -- Storage interface --> S3
  W -- "Storage interface (instance role)" --> S3
  W --- CACHE
  API --- CACHE
  SSM -. "read at boot" .-> EC2
```

The same `Storage` protocol (`backend/touchline/storage.py`) has a local-filesystem implementation for development and an
S3 implementation for AWS (instance-role credentials, SSE on every object, prefix-scoped keys). Processing always works on
local files: inputs are fetched into a cache, outputs are uploaded when finished.

## Vision pipeline (per sampled frame)

```mermaid
flowchart TB
  A["cv2.VideoCapture decode<br/>(sequential grab/read, stride s)"] --> B["Field mask<br/>HSV inRange + morphology + connectedComponents"]
  B --> C["YOLOX-S via cv2.dnn<br/>640 px tiles over grass bbox + NMSBoxes"]
  C --> D["Player filter<br/>foot on grass · aspect · size vs median"]
  A --> M["Camera motion to reference frame<br/>ORB + BFMatcher ratio test + findHomography(USAC_MAGSAC)<br/>fallback: goodFeaturesToTrack + calcOpticalFlowPyrLK chain"]
  D --> P["Ground point = box bottom-centre<br/>(warped into reference frame if compensating)"]
  M --> P
  P --> H["Image→pitch: perspectiveTransform with H_ref · M_t<br/>off-pitch (&gt;3 m) rejected"]
  H --> T["Tracker: cv2.KalmanFilter (CV model)<br/>+ Hungarian on position/box-height + HSV hist (compareHist)"]
  A --> LA["Line alignment<br/>top-hat line mask · warpPerspective · distanceTransform"]
  M --> LA
  T --> Q["Quality signals"]
  LA --> Q
  M --> Q
```

## Agent workflow

```mermaid
stateDiagram-v2
  [*] --> INGEST
  INGEST --> CALIBRATE
  CALIBRATE --> REQUEST_HUMAN_REVIEW: calibration invalid
  CALIBRATE --> ANALYZE: valid (per segment)
  ANALYZE --> ASSESS_QUALITY
  ASSESS_QUALITY --> PLAN_RECOVERY
  PLAN_RECOVERY --> COMPLETE: no breaches → accept
  PLAN_RECOVERY --> RERUN_SEGMENT: camera drift / line misalignment<br/>AND motion validated<br/>→ estimate_camera_motion → update_calibration
  PLAN_RECOVERY --> RERUN_SEGMENT: fragmentation / jumps / gaps<br/>AND stride > 1 → denser stride
  PLAN_RECOVERY --> REQUEST_HUMAN_REVIEW: motion not validated · misaligned after compensation<br/>· full frame rate reached · attempts ≥ 2
  RERUN_SEGMENT --> REQUEST_HUMAN_REVIEW: update_calibration rejected
  RERUN_SEGMENT --> COMPARE
  COMPARE --> ASSESS_QUALITY: accepted (new result kept)
  COMPARE --> REQUEST_HUMAN_REVIEW: rejected (previous result preserved)
  REQUEST_HUMAN_REVIEW --> COMPLETE: segment marked uncertain
  COMPLETE --> [*]
```

Human actions after `REQUEST_HUMAN_REVIEW` (UI "Human review" panel): **approve proposed rerun** (runs the agent's proposal
beyond the automatic budget, still subject to `compare_runs`), **correct field points** (new calibration → new run), or
**accept as-is** (segment included, flagged `reviewer_accepted`). Team colours/track overrides re-assemble the outputs
without re-running vision. Human steps are logged in the same trace with `actor: "human"`.

## Tools

| Tool | Input | Output |
|---|---|---|
| `analyze_segment` | segment id, frame range, `SegmentConfig` (stride, motion_compensation, motion_model) | result id, per-frame detections/motion/alignment, tracks |
| `assess_tracking_quality` | result id, optional frame set | signals + breaches |
| `estimate_camera_motion` | segment, stride | per-frame frame→reference homographies + validation stats |
| `update_calibration` | motion id | accepted?, probe-frame line alignment static vs updated, per-frame model |
| `rerun_segment` | segment, new config, attempt | new result id |
| `compare_runs` | before id, after id, target signals | accepted?, per-signal deltas on identical frames |
| `request_human_review` | segment, reason, kind, proposed config | review id |
