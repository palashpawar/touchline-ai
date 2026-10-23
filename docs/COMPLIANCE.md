# Competition compliance checklist

Status key: **Verified** = run and observed in this environment · **Implemented, unverified** = code/scripts exist but were not
executed end-to-end here · **Incomplete** = not done.

Environments used for verification: Apple M3 (arm64), macOS, Python 3.12, Node 23.11, Docker Desktop 29.3 (linux/arm64 containers); AWS EC2 `m7i-flex.large` (x86_64, Amazon Linux 2023) in us-east-2 with S3.

| Requirement | Status | Evidence |
|---|---|---|
| OpenCV 5 for substantive video analysis | **Verified** | Runtime `cv2.__version__ == "5.0.0"` (official `opencv-python-headless==5.0.0.93`, OpenCV release 5.0.0 tagged 2026-06-06). Docker build asserts `5.x`; every report records it (`provenance.opencv`). Operations listed in TECHNICAL_REPORT §3. No OpenCV 4 fallback is installed anywhere. |
| Meaningful processing component on AWS (EC2 worker processing video, S3 artifacts) | **Verified** | `deploy/aws/verify.sh` against the deployed instance (2026-10-23, account 459374152024, us-east-2; `docs/demo/aws/`): upload → EC2 worker (Amazon Linux 2023, x86_64, OpenCV 5.0.0) analysed the clip from S3 in 586.9 s; all 4 segments recovered via camera-motion update; 10 artifacts written to S3 |
| S3 storage behind the same interface as local | **Verified (real S3, emulator, moto unit test)** | `storage.py`; `tests/test_quality_and_storage.py::test_s3_storage_encrypts_and_uses_prefix`; full containerized run in S3 mode against a moto S3 emulator wrote 10 objects under `touchline/jobs/<id>/`, all `AES256` (`docs/demo/s3_listing_emulator.txt`) |
| EC2 + Docker Compose | **Verified** | `m7i-flex.large` (free-plan account) built and started the api + worker compose stack via user data in ~85 s after boot; also verified locally on linux/arm64 |
| Least-privilege IAM scoped to bucket/prefix | **Verified** | Inline role policy read back from IAM: objects under `touchline-459374152024-us-east-2/touchline/*`, prefix-conditioned ListBucket, one SSM parameter; plus AWS-managed `AmazonSSMManagedInstanceCore` |
| Instance roles, no embedded keys | **Verified** | Instance profile `touchline-ec2-role`, IMDSv2 `required`; the worker wrote S3 with no AWS keys in `.env` |
| S3 encryption + lifecycle cleanup; raw clips private | **Verified** | Read back: default SSE `AES256`, per-object `AES256`, all four public-access blocks on, anonymous GET of the uploaded clip → 403, lifecycle `touchline/jobs/` expires after 7 days |
| Upload size and duration limits | **Verified** | Streaming 200 MB limit (413), 10–30 s by decoding, ≥640×360, extension whitelist; `tests/test_upload_api.py` |
| Bounded worker concurrency | **Verified (local)** | `TOUCHLINE_WORKER_CONCURRENCY` threads claim tasks atomically; compose CPU limit |
| No unauthenticated public processing endpoints | **Verified** | Deployed `/api/jobs` without token → 401; security group allows only port 80 from one /32 (`allow-my-ip.sh` swaps it); script refuses `0.0.0.0/0`. Plain HTTP: use TLS (ALB+ACM) beyond a demo |
| Structured logs, health checks | **Verified** | JSON-lines logging; `/api/health`; Docker `HEALTHCHECK` |
| Cleanup script | **Implemented, not yet run** | `deploy/aws/destroy.sh` — run after recording the demo |
| Agentic Vision: perception → decision → action, OpenCV outputs change the next tool call | **Verified** | Real trace `docs/demo/trace.jsonl` (containerized run); 8/9 eval cases chose the expected action; `test_decision_depends_on_signals_not_fixed_sequence`; eval cases choose different tools from different evidence (eval/RESULTS.md) |
| Not a chatbot over fixed results | **Verified** | No LLM in control; state machine re-runs vision with changed configuration |
| Explicit states INGEST…COMPLETE | **Verified** | `agent/trace.py::STATES`, trace |
| Structured tools (analyze_segment … request_human_review) | **Verified** | `agent/tools.py::TOOL_SPECS`, `/api/config` |
| ≥2 real recovery strategies | Camera-motion update **verified on natural footage**; densify sampling **verified on synthetic sampling stress** (accepted in 2 cases: short-track ratio 0.41→0.26 and 0.47→0.36); not triggered on the natural clip | eval/RESULTS.md |
| Unvalidated correction → keep previous, mark uncertain, request review | **Verified** | `test_rejected_correction_preserves_previous_result`; eval failure cases |
| Retry bound (2) / no loops | **Verified** | `test_retry_limit_is_two_and_escalates`; eval max attempts 2 across 9 cases |
| Step log fields (timestamp, ids, signals, decision, reason, tool+args, configs, before/after, outcome, accepted) | **Verified** | `test_trace_roundtrip_and_required_fields`; trace |
| Quality signals defined with units, limitations, thresholds; not probabilities | **Verified** | `docs/QUALITY_SIGNALS.md`, UI labels |
| Interactive 4-point calibration; validation of degeneracy, order, plausibility, visibility | **Verified** | UI checked in browser; `tests/test_calibration.py` |
| No metric distances with invalid calibration | **Verified** | API refuses analysis; `assemble` withholds metrics; UI empty states |
| Camera moves → update calibration or mark unreliable | **Verified** | natural run (4/4 segments updated); scene-cut case |
| Pretrained detector pinned, licensed, documented; not claimed as OpenCV contribution | **Verified** | `docs/DATA_AND_MODELS.md`, SHA-256 check |
| Bottom-centre ground point; spectator filtering documented | **Verified** | `detector.py::filter_players` |
| Tracking with unmatched detections and occlusion handling; identity-switch limitation documented | **Verified** | `tracker.py`; identity eval |
| Trails, top-down locations, width/depth with visibility caveats, defined heatmap | **Verified** | UI + report |
| Manual team colours / track correction; uncertainty exposed | **Verified (unit + UI)** | Team panel, `team_update` review action; confidence labelled heuristic. Verified through the API in Docker: kit colours + override `3:other` applied |
| No possession/passing/identity/ball claims | **Verified** | `unavailable` block in report and UI |
| Frontend features (upload/status, overlays, calibration, pitch view, metrics + caveats, timeline, before/after, review, downloads) | **Verified** | Browser inspection during development |
| Pinned dependencies, build instructions | **Verified** | `backend/requirements*.txt`, `frontend/package-lock.json`, Dockerfile with pinned base images |
| Architecture + agent Mermaid diagrams | **Verified** | `docs/ARCHITECTURE.md` |
| Evaluation evidence with reproducible commands | **Verified** | `eval/run_eval.py`, `eval/RESULTS.md` |
| Failure cases | **Verified** | TECHNICAL_REPORT §8 |
| Demo script < 5 min | **Verified (written)** | `docs/DEMO_SCRIPT.md`; video not recorded |
| Demonstration video recorded | **Incomplete** | Needs human recording |
| Optional COOL path | Not pursued (by instruction) | — |
| Submission (deadline 2026-10-26 23:59 PT) | **Incomplete** | Public repo github.com/palashpawar/touchline-ai; demo video and submission form still to do |
