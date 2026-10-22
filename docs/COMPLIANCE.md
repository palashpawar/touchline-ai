# Competition compliance checklist

Status key: **Verified** = run and observed in this environment · **Implemented, unverified** = code/scripts exist but were not
executed end-to-end here · **Incomplete** = not done.

Environment used for verification: Apple M3 (arm64), macOS, Python 3.12, Node 23.11, Docker Desktop 29.3 (linux/arm64 containers).

| Requirement | Status | Evidence |
|---|---|---|
| OpenCV 5 for substantive video analysis | **Verified** | Runtime `cv2.__version__ == "5.0.0"` (official `opencv-python-headless==5.0.0.93`, OpenCV release 5.0.0 tagged 2026-06-06). Docker build asserts `5.x`; every report records it (`provenance.opencv`). Operations listed in TECHNICAL_REPORT §3. No OpenCV 4 fallback is installed anywhere. |
| Meaningful processing component on AWS (EC2 worker processing video, S3 artifacts) | **Implemented, unverified** | `deploy/aws/deploy.sh`, `user-data.sh.tpl`, `verify.sh`, `destroy.sh`. Not executed: no AWS credentials/CLI in this environment. The S3 code path was exercised against an S3 emulator (see below), not real S3. |
| S3 storage behind the same interface as local | **Verified (emulator + moto unit test)** | `storage.py`; `tests/test_quality_and_storage.py::test_s3_storage_encrypts_and_uses_prefix`; full containerized run in S3 mode against a moto S3 emulator wrote 10 objects under `touchline/jobs/<id>/`, all `AES256` (`docs/demo/s3_listing_emulator.txt`) |
| EC2 + Docker Compose | Compose **verified locally**; EC2 **unverified** | `docker compose` api + worker containers (linux/arm64): upload → agent → report in 281 s; 401 without token; team re-assignment task completed (`docs/demo/`) |
| Least-privilege IAM scoped to bucket/prefix | **Implemented, unverified** | `deploy/aws/iam-policy.json.tpl` (objects under `touchline/*` only, ListBucket prefix-conditioned, one SSM parameter) + AWS-managed `AmazonSSMManagedInstanceCore` for Session Manager |
| Instance roles, no embedded keys | **Implemented, unverified** | Instance profile + IMDSv2 (hop limit 2); `.env` contains no AWS keys |
| S3 encryption + lifecycle cleanup; raw clips private | **Implemented, unverified** | SSE-S3 default + per-object `AES256`, public-access block, TLS-only bucket policy, 7-day expiry on `touchline/jobs/` |
| Upload size and duration limits | **Verified** | Streaming 200 MB limit (413), 10–30 s by decoding, ≥640×360, extension whitelist; `tests/test_upload_api.py` |
| Bounded worker concurrency | **Verified (local)** | `TOUCHLINE_WORKER_CONCURRENCY` threads claim tasks atomically; compose CPU limit |
| No unauthenticated public processing endpoints | **Verified (local)** / AWS unverified | Token middleware (401 test); SG restricted to `ALLOWED_CIDR`, script refuses `0.0.0.0/0` |
| Structured logs, health checks | **Verified** | JSON-lines logging; `/api/health`; Docker `HEALTHCHECK` |
| Cleanup script | **Implemented, unverified** | `deploy/aws/destroy.sh` |
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
| Submission (deadline 2026-10-26 23:59 PT) | **Incomplete** | Repository not pushed/submitted; AWS deployment must be run and verified with real credentials |
