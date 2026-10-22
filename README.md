# Touchline AI

Soccer clip analysis for college coaches, built so you can tell when the numbers are trustworthy. A **rule-based agent**
checks its own OpenCV 5 analysis segment by segment. When it finds a problem, it changes its processing approach (re-estimates camera motion and updates the calibration, or
re-samples frames more densely), reruns the affected segment, and keeps the correction only if it measurably improves
the result on identical frames. If it can't recover, it marks the segment uncertain and asks a human.

* **Vision:** OpenCV **5.0.0** (official `opencv-python-headless==5.0.0.93` wheel). Used for decoding, field segmentation, line
  detection, homography/perspective transforms, ORB + MAGSAC and LK optical-flow camera motion, Kalman tracking, DNN inference,
  overlays, and video encoding.
* **Detector:** YOLOX-S ONNX from OpenCV Zoo (Apache-2.0), run by `cv2.dnn`. See [docs/DATA_AND_MODELS.md](docs/DATA_AND_MODELS.md).
* **Agent:** an explicit state machine (`INGEST → CALIBRATE → ANALYZE → ASSESS_QUALITY → PLAN_RECOVERY → RERUN_SEGMENT →
  COMPARE → REQUEST_HUMAN_REVIEW → COMPLETE`). No LLM is involved in control.
* **Stack:** FastAPI + SQLite + bounded background worker · React/TypeScript/Vite · S3 + EC2/Docker Compose.

Docs: [Technical report](docs/TECHNICAL_REPORT.md) · [Architecture & diagrams](docs/ARCHITECTURE.md) ·
[Quality signals](docs/QUALITY_SIGNALS.md) · [Evaluation results](eval/RESULTS.md) · [Compliance checklist](docs/COMPLIANCE.md) ·
[Demo script](docs/DEMO_SCRIPT.md) · [Environment variables](docs/ENVIRONMENT.md) · [Demo trace](docs/demo/)

## Quick start (local, macOS/Linux)

Requires Python 3.12 and Node 18+ (tested with Node 23.11).

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt
```

```bash
scripts/fetch_model.sh && scripts/fetch_clips.sh
```

```bash
npm --prefix frontend ci && npm --prefix frontend run build
```

```bash
.venv/bin/python -m uvicorn touchline.api:app --app-dir backend --port 8000
```

Open http://localhost:8000. For frontend development with hot reload, run `npm --prefix frontend run dev` (it proxies `/api` to port 8780 by default; set `TOUCHLINE_API=http://localhost:8000` to change that).

Verify the runtime really is OpenCV 5:

```bash
.venv/bin/python -c "import cv2; print(cv2.__version__)"
```

## Using it

1. **Upload** a 10–30 s clip (mp4/mov/webm/mkv/avi/ogv, ≤200 MB, ≥640×360).
2. **Calibrate:** pick a frame, choose a known region (penalty area, goal area, a custom rectangle, or four named
   landmarks), and click P1→P4 in the order shown on the mini-pitch. You can also add held-out check points. Invalid geometry
   (duplicate, collinear, crossed, or mirrored points, a horizon through the field, or an implausible scale) is rejected with a reason.
   For the demo clip, use the landmarks in `eval/labels/fg_landmarks.json` (frame 0).
3. **Analyze:** watch the agent timeline live.
4. **Results:** view the annotated video (corrected vs initial), the top-down view with trails and heatmap, team width/depth with caveats,
   per-segment quality before/after, comparison frames, human-review actions, and team-colour correction.
5. **Export** the JSON report, the JSONL agent trace, and the annotated videos.

A scripted end-to-end check over HTTP:

```bash
scripts/smoke_e2e.sh http://localhost:8000
```

## Tests and evaluation

```bash
cd backend && ../.venv/bin/python -m pytest -q
```

```bash
.venv/bin/python eval/run_eval.py
```

The test suite covers homography ordering and degeneracy, image↔pitch transforms, policy decisions, retry limits,
accept/reject comparisons, trace serialization, upload validation, auth, and S3 storage (moto). The evaluation writes
`eval/out/results.json` and [eval/RESULTS.md](eval/RESULTS.md).

## Docker

```bash
docker compose up --build
```

This serves the UI and API at http://127.0.0.1:8000, with a separate worker container. To test the S3 storage path locally against an emulator:

```bash
docker compose -f docker-compose.yml -f deploy/docker-compose.s3test.yml up -d --build
```

## AWS deployment (EC2 + S3)

`deploy/aws/deploy.sh` (AWS CLI v2) creates:

* a private S3 bucket with SSE-S3, public access blocked, a TLS-only policy, and a lifecycle rule that deletes `touchline/jobs/*` after 7 days;
* an IAM role whose S3 access is limited to `arn:aws:s3:::<bucket>/touchline/*`, plus read access to one SSM parameter and SSM Session Manager (no SSH);
* an API token stored as an SSM SecureString;
* a security group that allows port 80 only from `ALLOWED_CIDR` (the script refuses `0.0.0.0/0`);
* one EC2 instance (Amazon Linux 2023, IMDSv2 required, encrypted gp3) that builds and runs the Docker Compose stack. The worker on that instance processes the video and reads/writes S3.

```bash
export AWS_REGION=us-east-1 ALLOWED_CIDR=$(curl -s https://checkip.amazonaws.com)/32
```

```bash
deploy/aws/deploy.sh
```

```bash
deploy/aws/verify.sh
```

```bash
DELETE_BUCKET=1 deploy/aws/destroy.sh
```

`verify.sh` tests the deployed endpoint for real: health, a 401 check without the token, the full upload→report smoke
test, and an S3 listing of the artifacts the EC2 worker wrote. **Status: the AWS deployment has not been run or verified
from this environment (no AWS credentials or CLI available).** See [docs/COMPLIANCE.md](docs/COMPLIANCE.md).

The endpoint is plain HTTP restricted to your IP. For anything beyond a demo, put it behind TLS (ALB + ACM) or use SSM port
forwarding.

## Limitations (short version)

Identity switches happen (there is no re-identification). Team labels come from a colour heuristic. Metric accuracy degrades
far from the calibrated rectangle. Team width/depth describe visible players only. No ball, possession, passing, or speed
outputs. The details are in the [technical report](docs/TECHNICAL_REPORT.md#limitations).

## Licensing

Code: MIT (see `LICENSE`). Detector weights: Apache-2.0. The demo clip is CC BY-SA 4.0 (Like tears in rain / Wikimedia
Commons), so derived annotated videos are shared under CC BY-SA 4.0.
