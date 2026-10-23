# Touchline AI

Turns a short soccer clip into tactical analysis for coaches, and tells you when not to trust it.

An agent checks its own video analysis every few seconds. If something looks wrong (the camera moved, or tracking broke
up), it changes its approach, reruns that part, and keeps the fix only if it measurably helps. If it can't fix it, it
marks that part uncertain and asks you.

Built with **OpenCV 5.0.0**, FastAPI, React, and AWS (EC2 + S3). The agent is rule-based; no LLM or paid API is involved.

## Results

* On a real handheld clip, the agent detected camera drift and corrected it in all 4 segments. After the camera moved,
  position error on held-out landmarks dropped from **7.9 m to 0.6 m**.
* It chose the expected action in **8 of 9** test cases, including asking for help when recovery failed.
* Verified running on AWS: the EC2 worker processed the clip from S3 in 587 s.

Details: [evaluation results](eval/RESULTS.md) · [technical report](docs/TECHNICAL_REPORT.md)

## Run it locally

Needs Python 3.12 and Node 18+.

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r backend/requirements-dev.txt && scripts/fetch_model.sh && scripts/fetch_clips.sh && npm --prefix frontend ci && npm --prefix frontend run build
```
```bash
.venv/bin/python -m uvicorn touchline.api:app --app-dir backend --port 8000
```

Open http://localhost:8000, then:

1. **Upload** a 10–30 s clip.
2. **Calibrate:** click four points of a known pitch area (e.g. the penalty box). For the demo clip, use the points
   in `eval/labels/fg_landmarks.json`.
3. **Analyze** and watch the agent's steps live.
4. **Review** the annotated video, top-down pitch view, team shape, before/after comparisons, and any review requests.
5. **Export** the report, agent trace, and videos.

Or with Docker:

```bash
docker compose up --build
```

## Test

```bash
cd backend && ../.venv/bin/python -m pytest -q
```
```bash
.venv/bin/python eval/run_eval.py
```

## Deploy to AWS

```bash
deploy/aws/deploy.sh && deploy/aws/verify.sh
```

Setup, account notes, and teardown: [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## Docs

[Architecture & diagrams](docs/ARCHITECTURE.md) · [Quality signals](docs/QUALITY_SIGNALS.md) ·
[Competition checklist](docs/COMPLIANCE.md) · [Data & model sources](docs/DATA_AND_MODELS.md) ·
[Environment variables](docs/ENVIRONMENT.md) · [Demo script](docs/DEMO_SCRIPT.md) · [Demo run](docs/demo/)

## Limitations

Player IDs can switch (no re-identification). Teams are guessed from shirt colour. Accuracy drops far from the
calibrated area, and team width/depth only cover visible players. It does not track the ball, possession, passing, or speed.

## License

Code: MIT. Detector (YOLOX-S, OpenCV Zoo): Apache-2.0. Demo clip: CC BY-SA 4.0 (Like tears in rain, Wikimedia Commons).
