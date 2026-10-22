# Demo run (real, containerized)

Produced by `scripts/smoke_e2e.sh` against `docker compose -f docker-compose.yml -f deploy/docker-compose.s3test.yml` on 2026-10-22:
API and worker containers (linux/arm64, Docker Desktop on Apple M3), **S3 storage mode against a local S3 emulator (moto)**, not real AWS.

* Clip: France–Germany 2018, CC BY-SA 4.0 (Like tears in rain / Wikimedia Commons) — 16.9 s, 1920×1080
* Runtime: OpenCV 5.0.0 on Linux-7.0.14-linuxkit-aarch64-with-glibc2.36; processing 278.6 s (agent + rendering)
* Agent steps: 39; segments: seg1 recovered, seg2 recovered, seg3 recovered, seg4 recovered

| File | What |
|---|---|
| `trace.jsonl` | Machine-readable agent trace (one JSON object per step) |
| `report.json` | Full exported report (calibration, signal definitions, per-segment quality, tracks, metrics, provenance) |
| `compare_seg2.jpg`, `compare_seg4.jpg` | Same frame before/after the camera-motion calibration update (CC BY-SA 4.0 derivative) |
| `s3_listing_emulator.txt` | Objects written by the worker to the emulated bucket, with SSE status |

## Decisions in this run

| Segment | Initial breaches | Action | Line alignment before → after | Accepted |
|---|---|---|---|---|
| seg1 | camera_drift_px, line_alignment | camera_motion_update | 0.46 → 0.90 | yes |
| seg2 | camera_drift_px, line_alignment | camera_motion_update | 0.20 → 0.83 | yes |
| seg3 | camera_drift_px, line_alignment | camera_motion_update | 0.20 → 0.81 | yes |
| seg4 | camera_drift_px, line_alignment | camera_motion_update | 0.21 → 0.83 | yes |
