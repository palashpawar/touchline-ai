# Demo video script (target 4:30, hard limit 5:00)

Setup before recording: `scripts/fetch_clips.sh`, app running (local or EC2), browser at 1440×900, a fresh job list.
Use the France–Germany CC BY-SA clip; show the attribution line in the closing card.

| # | Time | Shot | Narration (spoken) |
|---|---|---|---|
| 1 | 0:00–0:20 | Title card → upload page | "College coaches film from the stands with a phone. Touchline AI turns a short clip into tactical measurements — and, more importantly, checks whether those measurements can be trusted." |
| 2 | 0:20–0:40 | Drag `france_germany_2018_phase1_ccbysa.ogv` into the drop zone; show validation (17 s, 1080p) | "Uploads are validated by actually decoding the clip with OpenCV 5: duration 10–30 s, size and resolution limits." |
| 3 | 0:40–1:30 | Calibrate tab: choose **4 landmarks** (goal-area and penalty-area near corners), click P1–P4, tick *Snap to detected line corners*, add the near goal post as a check point, **Validate** | "We don't assume the whole pitch is visible. I map four points I can see, in a fixed order. The tool rejects collinear, crossed or mirrored points — watch: swapping two points gives an ordering error. When it's valid, the magenta template shows the full pitch projected back; the held-out goal post is off by about 2.4 m, and it warns that error grows far from the calibrated area." *(briefly show the swapped-point error, then fix)* |
| 4 | 1:30–2:10 | Analyze tab: **Run analysis**; the agent timeline fills live | "Analysis runs on AWS: the EC2 worker reads the clip from S3, runs YOLOX through OpenCV's DNN module on tiles, filters photographers off the grass, tracks players with a Kalman filter, and projects them onto the pitch. Every ~4 seconds it stops and measures its own quality." |
| 5 | 2:10–3:00 | Expand seg2 PLAN_RECOVERY → estimate_camera_motion → update_calibration → COMPARE | "Here's the agent loop. In segment 2 the phone has drifted 135 pixels from the calibration frame, and painted-line alignment dropped to 0.19. The ORB + MAGSAC motion model is validated — 93% inliers, 2 px error — so it chooses *re-estimate camera motion and update calibration*, reruns the segment, and compares on the same frames: line alignment 0.19 → 0.81, off-pitch projections down. Accepted. If the motion model hadn't validated, it would have asked me instead." |
| 6 | 3:00–3:40 | Results: before/after comparison image; toggle Corrected/Initial video; top-down view + heatmap | "Before and after on the identical frame: the projected lines slide off the pitch without correction and lock on with it. The top-down view and occupancy heatmap use the corrected geometry. Team width and depth are labelled as partial-view measurements of visible players only — and possession, passing, speed are explicitly *not* measured." |
| 7 | 3:40–4:15 | Show eval/RESULTS.md table, then the `textureless_pan` trace: REQUEST_HUMAN_REVIEW; in UI, Human review panel buttons | "When recovery can't be validated — this synthetic stress clip destroys the texture the motion model needs — the agent keeps the previous result, marks the segment uncertain and requests review. I can approve its proposed rerun, correct the field points, or accept it flagged. It never retries more than twice." |
| 8 | 4:15–4:30 | Export buttons → open trace.jsonl | "Everything is exportable: the JSON report and a step-by-step agent trace with signals, tool calls, configs, and before/after quality." |
| 9 | 4:30–4:40 | Closing card: GitHub URL, "OpenCV 5.0.0 · YOLOX-S (Apache-2.0) · clip: Like tears in rain, CC BY-SA 4.0" | — |

## Shot list / capture checklist

1. Upload page empty state, drop-zone hover state.
2. Calibration: point placement with colour-coded P1–P4 and mini-pitch diagram; ordering error; valid report with overlay.
3. Live timeline during the run (record the full run, cut to ~30 s).
4. Expanded agent step showing tool arguments and quality before/after JSON.
5. Comparison figure `compare_seg2.jpg` full-screen.
6. Video player toggling Corrected / Initial at ~7 s (strongest drift).
7. Top-down view with trails, then heatmap: all players.
8. Tactical measurements cards with caveats; "Not measured" card.
9. Terminal: `deploy/aws/verify.sh` output (health, 401 check, S3 listing) — only if the AWS deployment was actually run.
10. `eval/RESULTS.md` trigger-accuracy table; Human review panel.
