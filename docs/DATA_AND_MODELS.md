# Data and model provenance

## Clips (not committed; fetch with `scripts/fetch_clips.sh`)

| File | Source | License | Author | Use |
|---|---|---|---|---|
| `france_germany_2018_phase1_ccbysa.ogv` (17.4 s, 1920×1080, 30 fps; 508 decodable frames) | [Wikimedia Commons: *Match de football France-Allemagne - 16 octobre 2018 - Phase de jeu (1).ogv*](https://commons.wikimedia.org/wiki/File:Match_de_football_France-Allemagne_-_16_octobre_2018_-_Phase_de_jeu_(1).ogv) | CC BY-SA 4.0 | Like tears in rain | Main demo + evaluation (handheld phone, natural camera drift) |
| `latvia_gibraltar_2026-03-31_cc0.webm` (16.9 s) | [Wikimedia Commons: *Latvia-Gibraltar football 2026-03-31.webm*](https://commons.wikimedia.org/wiki/File:Latvia-Gibraltar_football_2026-03-31.webm) | CC0 | Mārtiņš Bruņenieks | Natural failure case: almost no pitch visible (smoke/crowd) |

SHA-256 (as downloaded 2026-10-22):
`82cc79e72f8dc6884d693f29bd44671eff9a6c8c6d219ec3723d18e9135ba94a` (France–Germany),
`3612e26baf7d57a83e1ab7aa176b8ef7551fa7fe991d821e15434e9f0b5c3fe3` (Latvia–Gibraltar).

Annotated videos and comparison frames derived from the France–Germany clip are adaptations and are shared under
**CC BY-SA 4.0** with attribution to the author above. No other match footage was downloaded.

Synthetic stress clips (`eval/out/synthetic/*`) are generated from frame 0 of the France–Germany clip by
`eval/run_eval.py` (known homography warps, noise, frame decimation) and are labelled **synthetic** everywhere.

## Manual labels (`eval/labels/`)

* `fg_landmarks.json` — calibration landmarks on frame 0, two held-out goal-post points, and the same pitch landmarks
  re-annotated on frames 150/300/450.
* `fg_identity_seg1.json` — identities of 8 people over frames 0–90.

Both were produced **by the AI assistant that built this project**, by visual inspection of zoomed, gridded crops. They
were not verified by a human annotator; treat them as a small, honest sanity check, not a benchmark. They were created
before and independently of the agent's acceptance heuristic.

## Detector

| | |
|---|---|
| Model | YOLOX-S, `object_detection_yolox_2022nov.onnx`, COCO-80 (we use class 0 *person*) |
| Source | OpenCV Zoo mirror on Hugging Face `opencv/object_detection_yolox`, revision `78c368f74ce73ee28fc7a1be418a598c71b58b52` |
| SHA-256 | `c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063` (checked at load and at Docker build) |
| License | Apache-2.0 (`models/YOLOX_LICENSE`) |
| Runtime | `cv2.dnn.readNet` + `Net.forward` (OpenCV 5.0.0 DNN engine), OpenCV pre/post-processing and `cv2.dnn.NMSBoxes` |

The learned weights are an external pretrained network (Megvii YOLOX, packaged by OpenCV Zoo); they are not an OpenCV
contribution of this project. The detector has no notion of "player" — spectators, photographers and stewards are removed
by the documented filters (`backend/touchline/vision/detector.py::filter_players`).
