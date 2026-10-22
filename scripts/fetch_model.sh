#!/usr/bin/env bash
# Fetch the pinned YOLOX-S ONNX (OpenCV Zoo, Apache-2.0) and verify its SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
REV=78c368f74ce73ee28fc7a1be418a598c71b58b52
SHA=c5c2d13e59ae883e6af3b45daea64af4833a4951c92d116ec270d9ddbe998063
mkdir -p models
f=models/object_detection_yolox_2022nov.onnx
[ -f "$f" ] || curl -fsSL -o "$f" "https://huggingface.co/opencv/object_detection_yolox/resolve/$REV/object_detection_yolox_2022nov.onnx"
[ -f models/YOLOX_LICENSE ] || curl -fsSL -o models/YOLOX_LICENSE "https://huggingface.co/opencv/object_detection_yolox/resolve/$REV/LICENSE"
echo "$SHA  $f" | shasum -a 256 -c -
