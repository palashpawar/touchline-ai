# Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `TOUCHLINE_STORAGE` | `local` | `local` (filesystem under `TOUCHLINE_DATA_DIR/objects`) or `s3` |
| `TOUCHLINE_DATA_DIR` | `<repo>/var` (`/data` in Docker) | SQLite DB, local objects, S3 cache |
| `TOUCHLINE_S3_BUCKET` | — | Bucket name (required for `s3`) |
| `TOUCHLINE_S3_PREFIX` | `touchline/` | Every key is written under this prefix; IAM is scoped to it |
| `TOUCHLINE_S3_ENDPOINT` | — | Only for local S3-emulator tests; never set on AWS |
| `AWS_REGION` | `us-east-1` | Region for S3/SSM clients |
| `TOUCHLINE_MODEL_PATH` | `<repo>/models/object_detection_yolox_2022nov.onnx` | Detector weights (SHA-256 checked at load) |
| `TOUCHLINE_MAX_UPLOAD_MB` | `200` | Upload size limit (enforced while streaming) |
| `TOUCHLINE_MIN_DURATION_S` / `TOUCHLINE_MAX_DURATION_S` | `10` / `30` | Clip duration limits (±0.5 s tolerance), checked by decoding |
| `TOUCHLINE_WORKER_CONCURRENCY` | `1` | Max clips processed at once (worker threads) |
| `TOUCHLINE_WORKER_MODE` | `inline` | `inline`: API process also runs the worker. `external`: separate `python -m touchline.worker` (Docker) |
| `TOUCHLINE_API_TOKEN` | — | When set, every `/api/*` request (except health) needs header `X-Touchline-Token` or the login cookie |
| `TOUCHLINE_FRONTEND_DIST` | `<repo>/frontend/dist` | Built UI served at `/` when present |
| `TOUCHLINE_BIND` / `TOUCHLINE_PORT` | `127.0.0.1` / `8000` | docker-compose host binding (EC2 uses `0.0.0.0:80`, restricted by security group) |
| `TOUCHLINE_WORKER_CPUS` | `3.5` | docker-compose CPU limit for the worker |

AWS credentials are **never** configured through environment variables on EC2: boto3 uses the instance role via IMDSv2
(hop limit 2 so containers can reach it).
