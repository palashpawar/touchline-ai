#!/usr/bin/env bash
# Test the deployed endpoint for real: health, full upload->results smoke test, and S3 artifacts.
#   deploy/aws/verify.sh [STACK]
set -euo pipefail
cd "$(dirname "$0")/../.."
STACK=${1:-touchline}
# shellcheck disable=SC1090
source "deploy/aws/.state-${STACK}.env"
export AWS_REGION AWS_PAGER=""
TOKEN=$(aws ssm get-parameter --name "$PARAM" --with-decryption --query Parameter.Value --output text)
BASE="http://$PUBLIC_IP"
echo "==> waiting for $BASE/api/health"
for i in $(seq 1 60); do curl -sf "$BASE/api/health" && break; sleep 10; done; echo
echo "==> unauthenticated request must be rejected"
code=$(curl -s -o /dev/null -w '%{http_code}' "$BASE/api/jobs"); [ "$code" = 401 ] && echo "401 ok" || { echo "expected 401, got $code"; exit 1; }
OUT=eval/out/aws_smoke scripts/smoke_e2e.sh "$BASE" data/clips/france_germany_2018_phase1_ccbysa.ogv "$TOKEN"
JOB=$(cat eval/out/aws_smoke/job_id)
echo "==> artifacts written by the EC2 worker to S3"
aws s3 ls "s3://$BUCKET/${PREFIX}jobs/$JOB/" --recursive | tee eval/out/aws_smoke/s3_listing.txt
python3 -c "import json;r=json.load(open('eval/out/aws_smoke/report.json'));print('processed on',r['provenance']['platform'],r['provenance']['machine'],'opencv',r['provenance']['opencv'],'in',r['provenance']['processing_seconds'],'s')"
