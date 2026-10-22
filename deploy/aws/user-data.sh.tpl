#!/bin/bash
# EC2 bootstrap (Amazon Linux 2023). Credentials come from the instance role via IMDSv2.
set -euxo pipefail
exec > >(tee /var/log/touchline-bootstrap.log) 2>&1
dnf install -y docker tar gzip
systemctl enable --now docker
COMPOSE_VERSION=v2.29.7
COMPOSE_ARCH=__ARCH__; [ "$COMPOSE_ARCH" = arm64 ] && COMPOSE_ARCH=aarch64
mkdir -p /usr/local/lib/docker/cli-plugins
curl -fsSL -o /usr/local/lib/docker/cli-plugins/docker-compose \
  "https://github.com/docker/compose/releases/download/${COMPOSE_VERSION}/docker-compose-linux-${COMPOSE_ARCH}"
chmod +x /usr/local/lib/docker/cli-plugins/docker-compose
mkdir -p /opt/touchline && cd /opt/touchline
aws s3 cp "s3://__BUCKET__/__PREFIX__deploy/bundle.tgz" bundle.tgz --region __REGION__
tar -xzf bundle.tgz
TOKEN=$(aws ssm get-parameter --region __REGION__ --name "__PARAM__" --with-decryption --query Parameter.Value --output text)
NCPU=$(nproc)
cat > .env <<EOF
TOUCHLINE_STORAGE=s3
TOUCHLINE_S3_BUCKET=__BUCKET__
TOUCHLINE_S3_PREFIX=__PREFIX__
AWS_REGION=__REGION__
TOUCHLINE_API_TOKEN=${TOKEN}
TOUCHLINE_WORKER_CONCURRENCY=1
TOUCHLINE_MAX_UPLOAD_MB=200
TOUCHLINE_BIND=0.0.0.0
TOUCHLINE_PORT=80
TOUCHLINE_WORKER_CPUS=$(( NCPU > 1 ? NCPU - 1 : 1 ))
EOF
chmod 600 .env
docker compose build
docker compose up -d
