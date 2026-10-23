# AWS deployment (EC2 + S3)

Requires the AWS CLI v2, configured (`aws sts get-caller-identity` works).

```bash
export AWS_REGION=us-east-2 ALLOWED_CIDR=$(curl -s https://checkip.amazonaws.com)/32 INSTANCE_TYPE=m7i-flex.large
```
```bash
deploy/aws/deploy.sh      # create everything (~2 min, then ~2-8 min for the instance to build the app)
```
```bash
deploy/aws/verify.sh      # health, 401 without token, full upload→report run on EC2, S3 listing
```
```bash
DELETE_BUCKET=1 deploy/aws/destroy.sh   # remove everything
```

Get the access token for the web UI:

```bash
aws ssm get-parameter --name /touchline/api_token --with-decryption --query Parameter.Value --output text
```

## What `deploy.sh` creates

* Private S3 bucket: SSE-S3 encryption, public access blocked, TLS-only policy, `touchline/jobs/*` deleted after 7 days.
* IAM role limited to `<bucket>/touchline/*` plus one SSM parameter, and SSM Session Manager (no SSH).
* API token stored as an SSM SecureString.
* Security group allowing port 80 only from `ALLOWED_CIDR` (the script refuses `0.0.0.0/0`).
* One EC2 instance (Amazon Linux 2023, IMDSv2 required, encrypted disk) that runs the Docker Compose stack. Its worker
  processes the video and reads/writes S3 using the instance role (no stored keys).

## Account notes

* **Region-locked organization accounts:** if EC2 is denied outside one region, set `AWS_REGION` to that region.
* **Free-plan accounts:** only free-tier types launch; use `INSTANCE_TYPE=m7i-flex.large` (2 vCPU, 8 GB).
* **Your IP changed?** Run `deploy/aws/allow-my-ip.sh` to allow your current address instead.
* The endpoint is plain HTTP restricted to one IP. Beyond a demo, put it behind TLS (ALB + ACM).

## Verified run (2026-10-23)

us-east-2, `m7i-flex.large`: the EC2 worker processed the demo clip from S3 in 587 s, recovered all four segments, and
wrote all artifacts back to S3. Evidence: [docs/demo/aws/](demo/aws/) and [COMPLIANCE.md](COMPLIANCE.md).
