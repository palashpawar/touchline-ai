#!/usr/bin/env bash
# Reproducible AWS deployment of Touchline AI with the AWS CLI v2.
#
#   Creates: private SSE-encrypted S3 bucket (+ lifecycle cleanup), least-privilege IAM role and
#   instance profile, SecureString API token in SSM Parameter Store, a security group open only to
#   ALLOWED_CIDR on port 80, and one EC2 instance (Amazon Linux 2023, IMDSv2, encrypted EBS) that
#   runs docker compose. The worker on EC2 processes video and reads/writes artifacts in S3.
#
# Usage:
#   export AWS_REGION=us-east-1 ALLOWED_CIDR=$(curl -s https://checkip.amazonaws.com)/32
#   deploy/aws/deploy.sh
# Optional: STACK=touchline INSTANCE_TYPE=c7i.xlarge BUCKET=<name> RETENTION_DAYS=7
# Free-plan accounts only allow free-tier types: use INSTANCE_TYPE=m7i-flex.large (2 vCPU, 8 GB).
set -euo pipefail
cd "$(dirname "$0")/../.."

: "${AWS_REGION:?set AWS_REGION}"
: "${ALLOWED_CIDR:?set ALLOWED_CIDR to your IP/32 (processing endpoints must not be public)}"
STACK=${STACK:-touchline}
INSTANCE_TYPE=${INSTANCE_TYPE:-c7i.xlarge}
RETENTION_DAYS=${RETENTION_DAYS:-7}
ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
BUCKET=${BUCKET:-${STACK}-${ACCOUNT}-${AWS_REGION}}
PREFIX=touchline/
ROLE=${STACK}-ec2-role
PARAM=/${STACK}/api_token
STATE=deploy/aws/.state-${STACK}.env
export AWS_PAGER=""
log() { printf '\033[1;32m==>\033[0m %s\n' "$*"; }

[[ "$ALLOWED_CIDR" == "0.0.0.0/0" ]] && { echo "Refusing to expose processing endpoints to 0.0.0.0/0"; exit 1; }
[ -f models/object_detection_yolox_2022nov.onnx ] || scripts/fetch_model.sh

log "S3 bucket s3://$BUCKET (private, SSE-S3, lifecycle ${RETENTION_DAYS}d on ${PREFIX}jobs/)"
if ! aws s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  if [ "$AWS_REGION" = us-east-1 ]; then aws s3api create-bucket --bucket "$BUCKET" >/dev/null
  else aws s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$AWS_REGION" >/dev/null; fi
fi
aws s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
aws s3api put-bucket-encryption --bucket "$BUCKET" --server-side-encryption-configuration \
  '{"Rules":[{"ApplyServerSideEncryptionByDefault":{"SSEAlgorithm":"AES256"},"BucketKeyEnabled":true}]}'
aws s3api put-bucket-ownership-controls --bucket "$BUCKET" --ownership-controls 'Rules=[{ObjectOwnership=BucketOwnerEnforced}]'
aws s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration "{
  \"Rules\":[{\"ID\":\"expire-job-artifacts\",\"Status\":\"Enabled\",\"Filter\":{\"Prefix\":\"${PREFIX}jobs/\"},
              \"Expiration\":{\"Days\":${RETENTION_DAYS}},\"AbortIncompleteMultipartUpload\":{\"DaysAfterInitiation\":1}}]}"
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{
  \"Version\":\"2012-10-17\",\"Statement\":[{\"Sid\":\"DenyInsecureTransport\",\"Effect\":\"Deny\",\"Principal\":\"*\",
  \"Action\":\"s3:*\",\"Resource\":[\"arn:aws:s3:::${BUCKET}\",\"arn:aws:s3:::${BUCKET}/*\"],
  \"Condition\":{\"Bool\":{\"aws:SecureTransport\":\"false\"}}}]}"

log "API token in SSM Parameter Store ($PARAM, SecureString)"
if ! aws ssm get-parameter --name "$PARAM" >/dev/null 2>&1; then
  TOKEN=$(openssl rand -hex 24)
  aws ssm put-parameter --name "$PARAM" --type SecureString --value "$TOKEN" >/dev/null
fi

log "IAM role $ROLE (S3 limited to $BUCKET/$PREFIX*, SSM param read, SSM Session Manager)"
sed -e "s|__BUCKET__|$BUCKET|g" -e "s|__PREFIX__|$PREFIX|g" -e "s|__REGION__|$AWS_REGION|g" \
    -e "s|__ACCOUNT__|$ACCOUNT|g" -e "s|__PARAM__|$PARAM|g" deploy/aws/iam-policy.json.tpl > /tmp/${STACK}-policy.json
if ! aws iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  aws iam create-role --role-name "$ROLE" --assume-role-policy-document \
    '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"ec2.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  aws iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore
  aws iam create-instance-profile --instance-profile-name "$ROLE" >/dev/null
  aws iam add-role-to-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE"
  sleep 10  # IAM propagation
fi
aws iam put-role-policy --role-name "$ROLE" --policy-name "${STACK}-app" --policy-document file:///tmp/${STACK}-policy.json

log "Application bundle -> s3://$BUCKET/${PREFIX}deploy/"
BUNDLE=/tmp/${STACK}-bundle.tgz
tar --exclude=.venv --exclude=var --exclude=eval/out --exclude=data --exclude=node_modules \
    --exclude=frontend/dist --exclude=frontend/dist-demo --exclude=frontend/demo-public --exclude=.git --exclude='*.log' -czf "$BUNDLE" \
    Dockerfile docker-compose.yml .dockerignore backend frontend models scripts
aws s3 cp "$BUNDLE" "s3://$BUCKET/${PREFIX}deploy/bundle.tgz" --sse AES256 >/dev/null

log "Security group (port 80 from $ALLOWED_CIDR only; no SSH — use SSM Session Manager)"
VPC=$(aws ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
SG=$(aws ec2 describe-security-groups --filters Name=group-name,Values=${STACK}-sg Name=vpc-id,Values=$VPC \
     --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ "$SG" = None ] || [ -z "$SG" ]; then
  SG=$(aws ec2 create-security-group --group-name ${STACK}-sg --description "Touchline AI" --vpc-id "$VPC" --query GroupId --output text)
  aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr "$ALLOWED_CIDR" >/dev/null
fi

log "EC2 instance ($INSTANCE_TYPE, AL2023)"
ARCH=x86_64; [[ "$INSTANCE_TYPE" =~ ^[a-z]+[0-9]+g ]] && ARCH=arm64
AMI=$(aws ssm get-parameter --name /aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-$ARCH --query Parameter.Value --output text)
sed -e "s|__BUCKET__|$BUCKET|g" -e "s|__PREFIX__|$PREFIX|g" -e "s|__REGION__|$AWS_REGION|g" -e "s|__PARAM__|$PARAM|g" \
    -e "s|__ARCH__|$ARCH|g" deploy/aws/user-data.sh.tpl > /tmp/${STACK}-user-data.sh
IID=$(aws ec2 run-instances --image-id "$AMI" --instance-type "$INSTANCE_TYPE" \
  --iam-instance-profile Name="$ROLE" --security-group-ids "$SG" \
  --metadata-options HttpTokens=required,HttpEndpoint=enabled,HttpPutResponseHopLimit=2 \
  --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":30,"VolumeType":"gp3","Encrypted":true,"DeleteOnTermination":true}}]' \
  --user-data file:///tmp/${STACK}-user-data.sh \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${STACK}},{Key=app,Value=touchline}]" \
  --query 'Instances[0].InstanceId' --output text)
aws ec2 wait instance-running --instance-ids "$IID"
IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
cat > "$STATE" <<EOS
STACK=$STACK
AWS_REGION=$AWS_REGION
BUCKET=$BUCKET
PREFIX=$PREFIX
ROLE=$ROLE
PARAM=$PARAM
SG=$SG
INSTANCE_ID=$IID
PUBLIC_IP=$IP
EOS
log "Instance $IID at http://$IP (bootstrapping takes ~5-8 min: docker build on the instance)"
echo "Token:   aws ssm get-parameter --name $PARAM --with-decryption --query Parameter.Value --output text"
echo "Verify:  deploy/aws/verify.sh $STACK"
echo "Destroy: deploy/aws/destroy.sh $STACK"
