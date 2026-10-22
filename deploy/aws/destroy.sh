#!/usr/bin/env bash
# Remove every resource created by deploy.sh for STACK. Deleting the bucket requires DELETE_BUCKET=1.
#   deploy/aws/destroy.sh [STACK]      DELETE_BUCKET=1 deploy/aws/destroy.sh [STACK]
set -euo pipefail
cd "$(dirname "$0")/../.."
STACK=${1:-touchline}
STATE=deploy/aws/.state-${STACK}.env
[ -f "$STATE" ] || { echo "no state file $STATE"; exit 1; }
# shellcheck disable=SC1090
source "$STATE"
export AWS_REGION AWS_PAGER=""
echo "==> terminating $INSTANCE_ID"
aws ec2 terminate-instances --instance-ids "$INSTANCE_ID" >/dev/null || true
aws ec2 wait instance-terminated --instance-ids "$INSTANCE_ID" || true
echo "==> security group $SG"
aws ec2 delete-security-group --group-id "$SG" || true
echo "==> IAM role / instance profile $ROLE"
aws iam remove-role-from-instance-profile --instance-profile-name "$ROLE" --role-name "$ROLE" || true
aws iam delete-instance-profile --instance-profile-name "$ROLE" || true
aws iam delete-role-policy --role-name "$ROLE" --policy-name "${STACK}-app" || true
aws iam detach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore || true
aws iam delete-role --role-name "$ROLE" || true
echo "==> SSM parameter $PARAM"
aws ssm delete-parameter --name "$PARAM" || true
echo "==> application objects s3://$BUCKET/$PREFIX"
aws s3 rm "s3://$BUCKET/$PREFIX" --recursive || true
if [ "${DELETE_BUCKET:-0}" = 1 ]; then
  echo "==> deleting bucket $BUCKET"
  aws s3 rb "s3://$BUCKET" --force
fi
rm -f "$STATE"
echo "done"
