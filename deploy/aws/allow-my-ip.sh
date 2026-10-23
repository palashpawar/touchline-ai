#!/usr/bin/env bash
# Replace the security group's allowed source with your current public IP (e.g. after a network change).
#   deploy/aws/allow-my-ip.sh [STACK]
set -euo pipefail
cd "$(dirname "$0")/../.."
STACK=${1:-touchline}
# shellcheck disable=SC1090
source "deploy/aws/.state-${STACK}.env"
export AWS_REGION AWS_PAGER=""
NEW="$(curl -s https://checkip.amazonaws.com)/32"
for old in $(aws ec2 describe-security-groups --group-ids "$SG" --query 'SecurityGroups[0].IpPermissions[?FromPort==`80`].IpRanges[].CidrIp' --output text); do
  [ "$old" = "$NEW" ] && { echo "already allowed: $NEW"; exit 0; }
  aws ec2 revoke-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr "$old" >/dev/null && echo "revoked $old"
done
aws ec2 authorize-security-group-ingress --group-id "$SG" --protocol tcp --port 80 --cidr "$NEW" >/dev/null
echo "allowed $NEW on port 80 of $SG"
