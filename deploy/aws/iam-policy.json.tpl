{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "ObjectsUnderAppPrefixOnly",
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:AbortMultipartUpload"],
      "Resource": "arn:aws:s3:::__BUCKET__/__PREFIX__*"
    },
    {
      "Sid": "ListAppPrefixOnly",
      "Effect": "Allow",
      "Action": "s3:ListBucket",
      "Resource": "arn:aws:s3:::__BUCKET__",
      "Condition": {"StringLike": {"s3:prefix": ["__PREFIX__*"]}}
    },
    {
      "Sid": "ReadApiToken",
      "Effect": "Allow",
      "Action": "ssm:GetParameter",
      "Resource": "arn:aws:ssm:__REGION__:__ACCOUNT__:parameter__PARAM__"
    }
  ]
}
