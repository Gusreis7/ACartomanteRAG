"""Cria o backend antes do Terraform. Reexecução preserva o conteúdo do bucket."""
import argparse
from pathlib import Path
import json

import boto3
from botocore.exceptions import ClientError


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bucket", required=True)
    p.add_argument("--region", default="us-east-1")
    p.add_argument("--output", type=Path, default=Path("infra/aws/backend.hcl"))
    args = p.parse_args()
    sts = boto3.client("sts", region_name=args.region)
    account = sts.get_caller_identity()["Account"]
    s3 = boto3.client("s3", region_name=args.region)
    try:
        s3.head_bucket(Bucket=args.bucket, ExpectedBucketOwner=account)
    except ClientError as exc:
        if str(exc.response["Error"]["Code"]) not in {"404", "NoSuchBucket", "NotFound"}:
            raise
        kwargs = {"Bucket": args.bucket}
        if args.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": args.region}
        s3.create_bucket(**kwargs)
    s3.put_public_access_block(Bucket=args.bucket, PublicAccessBlockConfiguration={
        "BlockPublicAcls": True, "IgnorePublicAcls": True, "BlockPublicPolicy": True, "RestrictPublicBuckets": True,
    })
    s3.put_bucket_versioning(Bucket=args.bucket, VersioningConfiguration={"Status": "Enabled"})
    s3.put_bucket_encryption(Bucket=args.bucket, ServerSideEncryptionConfiguration={
        "Rules": [{"ApplyServerSideEncryptionByDefault": {"SSEAlgorithm": "AES256"}}],
    })
    s3.put_bucket_policy(Bucket=args.bucket, Policy=json.dumps({"Version": "2012-10-17", "Statement": [{
        "Effect": "Deny", "Principal": "*", "Action": "s3:*", "Resource": [f"arn:aws:s3:::{args.bucket}", f"arn:aws:s3:::{args.bucket}/*"],
        "Condition": {"Bool": {"aws:SecureTransport": "false"}},
    }]}))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(f'bucket = "{args.bucket}"\nkey = "acartomanterag/production.tfstate"\nregion = "{args.region}"\nencrypt = true\nuse_lockfile = true\n')
    print(f"Backend preparado: {args.output}")


if __name__ == "__main__":
    main()
