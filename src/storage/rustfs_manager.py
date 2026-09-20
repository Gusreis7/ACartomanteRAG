import os
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class RustFSManager:
    def __init__(
        self,
        endpoint: str,
        access_key: str,
        secret_key: str,
        region: str = "us-east-1",
    ):
        self.client = boto3.client(
            "s3",
            endpoint_url=endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": "path",
                },
            ),
        )

    @classmethod
    def from_env(cls) -> "RustFSManager":
        return cls(
            endpoint=os.environ["RUSTFS_ENDPOINT"],
            access_key=os.environ["RUSTFS_ACCESS_KEY"],
            secret_key=os.environ["RUSTFS_SECRET_KEY"],
        )

    def create_bucket(self, bucket: str) -> None:
        if self.bucket_exists(bucket):
            return

        self.client.create_bucket(
            Bucket=bucket,
        )

    def bucket_exists(self, bucket: str) -> bool:
        try:
            self.client.head_bucket(
                Bucket=bucket,
            )
            return True

        except ClientError:
            return False

    def object_exists(
        self,
        bucket: str,
        key: str,
    ) -> bool:
        try:
            self.client.head_object(
                Bucket=bucket,
                Key=key,
            )
            return True

        except ClientError as exc:
            status = exc.response.get(
                "ResponseMetadata",
                {},
            ).get("HTTPStatusCode")

            if status == 404:
                return False

            raise

    def upload_file(
        self,
        bucket: str,
        key: str,
        local_path: str | Path,
    ) -> None:
        self.client.upload_file(
            Filename=str(local_path),
            Bucket=bucket,
            Key=key,
        )

    def upload_bytes(
        self,
        bucket: str,
        key: str,
        data: bytes,
        content_type: str | None = None,
    ) -> None:
        kwargs = {
            "Bucket": bucket,
            "Key": key,
            "Body": data,
        }

        if content_type:
            kwargs["ContentType"] = content_type

        self.client.put_object(**kwargs)

    def download_file(
        self,
        bucket: str,
        key: str,
        local_path: str | Path,
    ) -> None:
        local_path = Path(local_path)

        local_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client.download_file(
            Bucket=bucket,
            Key=key,
            Filename=str(local_path),
        )

    def download_bytes(
        self,
        bucket: str,
        key: str,
    ) -> bytes:
        response = self.client.get_object(
            Bucket=bucket,
            Key=key,
        )

        return response["Body"].read()

    def list_objects(
        self,
        bucket: str,
        prefix: str = "",
    ) -> list[str]:
        paginator = self.client.get_paginator("list_objects_v2")

        keys = []

        for page in paginator.paginate(
            Bucket=bucket,
            Prefix=prefix,
        ):
            for item in page.get(
                "Contents",
                [],
            ):
                keys.append(item["Key"])

        return keys

    def delete_object(
        self,
        bucket: str,
        key: str,
    ) -> None:
        self.client.delete_object(
            Bucket=bucket,
            Key=key,
        )
