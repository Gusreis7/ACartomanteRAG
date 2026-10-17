from __future__ import annotations

from pathlib import Path
from typing import Protocol

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError


class RustFSConfigProtocol(Protocol):
    endpoint: str
    access_key: str
    secret_key: str


class RustFSManager:
    def __init__(
        self,
        endpoint: str | None = None,
        access_key: str | None = None,
        secret_key: str | None = None,
        region: str = "us-east-1",
    ) -> None:
        self.region = region
        kwargs = {}
        if endpoint:
            kwargs = {
                "endpoint_url": endpoint,
                "aws_access_key_id": access_key,
                "aws_secret_access_key": secret_key,
            }
        self.client = boto3.client(
            "s3",
            **kwargs,
            region_name=region,
            config=Config(
                signature_version="s3v4",
                s3={
                    "addressing_style": "path" if endpoint else "auto",
                },
            ),
        )

    @classmethod
    def from_config(
        cls,
        config: RustFSConfigProtocol,
        region: str = "us-east-1",
    ) -> "RustFSManager":
        """
        Cria o manager a partir da configuração RustFS já resolvida
        pelo ConfigManager.

        Exemplo:
            config = ConfigManager.load()
            storage = RustFSManager.from_config(
                config.environment.rustfs
            )
        """
        return cls(
            endpoint=config.endpoint,
            access_key=config.access_key,
            secret_key=config.secret_key,
            region=getattr(config, "region", region),
        )

    def create_bucket(
        self,
        bucket: str,
    ) -> None:
        if self.bucket_exists(bucket):
            return

        kwargs = {"Bucket": bucket}
        if self.region != "us-east-1":
            kwargs["CreateBucketConfiguration"] = {"LocationConstraint": self.region}
        self.client.create_bucket(**kwargs)

    def bucket_exists(
        self,
        bucket: str,
    ) -> bool:
        try:
            self.client.head_bucket(
                Bucket=bucket,
            )
            return True

        except ClientError as exc:
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            error_code = exc.response.get("Error", {}).get("Code")

            if status == 404 or error_code in {
                "404",
                "NoSuchBucket",
                "NotFound",
            }:
                return False

            raise

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
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")

            error_code = exc.response.get("Error", {}).get("Code")

            if status == 404 or error_code in {
                "404",
                "NoSuchKey",
                "NotFound",
            }:
                return False

            raise

    def upload_file(
        self,
        bucket: str,
        key: str,
        local_path: str | Path,
    ) -> None:
        local_path = Path(local_path)

        if not local_path.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {local_path}")

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

        if content_type is not None:
            kwargs["ContentType"] = content_type

        self.client.put_object(
            **kwargs,
        )

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

        keys: list[str] = []

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
