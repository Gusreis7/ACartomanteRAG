from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from src.config.config_manager import AppConfig
from src.storage.rustfs_manager import RustFSManager


@dataclass(frozen=True, slots=True)
class ReleasePointer:
    release: str
    index_id: str
    published_at_utc: str


class ReleaseManager:
    RELEASE_VERSION = 1

    def __init__(self, config: AppConfig, rustfs: RustFSManager) -> None:
        self.config = config
        self.rustfs = rustfs

    def load(self, release: str) -> ReleasePointer:
        release = self._validate_name(release)
        bucket = self.config.storage.embedding_indexes_bucket
        key = f"releases/{release}.json"
        if not self.rustfs.object_exists(bucket=bucket, key=key):
            raise FileNotFoundError(f"Release não publicada: s3://{bucket}/{key}")
        payload = json.loads(
            self.rustfs.download_bytes(bucket=bucket, key=key).decode("utf-8")
        )
        if payload.get("release_version") != self.RELEASE_VERSION:
            raise ValueError("Versão de release não suportada.")
        return ReleasePointer(
            release, str(payload["index_id"]), str(payload["published_at_utc"])
        )

    def publish(self, index_id: str, release: str) -> ReleasePointer:
        release = self._validate_name(release)
        index_id = str(index_id).strip()
        manifest = self.load_index_manifest(index_id)
        if manifest.get("status") != "success":
            raise ValueError("Somente índices status=success podem ser promovidos.")
        if str(manifest.get("index_id")) != index_id:
            raise ValueError("index_id do manifest não corresponde ao solicitado.")
        if not str(manifest.get("qdrant", {}).get("collection", "")).strip():
            raise ValueError("Manifest não possui qdrant.collection.")
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "release_version": 1,
            "release": release,
            "index_id": index_id,
            "published_at_utc": now,
        }
        bucket = self.config.storage.embedding_indexes_bucket
        self.rustfs.upload_bytes(
            bucket=bucket,
            key=f"releases/{release}.json",
            data=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            content_type="application/json",
        )
        return ReleasePointer(release, index_id, now)

    def load_index_manifest(self, index_id: str) -> dict[str, Any]:
        bucket = self.config.storage.embedding_indexes_bucket
        key = f"{index_id}/manifest.json"
        if not self.rustfs.object_exists(bucket=bucket, key=key):
            raise FileNotFoundError(f"Manifest não encontrado: s3://{bucket}/{key}")
        return json.loads(
            self.rustfs.download_bytes(bucket=bucket, key=key).decode("utf-8")
        )

    def exists(
        self,
        release: str = "default",
    ) -> bool:
        release = self._validate_name(release)

        bucket = self.config.storage.embedding_indexes_bucket

        key = f"releases/{release}.json"

        return self.rustfs.object_exists(
            bucket=bucket,
            key=key,
        )

    def bootstrap_if_missing(
        self,
        index_id: str,
        release: str = "default",
    ) -> bool:
        """Publica somente se a release ainda não existir."""
        release = self._validate_name(release)

        if self.exists(release):
            return False

        self.publish(
            index_id=index_id,
            release=release,
        )

        return True

    @staticmethod
    def _validate_name(name: str) -> str:
        name = str(name).strip()
        allowed = set(
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_"
        )
        if not name or any(char not in allowed for char in name):
            raise ValueError("Nome de release inválido.")
        return name
