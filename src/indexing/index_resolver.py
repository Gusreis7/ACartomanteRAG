from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from src.config.config_manager import AppConfig
from src.indexing.release_manager import ReleaseManager
from src.storage.rustfs_manager import RustFSManager


@dataclass(frozen=True, slots=True)
class ResolvedIndex:
    selection: str
    release: str | None
    index_id: str
    collection: str
    silver_dataset_id: str
    model_alias: str
    model: str
    configured_revision: str
    resolved_revision: str
    dimension: int
    distance: str
    manifest: dict[str, Any]


class IndexResolver:
    def __init__(self, config: AppConfig, rustfs: RustFSManager) -> None:
        self.config = config
        self.releases = ReleaseManager(config, rustfs)

    def resolve(self, index_id: str | None = None) -> ResolvedIndex:
        explicit = index_id or self.config.retrieval.index_id
        if explicit:
            return self._from_index(str(explicit), "explicit", None)
        release = os.getenv("ACARTOMANTE_RELEASE", self.config.retrieval.release)
        pointer = self.releases.load(release)
        return self._from_index(pointer.index_id, "release", pointer.release)

    def _from_index(
        self, index_id: str, selection: str, release: str | None
    ) -> ResolvedIndex:
        manifest = self.releases.load_index_manifest(index_id)
        if manifest.get("status") != "success":
            raise ValueError("Índice selecionado não possui status=success.")
        source = manifest.get("source", {})
        embedding = manifest.get("embedding", {})
        qdrant = manifest.get("qdrant", {})
        values = {
            "collection": qdrant.get("collection"),
            "silver_dataset_id": source.get("silver_dataset_id"),
            "model_alias": embedding.get("model_alias"),
            "model": embedding.get("model"),
            "configured_revision": embedding.get("configured_revision"),
            "resolved_revision": embedding.get("resolved_revision"),
            "dimension": embedding.get("dimension"),
            "distance": embedding.get("distance"),
        }
        missing = [
            k
            for k, v in values.items()
            if v is None or (isinstance(v, str) and not v.strip())
        ]
        if missing:
            raise ValueError("Manifest incompleto: " + ", ".join(missing))
        return ResolvedIndex(
            selection,
            release,
            index_id,
            str(values["collection"]),
            str(values["silver_dataset_id"]),
            str(values["model_alias"]),
            str(values["model"]),
            str(values["configured_revision"]),
            str(values["resolved_revision"]),
            int(values["dimension"]),
            str(values["distance"]),
            manifest,
        )
