#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from src.config.config_manager import ConfigManager
from src.embeddings.registry import EmbedderRegistry
from src.indexing.enrichment_embedding_index import (
    ENRICHMENT_INDEX_CURRENT_KEY,
    EnrichmentEmbeddingIndex,
    make_qdrant_manager,
    release_memory,
    utc_now,
)
from src.storage.rustfs_manager import RustFSManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cria collections Qdrant para o enrichment atual, uma por embedder."
        )
    )
    parser.add_argument(
        "--embedders",
        nargs="+",
        default=None,
        help="Aliases em embedding.models. Padrão: todos os modelos configurados.",
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs: dict[str, Any] = {}
    if args.env_file is not None:
        kwargs["env_file"] = args.env_file
    if args.config_file is not None:
        kwargs["yaml_file"] = args.config_file
    config = ConfigManager.load(**kwargs)

    aliases = args.embedders or list(config.embedding.models.keys())
    aliases = [str(alias) for alias in aliases]
    unknown = sorted(set(aliases) - set(config.embedding.models))
    if unknown:
        raise SystemExit(f"Embedders não configurados: {', '.join(unknown)}")

    rustfs = RustFSManager.from_config(config.environment.rustfs)
    qdrant = make_qdrant_manager(config)
    registry = EmbedderRegistry()
    indexer = EnrichmentEmbeddingIndex(config, rustfs, qdrant, registry)

    source_current, _, source_manifest_sha = indexer.resolve_current_dataset()
    indexes: dict[str, Any] = {}
    for alias in aliases:
        print(f"\n=== ENRICHMENT EMBEDDER: {alias} ===")
        try:
            manifest = indexer.build(model_alias=alias)
            indexes[alias] = {
                "index_id": manifest["index_id"],
                "collection": manifest["qdrant"]["collection"],
                "manifest_key": f"enrichment/{manifest['index_id']}/manifest.json",
                "model": manifest["embedding"]["model"],
                "resolved_revision": manifest["embedding"]["resolved_revision"],
                "dimension": manifest["embedding"]["dimension"],
                "query_prompt_name": manifest["embedding"].get("query_prompt_name"),
            }
        finally:
            release_memory()

    indexes_bucket = config.storage.embedding_indexes_bucket
    current = {
        "manifest_version": 1,
        "status": "success",
        "created_at_utc": utc_now(),
        "source_enrichment_dataset_id": source_current["enrichment_dataset_id"],
        "source_enrichment_manifest_sha256": source_manifest_sha,
        "indexes": indexes,
    }
    rustfs.upload_bytes(
        indexes_bucket,
        ENRICHMENT_INDEX_CURRENT_KEY,
        json.dumps(current, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )

    print("\n=== ENRICHMENT INDEXES OK ===")
    print(f"source: {current['source_enrichment_dataset_id']}")
    for alias, item in indexes.items():
        print(f"{alias}: {item['collection']}")
    print(
        "current: s3://"
        f"{indexes_bucket}/{ENRICHMENT_INDEX_CURRENT_KEY}"
    )


if __name__ == "__main__":
    main()
