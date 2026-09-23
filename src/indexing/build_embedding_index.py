from __future__ import annotations

import argparse
import json
import time

from src.config.config_manager import ConfigManager
from src.embeddings.registry import EmbedderRegistry
from src.indexing.embedding_index import EmbeddingIndex
from src.storage.qdrant_manager import QdrantManager
from src.storage.rustfs_manager import RustFSManager


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um índice vetorial a partir de um dataset Silver "
            "e persiste a collection no Qdrant + manifest no RustFS."
        )
    )

    parser.add_argument(
        "silver_dataset_id",
        help=(
            "ID do dataset Silver. Ex.: "
            "v1_hybrid_roberta_crawl_pt_512_o64_36a9f7af2a"
        ),
    )

    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Alias do modelo em embedding.models. "
            "Se omitido, usa embedding.selected_model do config.yaml."
        ),
    )

    return parser.parse_args()


def build_qdrant(config) -> QdrantManager:
    if hasattr(QdrantManager, "from_config"):
        return QdrantManager.from_config(
            config.environment.qdrant
        )

    return QdrantManager(
        url=config.environment.qdrant.url,
        api_key=config.environment.qdrant.api_key,
    )


def main() -> None:
    args = parse_args()
    config = ConfigManager.load()

    model_alias = (
        args.model
        or config.embedding.selected_model
    )

    if model_alias not in config.embedding.models:
        available = ", ".join(
            sorted(config.embedding.models)
        )
        raise ValueError(
            f"Modelo '{model_alias}' não configurado. "
            f"Disponíveis: {available}."
        )

    model_config = config.embedding.models[
        model_alias
    ]

    rustfs = RustFSManager.from_config(
        config.environment.rustfs
    )
    qdrant = build_qdrant(config)
    registry = EmbedderRegistry()

    print("=== BUILD EMBEDDING INDEX ===")
    print(
        f"silver_dataset_id: {args.silver_dataset_id}"
    )
    print(f"model_alias:       {model_alias}")
    print(f"backend:           {model_config.backend}")
    print(f"model:             {model_config.model}")
    print(
        "normalization:     "
        f"{config.embedding.index.normalize_embeddings}"
    )
    print(
        f"distance:          "
        f"{config.embedding.index.distance}"
    )
    print(
        f"device:            "
        f"{config.embedding.execution.device}"
    )
    print(
        f"batch_size:        "
        f"{config.embedding.execution.batch_size}"
    )
    print()

    started = time.perf_counter()

    try:
        indexer = EmbeddingIndex(
            config=config,
            rustfs=rustfs,
            qdrant=qdrant,
            registry=registry,
        )

        manifest = indexer.build(
            silver_dataset_id=args.silver_dataset_id,
            model_alias=model_alias,
        )

        elapsed = time.perf_counter() - started

        print()
        print("=== INDEX CREATED ===")
        print(
            f"index_id:          "
            f"{manifest['index_id']}"
        )
        print(
            f"collection:        "
            f"{manifest['qdrant']['collection']}"
        )
        print(
            f"silver_chunks:     "
            f"{manifest['validation']['silver_chunks']}"
        )
        print(
            f"qdrant_vectors:    "
            f"{manifest['validation']['qdrant_vectors']}"
        )
        print(
            f"dimension:         "
            f"{manifest['embedding']['dimension']}"
        )
        print(
            f"resolved_revision: "
            f"{manifest['embedding']['resolved_revision']}"
        )
        print(
            f"device:            "
            f"{manifest['execution']['resolved_device']}"
        )
        print(
            f"batch_effective:   "
            f"{manifest['execution']['batch_size_effective']}"
        )
        print(
            f"elapsed_seconds:   "
            f"{elapsed:.2f}"
        )
        print(
            "manifest:          "
            f"s3://{config.storage.embedding_indexes_bucket}/"
            f"{manifest['index_id']}/manifest.json"
        )

        print()
        print("[OK] Índice criado e validado.")

        print()
        print("=== MANIFEST SUMMARY ===")
        summary = {
            "index_id": manifest["index_id"],
            "silver_dataset_id": (
                manifest["source"]["silver_dataset_id"]
            ),
            "model": manifest["embedding"]["model"],
            "resolved_revision": (
                manifest["embedding"]["resolved_revision"]
            ),
            "dimension": (
                manifest["embedding"]["dimension"]
            ),
            "distance": (
                manifest["embedding"]["distance"]
            ),
            "normalize_embeddings": (
                manifest["embedding"][
                    "normalize_embeddings"
                ]
            ),
            "vectors": (
                manifest["validation"]["qdrant_vectors"]
            ),
        }

        print(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
        )

    finally:
        if hasattr(qdrant, "close"):
            qdrant.close()


if __name__ == "__main__":
    main()
