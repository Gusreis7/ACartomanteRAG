from __future__ import annotations

import argparse
import json
import textwrap
from typing import Any

from qdrant_client import QdrantClient

from src.config.config_manager import ConfigManager
from src.embeddings.registry import EmbedderRegistry
from src.storage.rustfs_manager import RustFSManager

DEFAULT_SILVER_DATASET_ID = "v1_hybrid_roberta_crawl_pt_512_o64_36a9f7af2a"

QUERIES = [
    "Quem é Capitu?",
    "Quem matou Quincas Borba?",
    "Qual a Igreja do Diabo?",
    "clientes de bras cubas",
    "personagens de dom casmurro",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test de dumb retrieval: "
            "query -> embedding -> nearest neighbors no Qdrant."
        )
    )

    parser.add_argument(
        "--silver-dataset-id",
        default=DEFAULT_SILVER_DATASET_ID,
        help=(
            "Dataset Silver usado pelo índice "
            f"(default: {DEFAULT_SILVER_DATASET_ID})."
        ),
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Quantidade de chunks por query (default: 5).",
    )

    parser.add_argument(
        "--max-chars",
        type=int,
        default=700,
        help="Máximo de caracteres exibidos por chunk (default: 700).",
    )

    return parser.parse_args()


def load_matching_manifest(
    *,
    rustfs: RustFSManager,
    indexes_bucket: str,
    silver_dataset_id: str,
    model_alias: str,
    embedder_info,
) -> dict[str, Any]:
    manifests: list[dict[str, Any]] = []

    for key in rustfs.list_objects(indexes_bucket):
        if not key.endswith("/manifest.json"):
            continue

        try:
            manifest = json.loads(
                rustfs.download_bytes(
                    indexes_bucket,
                    key,
                ).decode("utf-8")
            )
        except Exception:
            continue

        source = manifest.get("source", {})
        embedding = manifest.get("embedding", {})

        if manifest.get("status") != "success":
            continue

        if source.get("silver_dataset_id") != silver_dataset_id:
            continue

        if embedding.get("model_alias") != model_alias:
            continue

        if embedding.get("model") != embedder_info.model:
            continue

        if embedding.get("resolved_revision") != embedder_info.resolved_revision:
            continue

        if int(embedding.get("dimension", -1)) != embedder_info.dimension:
            continue

        manifests.append(manifest)

    if not manifests:
        raise RuntimeError(
            "Nenhum índice compatível encontrado para "
            f"selected_model={model_alias!r}, "
            f"silver_dataset_id={silver_dataset_id!r}, "
            f"revision={embedder_info.resolved_revision!r}. "
            "Gere o índice antes de executar este smoke."
        )

    manifests.sort(
        key=lambda item: item.get("created_at_utc", ""),
        reverse=True,
    )

    return manifests[0]


def print_result(
    rank: int,
    point,
    *,
    max_chars: int,
) -> None:
    payload = point.payload or {}

    text = str(payload.get("text", "")).strip()
    compact_text = " ".join(text.split())

    if len(compact_text) > max_chars:
        compact_text = compact_text[:max_chars].rstrip() + "..."

    print(f"\n[{rank}] score={point.score:.6f}")
    print(f"    documento: {payload.get('titulo')}")
    print(f"    source_document_id: {payload.get('source_document_id')}")
    print(f"    categoria: {payload.get('categoria')}")
    print(f"    chunk_id: {payload.get('chunk_id')}")
    print(f"    chunk_index: {payload.get('chunk_index')}")
    print(f"    strategy: {payload.get('effective_strategy')}")

    section_header = payload.get("section_header")
    section_title = payload.get("section_title")

    if section_header or section_title:
        print("    seção: " f"{section_header or '-'} | " f"{section_title or '-'}")

    print("    texto:")

    print(
        textwrap.fill(
            compact_text,
            width=100,
            initial_indent="      ",
            subsequent_indent="      ",
        )
    )


def main() -> None:
    args = parse_args()

    if args.top_k <= 0:
        raise ValueError("--top-k deve ser maior que zero.")

    config = ConfigManager.load()

    model_alias = config.embedding.selected_model
    model_config = config.embedding.models[model_alias]

    registry = EmbedderRegistry()

    embedder = registry.create(
        model_config,
        normalize_embeddings=(config.embedding.index.normalize_embeddings),
        device=config.embedding.execution.device,
        batch_size=config.embedding.execution.batch_size,
    )

    info = embedder.info

    rustfs = RustFSManager.from_config(config.environment.rustfs)

    indexes_bucket = config.storage.embedding_indexes_bucket

    manifest = load_matching_manifest(
        rustfs=rustfs,
        indexes_bucket=indexes_bucket,
        silver_dataset_id=args.silver_dataset_id,
        model_alias=model_alias,
        embedder_info=info,
    )

    collection_name = manifest["qdrant"]["collection"]

    qdrant = QdrantClient(
        url=config.environment.qdrant.url,
        api_key=(config.environment.qdrant.api_key or None),
        timeout=60,
    )

    try:
        print("=== DUMB RETRIEVAL SMOKE ===")
        print(f"selected_model:    {model_alias}")
        print(f"model:             {info.model}")
        print(f"resolved_revision: {info.resolved_revision}")
        print(f"dimension:         {info.dimension}")
        print(f"query_prompt:      {info.query_prompt_name}")
        print(f"silver_dataset:    {args.silver_dataset_id}")
        print(f"index_id:          {manifest['index_id']}")
        print(f"collection:        {collection_name}")
        print(f"top_k:             {args.top_k}")

        for query in QUERIES:
            print()
            print("=" * 110)
            print(f"QUERY: {query}")
            print("=" * 110)

            query_vector = embedder.embed_query(query)

            response = qdrant.query_points(
                collection_name=collection_name,
                query=query_vector.tolist(),
                with_payload=True,
                with_vectors=False,
                limit=args.top_k,
            )

            points = response.points

            if not points:
                print("[SEM RESULTADOS]")
                continue

            for rank, point in enumerate(points, start=1):
                print_result(
                    rank,
                    point,
                    max_chars=args.max_chars,
                )

        print()
        print("[OK] Dumb retrieval executado.")

    finally:
        qdrant.close()


if __name__ == "__main__":
    main()
