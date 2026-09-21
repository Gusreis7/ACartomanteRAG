#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.config.config_manager import ConfigManager
from src.utils.chunking import (
    HuggingFaceChunkTokenizer,
    build_sections,
)
from src.storage.rustfs_manager import RustFSManager

BRONZE_MANIFEST_KEY = "manifests/obras_bronze.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Gera uma versão da camada Silver " "a partir dos textos Bronze.")
    )

    parser.add_argument(
        "--env-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--config-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--strategy",
        choices=["fixed", "hybrid"],
        default=None,
        help=(
            "Sobrescreve silver.chunking.strategy do config.yaml "
            "somente para esta execução."
        ),
    )

    parser.add_argument(
        "--tokenizer",
        default=None,
        help=(
            "Sobrescreve silver.chunking.tokenizer do config.yaml "
            "somente para esta execução."
        ),
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Sobrescreve silver.chunking.chunk_size do config.yaml "
            "somente para esta execução."
        ),
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=None,
        help=(
            "Sobrescreve silver.chunking.overlap do config.yaml "
            "somente para esta execução."
        ),
    )

    return parser.parse_args()


def salvar_parquet(
    storage: RustFSManager,
    bucket: str,
    key: str,
    df: pd.DataFrame,
) -> None:
    buffer = io.BytesIO()

    df.to_parquet(
        buffer,
        index=False,
    )

    storage.upload_bytes(
        bucket=bucket,
        key=key,
        data=buffer.getvalue(),
        content_type=("application/vnd.apache.parquet"),
    )


def build_dataset_id(
    pipeline_version: int,
    strategy: str,
    tokenizer_alias: str,
    chunk_size: int,
    overlap: int,
    tokenizer_model: str,
    tokenizer_revision: str,
    bronze_manifest_sha256: str,
    semantic_config: dict,
) -> tuple[str, str]:
    signature = {
        "pipeline_version": (pipeline_version),
        "strategy": strategy,
        "tokenizer_alias": (tokenizer_alias),
        "tokenizer_model": (tokenizer_model),
        "tokenizer_revision": (tokenizer_revision),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "bronze_manifest_sha256": (bronze_manifest_sha256),
        "semantic": semantic_config,
    }

    config_hash = hashlib.sha256(
        json.dumps(
            signature,
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()[:10]

    tokenizer_name = (
        re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            tokenizer_alias,
        )
        .strip("_")
        .lower()
    )

    dataset_id = (
        f"v{pipeline_version}_"
        f"{strategy}_"
        f"{tokenizer_name}_"
        f"{chunk_size}_"
        f"o{overlap}_"
        f"{config_hash}"
    )

    return (
        dataset_id,
        config_hash,
    )


def main() -> None:
    args = parse_args()

    config_kwargs = {}

    if args.env_file is not None:
        config_kwargs["env_file"] = args.env_file

    if args.config_file is not None:
        config_kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**config_kwargs)

    storage = RustFSManager.from_config(config.environment.rustfs)

    bronze_bucket = config.storage.bronze_bucket
    silver_bucket = config.storage.silver_bucket

    if not storage.bucket_exists(bronze_bucket):
        raise RuntimeError("Bucket Bronze não encontrado: " f"{bronze_bucket}")

    storage.create_bucket(silver_bucket)

    bronze_manifest_bytes = storage.download_bytes(
        bucket=bronze_bucket,
        key=BRONZE_MANIFEST_KEY,
    )

    bronze_manifest_sha256 = hashlib.sha256(bronze_manifest_bytes).hexdigest()

    bronze_manifest = pd.read_parquet(io.BytesIO(bronze_manifest_bytes))

    if "status_bronze" in (bronze_manifest.columns):
        bronze_manifest = (
            bronze_manifest[bronze_manifest["status_bronze"].eq("success")]
            .copy()
            .reset_index(drop=True)
        )

    if bronze_manifest.empty:
        raise RuntimeError("Manifest Bronze não possui " "documentos processáveis.")

    chunking_config = config.silver.chunking

    strategy = (
        args.strategy
        if args.strategy is not None
        else chunking_config.strategy
    )

    tokenizer_alias = (
        args.tokenizer
        if args.tokenizer is not None
        else chunking_config.tokenizer
    )

    chunk_size = (
        args.chunk_size
        if args.chunk_size is not None
        else chunking_config.chunk_size
    )

    overlap = (
        args.overlap
        if args.overlap is not None
        else chunking_config.overlap
    )

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size deve ser maior que zero."
        )

    if overlap < 0:
        raise ValueError(
            "overlap não pode ser negativo."
        )

    if overlap >= chunk_size:
        raise ValueError(
            "overlap deve ser menor que chunk_size."
        )

    if tokenizer_alias not in config.silver.tokenizers:
        tokenizers_disponiveis = ", ".join(
            sorted(config.silver.tokenizers)
        )

        raise ValueError(
            f"Tokenizer '{tokenizer_alias}' não existe em "
            "silver.tokenizers. Disponíveis: "
            f"{tokenizers_disponiveis}"
        )

    tokenizer_config = config.silver.tokenizers[
        tokenizer_alias
    ]

    tokenizer = HuggingFaceChunkTokenizer(
        model=tokenizer_config.model,
        revision=(tokenizer_config.revision),
    )

    semantic_config = config.silver.semantic

    semantic_signature = (
        {
            "min_capitulos": (
                semantic_config.min_capitulos
            ),
            "min_mediana_chars": (
                semantic_config.min_mediana_chars
            ),
        }
        if strategy == "hybrid"
        else {}
    )

    dataset_id, config_hash = build_dataset_id(
        pipeline_version=(config.silver.pipeline_version),
        strategy=strategy,
        tokenizer_alias=tokenizer_alias,
        chunk_size=chunk_size,
        overlap=overlap,
        tokenizer_model=(tokenizer_config.model),
        tokenizer_revision=(tokenizer.resolved_revision),
        bronze_manifest_sha256=(bronze_manifest_sha256),
        semantic_config=(semantic_signature),
    )

    dataset_prefix = f"datasets/{dataset_id}"

    print(f"[+] Silver dataset: " f"{dataset_id}")

    print("[+] Tokenizer: " f"{tokenizer_config.model}")

    print(
        "[+] Chunking: "
        f"{strategy} | "
        f"{chunk_size} "
        f"tokens | "
        f"overlap="
        f"{overlap}"
    )

    chunks = []

    for row_index, row in bronze_manifest.iterrows():
        title = str(
            row.get(
                "titulo",
                "",
            )
        )

        category = str(
            row.get(
                "categoria",
                "",
            )
        )

        bronze_object_key = row.get("bronze_object_key")

        if pd.isna(bronze_object_key) or not bronze_object_key:
            print("[AVISO] Sem " "bronze_object_key: " f"{title}")
            continue

        text = storage.download_bytes(
            bucket=bronze_bucket,
            key=str(bronze_object_key),
        ).decode(
            "utf-8",
            errors="replace",
        )

        if not text.strip():
            print(f"[AVISO] Texto vazio: " f"{title}")
            continue

        sections, effective_strategy = build_sections(
            text=text,
            title=title,
            strategy=strategy,
            min_capitulos=(semantic_config.min_capitulos),
            min_mediana_chars=(semantic_config.min_mediana_chars),
        )

        source_document_id = row.get("source_document_id")

        if pd.isna(source_document_id):
            source_document_id = ""
        elif (
            isinstance(
                source_document_id,
                float,
            )
            and source_document_id.is_integer()
        ):
            source_document_id = str(int(source_document_id))
        else:
            source_document_id = str(source_document_id)

        chunk_index = 0

        for section in sections:
            section_chunks = tokenizer.split(
                text=section["text"],
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for (
                chunk_index_in_section,
                chunk,
            ) in enumerate(section_chunks):
                document_char_start = section["start_char"] + chunk["char_start"]

                document_char_end = section["start_char"] + chunk["char_end"]

                chunk_identity = (
                    f"{dataset_id}|"
                    f"{source_document_id}|"
                    f"{section['section_index']}|"
                    f"{chunk_index}|"
                    f"{chunk['text']}"
                )

                chunk_id = hashlib.sha256(chunk_identity.encode("utf-8")).hexdigest()[
                    :24
                ]

                chunks.append(
                    {
                        "chunk_id": (chunk_id),
                        "silver_dataset_id": (dataset_id),
                        "source_document_id": (source_document_id),
                        "titulo": title,
                        "autor": row.get("autor"),
                        "categoria": (category),
                        "source_bronze_bucket": (bronze_bucket),
                        "source_bronze_object_key": (bronze_object_key),
                        "requested_strategy": strategy,
                        "effective_strategy": (effective_strategy),
                        "section_index": (section["section_index"]),
                        "section_type": (section["section_type"]),
                        "section_number": (section["section_number"]),
                        "section_header": (section["section_header"]),
                        "section_title": (section["section_title"]),
                        "section_title_rule": (section["section_title_rule"]),
                        "chapter_rule": (section["chapter_rule"]),
                        "chunk_index": (chunk_index),
                        "chunk_index_in_section": (chunk_index_in_section),
                        "chunk_tokenizer": tokenizer_alias,
                        "chunk_tokenizer_model": (tokenizer_config.model),
                        "chunk_tokenizer_revision": (tokenizer.resolved_revision),
                        "chunk_size": chunk_size,
                        "overlap": overlap,
                        "token_count": (chunk["token_count"]),
                        "start_char": (document_char_start),
                        "end_char": (document_char_end),
                        "text": (chunk["text"]),
                    }
                )

                chunk_index += 1

        print(
            f"[{row_index + 1}/"
            f"{len(bronze_manifest)}] "
            f"{title}: "
            f"{chunk_index} chunks | "
            f"{effective_strategy}"
        )

    if not chunks:
        raise RuntimeError("Nenhum chunk foi gerado.")

    df_chunks = pd.DataFrame(chunks)

    chunks_key = f"{dataset_prefix}/" "chunks.parquet"

    manifest_key = f"{dataset_prefix}/" "manifest.parquet"

    salvar_parquet(
        storage=storage,
        bucket=silver_bucket,
        key=chunks_key,
        df=df_chunks,
    )

    manifest = pd.DataFrame(
        [
            {
                "silver_dataset_id": (dataset_id),
                "pipeline_version": (config.silver.pipeline_version),
                "config_hash": (config_hash),
                "created_at_utc": (datetime.now(timezone.utc).isoformat()),
                "source_bronze_bucket": (bronze_bucket),
                "source_bronze_manifest_key": (BRONZE_MANIFEST_KEY),
                "source_bronze_manifest_sha256": (bronze_manifest_sha256),
                "strategy": strategy,
                "chunk_size": chunk_size,
                "overlap": overlap,
                "chunk_tokenizer": tokenizer_alias,
                "chunk_tokenizer_model": (tokenizer_config.model),
                "chunk_tokenizer_revision": (tokenizer.resolved_revision),
                "min_capitulos": (semantic_config.min_capitulos),
                "min_mediana_chars": (semantic_config.min_mediana_chars),
                "documents": (df_chunks["source_document_id"].nunique()),
                "chunks": len(df_chunks),
            }
        ]
    )

    salvar_parquet(
        storage=storage,
        bucket=silver_bucket,
        key=manifest_key,
        df=manifest,
    )

    print()
    print("[OK] Camada Silver criada.")

    print(f"     Dataset: " f"s3://{silver_bucket}/" f"{dataset_prefix}/")

    print(f"     Chunks: " f"{len(df_chunks)}")

    print("     Estratégias efetivas:")

    print(df_chunks["effective_strategy"].value_counts().to_string())


if __name__ == "__main__":
    main()
