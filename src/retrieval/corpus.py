from __future__ import annotations

from io import BytesIO

import pandas as pd

REQUIRED_CHUNK_COLUMNS = {"chunk_id", "silver_dataset_id", "text"}


def load_resolved_silver_chunks(config, rustfs, resolved_index) -> pd.DataFrame:
    """Carrega a Silver exata registrada no manifest do índice resolvido."""
    source = resolved_index.manifest.get("source") or {}
    silver_dataset_id = resolved_index.silver_dataset_id
    silver_bucket = source.get("silver_bucket") or config.storage.silver_bucket
    chunks_key = source.get("chunks_key") or (
        f"datasets/{silver_dataset_id}/chunks.parquet"
    )

    if not rustfs.object_exists(silver_bucket, chunks_key):
        raise FileNotFoundError(
            "Chunks da Silver associados ao índice não encontrados: "
            f"s3://{silver_bucket}/{chunks_key}"
        )

    payload = rustfs.download_bytes(silver_bucket, chunks_key)
    chunks = pd.read_parquet(BytesIO(payload))

    missing = REQUIRED_CHUNK_COLUMNS - set(chunks.columns)
    if missing:
        raise ValueError(
            "Silver sem colunas obrigatórias: " + ", ".join(sorted(missing))
        )

    internal_ids = set(
        chunks["silver_dataset_id"].dropna().astype(str).unique()
    )
    if internal_ids != {silver_dataset_id}:
        raise ValueError(
            "Silver carregada não corresponde ao índice resolvido. "
            f"Esperado={silver_dataset_id}; encontrado={sorted(internal_ids)}"
        )

    chunk_ids = chunks["chunk_id"].astype(str)
    if chunk_ids.duplicated().any():
        duplicated = chunk_ids[chunk_ids.duplicated(keep=False)].head(10).tolist()
        raise ValueError(f"chunk_id duplicado na Silver: {duplicated}")

    return chunks.reset_index(drop=True)
