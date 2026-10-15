"""Copia Raw, Bronze e Silver selecionada para S3 sem levar a matriz experimental."""
from __future__ import annotations
import argparse
import hashlib
import io
import json
from pathlib import Path
import tempfile

import pandas as pd
import yaml
from dotenv import dotenv_values
from src.storage.rustfs_manager import RustFSManager
from src.config.config_manager import StorageConfig


def digest(file):
    checksum = hashlib.sha256()
    with file.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def copy_verified(source, target, source_bucket, destination_bucket, key):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "object"
        source.download_file(source_bucket, key, path)
        sha = digest(path)
        if target.object_exists(destination_bucket, key):
            # ETag não é checksum confiável para upload multipart.
            existing = Path(tmp) / "existing"
            target.download_file(destination_bucket, key, existing)
            if digest(existing) != sha:
                raise ValueError(f"Destino já contém bytes diferentes: {destination_bucket}/{key}. Nenhum overwrite automático.")
            return False
        target.client.upload_file(str(path), destination_bucket, key, ExtraArgs={"Metadata": {"source-sha256": sha}})
        head = target.client.head_object(Bucket=destination_bucket, Key=key)
        if head["ContentLength"] != path.stat().st_size or head.get("Metadata", {}).get("source-sha256") != sha:
            raise ValueError(f"Falha na verificação da cópia: {destination_bucket}/{key}")
        return True


def put_pointer(target, bucket, key, payload):
    if target.object_exists(bucket, key):
        old = json.loads(target.download_bytes(bucket, key))
        if old != payload:
            raise ValueError(f"Ponteiro de destino já existe e diverge: {bucket}/{key}.")
        return
    target.upload_bytes(bucket, key, json.dumps(payload, ensure_ascii=False, indent=2).encode(), "application/json")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-env", type=Path, required=True)
    p.add_argument("--config-file", type=Path, default=Path("config/config.yaml"))
    p.add_argument("--source-endpoint", help="No profile tools: http://rustfs:9000")
    p.add_argument("--outputs", type=Path, required=True)
    p.add_argument("--index-id")
    p.add_argument("--apply", action="store_true", help="Sem esta opção, apenas mostra o plano")
    args = p.parse_args()
    env = dotenv_values(args.source_env)
    config = yaml.safe_load(args.config_file.read_text())
    buckets = StorageConfig.model_validate(config.get("storage", {}))
    source = RustFSManager(args.source_endpoint or env["RUSTFS_ENDPOINT"], env["RUSTFS_ACCESS_KEY"], env["RUSTFS_SECRET_KEY"])
    output = json.loads(args.outputs.read_text())
    destination = output["buckets"]["value"]
    target = RustFSManager(region=output["region"]["value"])
    index_id = args.index_id or config["retrieval_index"].get("index_id")
    if not index_id:
        index_id = json.loads(source.download_bytes(buckets.embedding_indexes_bucket, "releases/default.json"))["index_id"]
    index_key = f"{index_id}/manifest.json"
    index = json.loads(source.download_bytes(buckets.embedding_indexes_bucket, index_key))
    if index.get("status") != "success" or index["embedding"]["model_alias"] != "colibri":
        raise ValueError("Somente o índice Colibri validado pode ser migrado por este script.")
    silver_id = index["source"]["silver_dataset_id"]
    silver_prefix = f"datasets/{silver_id}/"
    silver_key = f"{silver_prefix}manifest.parquet"
    silver = pd.read_parquet(io.BytesIO(source.download_bytes(buckets.silver_bucket, silver_key))).iloc[0]
    if (silver["strategy"], int(silver["chunk_size"]), int(silver["overlap"])) != ("fixed", 256, 32):
        raise ValueError("O índice selecionado não usa fixed/256/o32.")
    bronze_id = str(silver["source_bronze_artifact_id"])
    bronze_manifest_key = str(silver["source_bronze_manifest_key"])
    bronze_manifest = source.download_bytes(buckets.bronze_bucket, bronze_manifest_key)
    bronze_artifact_key = f"datasets/{bronze_id}/artifact.json"
    bronze_pointer = {
        "bronze_artifact_id": bronze_id, "manifest_key": bronze_manifest_key,
        "artifact_key": bronze_artifact_key, "manifest_sha256": hashlib.sha256(bronze_manifest).hexdigest(),
    }
    silver_pointer = {"silver_dataset_id": silver_id, "manifest_key": silver_key,
                      "chunks_key": f"{silver_prefix}chunks.parquet", "source_bronze_artifact_id": bronze_id}
    selections = [
        (buckets.raw_bucket, destination["RAW_BUCKET"], source.list_objects(buckets.raw_bucket)),
        (buckets.bronze_bucket, destination["BRONZE_BUCKET"], source.list_objects(buckets.bronze_bucket, f"datasets/{bronze_id}/")),
        (buckets.silver_bucket, destination["SILVER_BUCKET"], source.list_objects(buckets.silver_bucket, silver_prefix)),
        (buckets.embedding_indexes_bucket, destination["EMBEDDING_INDEXES_BUCKET"], [index_key]),
    ]
    for origin, dest, keys in selections:
        if not keys:
            raise ValueError(f"Não há artefatos selecionados em {origin}")
        print(f"{origin} -> {dest}: {len(keys)} objeto(s)")
    if not args.apply:
        print("Plano pronto; use --apply para copiar. Collections Qdrant existentes são preservadas.")
        return
    count = 0
    for origin, dest, keys in selections:
        for key in keys:
            count += int(copy_verified(source, target, origin, dest, key))
    put_pointer(target, destination["BRONZE_BUCKET"], "current.json", bronze_pointer)
    put_pointer(target, destination["SILVER_BUCKET"], "current.json", silver_pointer)
    print(f"Migração concluída: {count} objeto(s) novo(s); bytes existentes iguais reutilizados. Enrichment não foi copiado nem gerado.")


if __name__ == "__main__":
    main()
