"""ETL escolhido para produção; executado explicitamente, nunca no startup."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import runpy
import sys
from datetime import datetime, timezone

from botocore.exceptions import ClientError
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

STEPS = ("crawl", "bronze", "silver", "text-index", "enrichment", "enrichment-index")


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def read_json(storage, bucket, key):
    return json.loads(storage.download_bytes(bucket, key))


def write_json(storage, bucket, key, value):
    storage.upload_bytes(bucket, key, json.dumps(value, ensure_ascii=False, indent=2).encode(), "application/json")


@contextmanager
def execution_lock(storage, bucket, run_id):
    # Lock global: duas execuções não podem disputar current.json da Bronze/Silver.
    key = "production/etl.lock"
    try:
        storage.client.put_object(Bucket=bucket, Key=key, Body=run_id.encode(), IfNoneMatch="*")
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {"PreconditionFailed", "412", "ConditionalRequestConflict"}:
            raise RuntimeError("Há um ETL em execução ou um lock de execução interrompida. Confira o job antes de liberar o lock.") from exc
        raise
    try:
        yield
    finally:
        storage.delete_object(bucket, key)


def run_module(module, arguments=()):
    previous = sys.argv
    try:
        sys.argv = [module, *arguments]
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = previous


def validate_production_config(config):
    c = config.silver.chunking
    if (c.strategy, c.chunk_size, c.overlap, config.embedding.selected_model) != ("fixed", 256, 32, "colibri"):
        raise ValueError("Produção admite somente fixed/256/o32 + Colibri.")
    if config.enrichment.provider == "bedrock" and config.enrichment.model != "qwen.qwen3-32b-v1:0":
        raise ValueError("O enrichment de produção usa exclusivamente Qwen3 32B.")


def promote(storage, config, state, run_id):
    from src.storage.qdrant_manager import QdrantManager
    index_id = state["outputs"]["index_id"]
    index = read_json(storage, config.storage.embedding_indexes_bucket, f"{index_id}/manifest.json")
    enrichment_pointer = state["outputs"]["enrichment_index_pointer"]
    enrichment = read_json(storage, config.storage.embedding_indexes_bucket, enrichment_pointer)
    bronze = read_json(storage, config.storage.bronze_bucket, config.storage.bronze_current_key)
    silver = read_json(storage, config.storage.silver_bucket, config.storage.silver_current_key)
    enrichment_source = read_json(storage, config.storage.enrichment_bucket, os.environ["ENRICHMENT_DATASET_POINTER"])
    if silver["silver_dataset_id"] != index["source"]["silver_dataset_id"]:
        raise ValueError("A Silver selecionada não corresponde ao índice textual.")
    if enrichment_source["source_bronze_artifact_id"] != bronze["bronze_artifact_id"]:
        raise ValueError("O enrichment não corresponde à Bronze selecionada.")
    if enrichment["source_enrichment_dataset_id"] != enrichment_source["enrichment_dataset_id"]:
        raise ValueError("O índice auxiliar não corresponde ao dataset de enrichment selecionado.")
    if set(enrichment.get("indexes", {})) != {"colibri"}:
        raise ValueError("Promova somente a collection auxiliar Colibri escolhida.")
    qdrant = QdrantManager.from_config(config.environment.qdrant)
    try:
        if index.get("status") != "success" or enrichment.get("status") != "success":
            raise ValueError("Não é possível promover índices incompletos.")
        count = qdrant.count(collection_name=index["qdrant"]["collection"], exact=True)
        if count != index["validation"]["silver_chunks"]:
            raise ValueError("Contagem do índice textual diverge do manifest.")
        for item in enrichment["indexes"].values():
            manifest = read_json(storage, config.storage.embedding_indexes_bucket, item["manifest_key"])
            count = qdrant.count(collection_name=item["collection"], exact=True)
            if count != manifest["validation"]["qdrant_vectors"]:
                raise ValueError("Contagem do enrichment diverge do manifest.")
    finally:
        qdrant.close()
    release = {
        "status": "success", "run_id": run_id, "index_id": index_id,
        "enrichment_index_pointer": enrichment_pointer, "created_at": timestamp(),
        "bronze_pointer": config.storage.bronze_current_key, "silver_pointer": config.storage.silver_current_key,
        "raw_manifest_key": config.storage.raw_manifest_key,
        "enrichment_dataset_pointer": os.environ["ENRICHMENT_DATASET_POINTER"],
    }
    bucket = config.storage.embedding_indexes_bucket
    immutable = f"production/releases/{run_id}.json"
    if storage.object_exists(bucket, immutable):
        previous = read_json(storage, bucket, immutable)
        if any(previous.get(key) != value for key, value in release.items() if key != "created_at"):
            raise ValueError("Este run-id já possui outra promoção. Use um novo run-id.")
        release = previous
    write_json(storage, bucket, immutable, release)
    # Um único PUT seleciona as duas trilhas; objetos referenciados são imutáveis.
    write_json(storage, bucket, "production/current.json", release)
    state["release"] = release


def execute(args):
    config = ConfigManager.load()
    config.storage.raw_manifest_key = f"production/candidates/{args.run_id}/raw-manifest.parquet"
    config.storage.bronze_current_key = f"production/candidates/{args.run_id}/bronze.json"
    config.storage.silver_current_key = f"production/candidates/{args.run_id}/silver.json"
    os.environ["BRONZE_CURRENT_KEY"] = config.storage.bronze_current_key
    os.environ["SILVER_CURRENT_KEY"] = config.storage.silver_current_key
    os.environ["RAW_MANIFEST_KEY"] = config.storage.raw_manifest_key
    validate_production_config(config)
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.embedding_indexes_bucket
    if args.reuse_index_id:
        selected = read_json(storage, bucket, f"{args.reuse_index_id}/manifest.json")
        revision = selected["embedding"]["resolved_revision"]
        config.embedding.models["colibri"].revision = revision
        os.environ["COLIBRI_REVISION"] = revision
    key = f"production/runs/{args.run_id}.json"
    signature = hashlib.sha256(json.dumps({
        "source_revision": os.getenv("SOURCE_REVISION"),
        "bronze": config.bronze.model_dump(), "pages": [args.first_page, args.last_page],
        "silver": config.silver.model_dump(), "embedding": config.embedding.model_dump(exclude={"execution"}),
        "enrichment": config.enrichment.model_dump(exclude={"max_parallel", "timeout_seconds", "keep_alive", "num_batch"}), "reuse_index_id": args.reuse_index_id,
    }, sort_keys=True).encode()).hexdigest()
    state = read_json(storage, bucket, key) if storage.object_exists(bucket, key) else {
        "run_id": args.run_id, "signature": signature, "completed": [], "outputs": {},
    }
    if state["signature"] != signature:
        raise ValueError("A configuração mudou. Use outro run-id para preservar o checkpoint.")
    os.environ["BRONZE_ARTIFACT_ID"] = f"bronze_production_{args.run_id}"
    os.environ["ENRICHMENT_DATASET_POINTER"] = f"production/candidates/{args.run_id}/enrichment-dataset.json"
    os.environ["ENRICHMENT_INDEX_POINTER"] = f"production/candidates/{args.run_id}/enrichment-indexes.json"
    with execution_lock(storage, bucket, args.run_id):
        active = read_json(storage, bucket, "production/current.json") if storage.object_exists(bucket, "production/current.json") else {}
        if active.get("index_id"):
            state["outputs"].setdefault("index_id", active["index_id"])
            state["outputs"].setdefault("enrichment_index_pointer", active["enrichment_index_pointer"])
        if "crawl" not in args.steps and not storage.object_exists(config.storage.raw_bucket, config.storage.raw_manifest_key):
            raw = storage.download_bytes(config.storage.raw_bucket, active.get("raw_manifest_key", "manifests/obras_raw.parquet"))
            storage.upload_bytes(config.storage.raw_bucket, config.storage.raw_manifest_key, raw)
        for layer in ("bronze", "silver"):
            layer_bucket = getattr(config.storage, f"{layer}_bucket")
            pointer = getattr(config.storage, f"{layer}_current_key")
            if layer not in args.steps and not storage.object_exists(layer_bucket, pointer):
                source_pointer = active.get(f"{layer}_pointer", "current.json")
                if not storage.object_exists(layer_bucket, source_pointer):
                    raise ValueError(f"{layer}: execute a etapa na nuvem antes de tentar reutilizar seus artefatos.")
                write_json(storage, layer_bucket, pointer, read_json(storage, layer_bucket, source_pointer))
        if "enrichment" not in args.steps and active.get("enrichment_dataset_pointer"):
            pointer = os.environ["ENRICHMENT_DATASET_POINTER"]
            if not storage.object_exists(config.storage.enrichment_bucket, pointer):
                write_json(storage, config.storage.enrichment_bucket, pointer, read_json(storage, config.storage.enrichment_bucket, active["enrichment_dataset_pointer"]))
        for step in args.steps:
            if step in state["completed"]:
                print(f"[reuse] {step}")
                continue
            state.update(status="running", current_step=step, updated_at=timestamp())
            write_json(storage, bucket, key, state)
            if step == "crawl":
                run_module("src.etl.crawler_ingestao_raw_rustfs", ["--strict", "--pagina-inicial", str(args.first_page), "--pagina-final", str(args.last_page)])
                raw = storage.download_bytes(config.storage.raw_bucket, "manifests/obras_raw.parquet")
                storage.upload_bytes(config.storage.raw_bucket, config.storage.raw_manifest_key, raw)
            elif step == "bronze":
                run_module("src.etl.preparar_camada_bronze_rustfs")
            elif step == "silver":
                run_module("src.etl.preparar_camada_silver_chunking")
            elif step == "text-index":
                from src.storage.qdrant_manager import QdrantManager
                qdrant = QdrantManager.from_config(config.environment.qdrant)
                try:
                    if args.reuse_index_id:
                        manifest = read_json(storage, bucket, f"{args.reuse_index_id}/manifest.json")
                        if manifest.get("status") != "success" or manifest["embedding"]["model_alias"] != "colibri":
                            raise ValueError("Índice existente não corresponde à seleção de produção.")
                        if qdrant.count(collection_name=manifest["qdrant"]["collection"], exact=True) != manifest["validation"]["silver_chunks"]:
                            raise ValueError("A collection existente está ausente ou incompleta.")
                        import io
                        import pandas as pd
                        silver_manifest = pd.read_parquet(io.BytesIO(storage.download_bytes(
                            config.storage.silver_bucket, manifest["source"]["manifest_key"],
                        ))).iloc[0]
                        if (silver_manifest["strategy"], int(silver_manifest["chunk_size"]), int(silver_manifest["overlap"])) != ("fixed", 256, 32):
                            raise ValueError("A collection existente não usa fixed/256/o32.")
                    else:
                        from src.embeddings.registry import EmbedderRegistry
                        from src.indexing.embedding_index import EmbeddingIndex
                        silver = read_json(storage, config.storage.silver_bucket, config.storage.silver_current_key)
                        manifest = EmbeddingIndex(config, storage, qdrant, EmbedderRegistry()).build(
                            silver["silver_dataset_id"], "colibri", resume=True, publish_default=False,
                        )
                    state["outputs"]["index_id"] = manifest["index_id"]
                finally:
                    qdrant.close()
            elif step == "enrichment":
                run_module("src.enrichment.build_enrichment_dataset", ["--resume", "--fail-fast"])
                pointer = read_json(storage, config.storage.enrichment_bucket, os.environ["ENRICHMENT_DATASET_POINTER"])
                state["outputs"]["enrichment_dataset_id"] = pointer["enrichment_dataset_id"]
            elif step == "enrichment-index":
                run_module("src.indexing.build_enrichment_embedding_indexes", ["--embedders", "colibri", "--resume"])
                state["outputs"]["enrichment_index_pointer"] = os.environ["ENRICHMENT_INDEX_POINTER"]
            state["completed"].append(step)
            state.update(status="staged", updated_at=timestamp())
            write_json(storage, bucket, key, state)
            print(f"[ok] {step}")
        if args.promote:
            if "release" not in state:
                promote(storage, config, state, args.run_id)
            else:
                write_json(storage, bucket, "production/current.json", state["release"])
            state["status"] = "promoted"
            write_json(storage, bucket, key, state)
            print(f"[ok] dados promovidos: {args.run_id}")
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS[:4]))
    parser.add_argument("--reuse-index-id")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--first-page", type=int, default=1)
    parser.add_argument("--last-page", type=int, default=7)
    args = parser.parse_args()
    if args.steps != sorted(set(args.steps), key=STEPS.index):
        parser.error("Informe as etapas uma vez, na ordem crawl, bronze, silver, text-index, enrichment, enrichment-index")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", args.run_id):
        parser.error("run-id deve conter somente letras, números, _ e -")
    if args.first_page < 1 or args.last_page < args.first_page:
        parser.error("Intervalo de páginas inválido")
    execute(args)


if __name__ == "__main__":
    main()
