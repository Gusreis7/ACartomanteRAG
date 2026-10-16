#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any
from urllib import request as urllib_request

import pandas as pd
from tqdm.auto import tqdm

from src.config.config_manager import ConfigManager, ENRICHMENT_RUNTIME_FIELDS
from src.generation.provider import GenerationServiceError
from src.enrichment.service import (
    WorkEnrichmentService,
    WorkSource,
    prompt_sha256,
    structured_output_schema_sha256,
)
from src.storage.rustfs_manager import RustFSManager


BRONZE_CURRENT_KEY = "current.json"
ENRICHMENT_CURRENT_KEY = os.getenv("ENRICHMENT_DATASET_POINTER", "current.json")
ENRICHMENT_DATASETS_PREFIX = "datasets"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera o dataset versionado de enrichment das obras Bronze "
            "e persiste uma obra por vez no RustFS."
        )
    )
    parser.add_argument(
        "--resume", "--retry",
        action="store_true",
        help="Retoma o artifact determinístico atual, pulando obras success.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help=(
            "Processa no máximo N obras nesta execução. "
            "As demais permanecem pending."
        ),
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Processa somente um título pendente/erro.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Interrompe no primeiro erro; por padrão persiste o erro e continua.",
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    return parser.parse_args()


def normalize_text(value: Any) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def safe_component(value: str) -> str:
    value = normalize_text(value)
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "na"


def safe_filename(value: str) -> str:
    value = re.sub(r'[<>:"/\\|?*]', "", str(value).strip())
    value = re.sub(r"\s+", "_", value).strip("._")
    return value[:140] or "sem_titulo"


def json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        indent=2,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def upload_json(
    storage: RustFSManager,
    *,
    bucket: str,
    key: str,
    payload: dict[str, Any],
) -> tuple[bytes, str]:
    data = json_bytes(payload)
    storage.upload_bytes(
        bucket=bucket,
        key=key,
        data=data,
        content_type="application/json",
    )
    return data, sha256_bytes(data)


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    return buffer.getvalue()


def upload_manifest(
    storage: RustFSManager,
    *,
    bucket: str,
    key: str,
    frame: pd.DataFrame,
) -> tuple[bytes, str]:
    data = parquet_bytes(frame)
    storage.upload_bytes(
        bucket=bucket,
        key=key,
        data=data,
        content_type="application/vnd.apache.parquet",
    )
    return data, sha256_bytes(data)


def load_parquet_bytes(data: bytes) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(data))


def resolve_ollama_model_digest(*, base_url: str, model: str) -> str:
    endpoint = base_url.rstrip("/") + "/api/tags"
    try:
        with urllib_request.urlopen(endpoint, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except Exception:
        return ""

    wanted = str(model).strip()
    for item in payload.get("models", []) or []:
        names = {
            str(item.get("name", "")).strip(),
            str(item.get("model", "")).strip(),
        }
        if wanted in names:
            return str(item.get("digest", "")).strip()
    return ""


def load_bronze_state(
    *,
    storage: RustFSManager,
    bucket: str,
) -> tuple[dict[str, Any], pd.DataFrame, str]:
    if not storage.bucket_exists(bucket):
        raise RuntimeError(f"Bucket Bronze não encontrado: {bucket}")

    current = json.loads(
        storage.download_bytes(
            bucket=bucket,
            key=BRONZE_CURRENT_KEY,
        ).decode("utf-8")
    )

    bronze_artifact_id = str(current.get("bronze_artifact_id", "")).strip()
    manifest_key = str(current.get("manifest_key", "")).strip()
    if not bronze_artifact_id or not manifest_key:
        raise RuntimeError(
            "current.json da Bronze inválido: bronze_artifact_id/manifest_key ausentes."
        )

    manifest_bytes = storage.download_bytes(bucket=bucket, key=manifest_key)
    manifest_sha256 = sha256_bytes(manifest_bytes)
    expected_sha = str(current.get("manifest_sha256", "")).strip()
    if expected_sha and expected_sha != manifest_sha256:
        raise RuntimeError(
            "Falha de integridade no manifest Bronze. "
            f"Esperado={expected_sha}; obtido={manifest_sha256}."
        )

    manifest = load_parquet_bytes(manifest_bytes)
    artifact_ids = set(
        manifest["bronze_artifact_id"].dropna().astype(str).unique()
    )
    if artifact_ids != {bronze_artifact_id}:
        raise RuntimeError(
            "bronze_artifact_id do manifest não corresponde ao current.json. "
            f"current={bronze_artifact_id}; manifest={sorted(artifact_ids)}."
        )

    if "status_bronze" in manifest.columns:
        manifest = (
            manifest[manifest["status_bronze"].eq("success")]
            .copy()
            .reset_index(drop=True)
        )

    if manifest.empty:
        raise RuntimeError("Manifest Bronze não possui documentos processáveis.")

    return current, manifest, manifest_sha256


def build_signature(
    *,
    config,
    bronze_artifact_id: str,
    bronze_manifest_sha256: str,
    model_digest: str,
    prompt_hash: str,
    schema_hash: str,
) -> tuple[dict[str, Any], str, str]:
    enrichment = config.enrichment
    signature = {
        "pipeline_version": enrichment.pipeline_version,
        "source_bronze_artifact_id": bronze_artifact_id,
        "source_bronze_manifest_sha256": bronze_manifest_sha256,
        "model": enrichment.model,
        "provider": enrichment.provider,
        "region": enrichment.region if enrichment.provider in {"bedrock", "bedrock-mantle"} else None,
        "model_digest": model_digest,
        "prompt_version": enrichment.prompt_version,
        "prompt_sha256": prompt_hash,
        "schema_version": enrichment.schema_version,
        "schema_sha256": schema_hash,
        "strategy": enrichment.strategy,
        "unit_chars": enrichment.unit_chars,
        "char_overlap": enrichment.char_overlap,
        "final_context_chars": enrichment.final_context_chars,
        "temperature": enrichment.temperature,
        "top_p": enrichment.top_p,
        "max_tokens": enrichment.max_tokens,
        "num_ctx": enrichment.num_ctx,
        "reasoning": enrichment.reasoning,
        "semantic": enrichment.semantic.model_dump(mode="json"),
    }

    config_hash = hashlib.sha256(
        json.dumps(signature, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]

    dataset_id = (
        f"enrichment_v{enrichment.pipeline_version}_"
        f"{safe_component(enrichment.model)}_"
        f"{safe_component(enrichment.prompt_version)}_"
        f"{safe_component(enrichment.strategy)}_"
        f"{config_hash}"
    )
    return signature, config_hash, dataset_id


def work_object_key(
    *,
    dataset_prefix: str,
    source_document_id: str,
    title: str,
) -> str:
    document = re.sub(
        r"[^0-9A-Za-z_-]+",
        "-",
        str(source_document_id).strip(),
    ).strip("-") or "sem-id"
    return (
        f"{dataset_prefix}/works/"
        f"{document}_{safe_filename(title)}.json"
    )


def normalize_source_document_id(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def build_initial_manifest(
    *,
    bronze_manifest: pd.DataFrame,
    dataset_id: str,
    bronze_artifact_id: str,
    enrichment_bucket: str,
    dataset_prefix: str,
    config,
    model_digest: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in bronze_manifest.sort_values("titulo", na_position="last").iterrows():
        source_document_id = normalize_source_document_id(
            row.get("source_document_id", "")
        )
        title = str(row.get("titulo", ""))
        object_key = work_object_key(
            dataset_prefix=dataset_prefix,
            source_document_id=source_document_id,
            title=title,
        )
        rows.append({
            "enrichment_dataset_id": dataset_id,
            "pipeline_version": config.enrichment.pipeline_version,
            "source_bronze_artifact_id": bronze_artifact_id,
            "source_document_id": source_document_id,
            "titulo": title,
            "autor": row.get("autor"),
            "categoria": row.get("categoria"),
            "source_bronze_bucket": config.storage.bronze_bucket,
            "source_bronze_object_key": row.get("bronze_object_key"),
            "enrichment_bucket": enrichment_bucket,
            "enrichment_object_key": object_key,
            "enrichment_uri": f"s3://{enrichment_bucket}/{object_key}",
            "model": config.enrichment.model,
            "model_digest": model_digest,
            "requested_strategy": config.enrichment.strategy,
            "effective_strategy": "",
            "plan_mode": "",
            "blocks": 0,
            "source_chars": 0,
            "output_json_bytes": 0,
            "elapsed_seconds": 0.0,
            "reduce_calls": 0,
            "source_text_sha256": "",
            "work_sha256": "",
            "status_enrichment": "pending",
            "processed_at_utc": "",
            "erro_enrichment": "",
            "tentativas_enrichment": 0,
            "llm_calls": 0, "llm_input_tokens": 0, "llm_output_tokens": 0,
            "erro_tipo": "", "erro_detalhes_json": "",
        })

    return pd.DataFrame(rows).reset_index(drop=True)


def ensure_retry_columns(manifest):
    # Compatível com manifests existentes; identidade e objetos success permanecem.
    for field, default in {"tentativas_enrichment": 0, "llm_calls": 0, "llm_input_tokens": 0,
                           "llm_output_tokens": 0, "erro_tipo": "", "erro_detalhes_json": ""}.items():
        if field not in manifest:
            manifest[field] = default
    return manifest


def artifact_payload(
    *,
    dataset_id: str,
    config_hash: str,
    signature: dict[str, Any],
    config,
    bronze_current: dict[str, Any],
    bronze_manifest_sha256: str,
    dataset_prefix: str,
    manifest_key: str,
    manifest_sha256: str,
    manifest: pd.DataFrame,
    status: str,
    created_at_utc: str,
) -> dict[str, Any]:
    counts = manifest["status_enrichment"].value_counts().to_dict()
    return {
        "artifact_version": config.enrichment.pipeline_version,
        "enrichment_dataset_id": dataset_id,
        "config_hash": config_hash,
        "status": status,
        "created_at_utc": created_at_utc,
        "updated_at_utc": utc_now(),
        "enrichment_bucket": config.storage.enrichment_bucket,
        "dataset_prefix": dataset_prefix,
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "source_bronze_bucket": config.storage.bronze_bucket,
        "source_bronze_artifact_id": bronze_current["bronze_artifact_id"],
        "source_bronze_manifest_key": bronze_current["manifest_key"],
        "source_bronze_manifest_sha256": bronze_manifest_sha256,
        "documents_total": int(len(manifest)),
        "documents_pending": int(counts.get("pending", 0)),
        "documents_success": int(counts.get("success", 0)),
        "documents_error": int(counts.get("error", 0)),
        "signature": signature,
        "enrichment_config": config.enrichment.model_dump(mode="json"),
    }


def sync_manifest_and_artifact(
    *,
    storage: RustFSManager,
    bucket: str,
    manifest_key: str,
    artifact_key: str,
    manifest: pd.DataFrame,
    dataset_id: str,
    config_hash: str,
    signature: dict[str, Any],
    config,
    bronze_current: dict[str, Any],
    bronze_manifest_sha256: str,
    dataset_prefix: str,
    status: str,
    created_at_utc: str,
) -> tuple[str, str]:
    _, manifest_sha = upload_manifest(
        storage,
        bucket=bucket,
        key=manifest_key,
        frame=manifest,
    )
    artifact = artifact_payload(
        dataset_id=dataset_id,
        config_hash=config_hash,
        signature=signature,
        config=config,
        bronze_current=bronze_current,
        bronze_manifest_sha256=bronze_manifest_sha256,
        dataset_prefix=dataset_prefix,
        manifest_key=manifest_key,
        manifest_sha256=manifest_sha,
        manifest=manifest,
        status=status,
        created_at_utc=created_at_utc,
    )
    _, artifact_sha = upload_json(
        storage,
        bucket=bucket,
        key=artifact_key,
        payload=artifact,
    )
    return manifest_sha, artifact_sha


def select_execution_rows(
    *,
    manifest: pd.DataFrame,
    title: str | None,
    limit: int | None,
) -> list[int]:
    candidates = manifest[
        ~manifest["status_enrichment"].eq("success")
    ].copy()

    if title:
        wanted = normalize_text(title)
        normalized = candidates["titulo"].fillna("").astype(str).map(normalize_text)
        candidates = candidates[normalized.eq(wanted)]
        if candidates.empty:
            raise RuntimeError(
                f"Nenhuma obra pendente/erro encontrada para o título: {title}"
            )

    if limit is not None:
        if limit < 0:
            raise ValueError("--limit deve ser >= 0.")
        candidates = candidates.head(limit)

    return [int(index) for index in candidates.index]


def final_status_from_manifest(manifest: pd.DataFrame) -> str:
    counts = manifest["status_enrichment"].value_counts().to_dict()
    pending = int(counts.get("pending", 0))
    errors = int(counts.get("error", 0))
    if pending == 0 and errors == 0:
        return "complete"
    if pending == 0:
        return "complete_with_errors"
    return "partial"


def main() -> None:
    args = parse_args()
    config_kwargs: dict[str, Any] = {}
    if args.env_file is not None:
        config_kwargs["env_file"] = args.env_file
    if args.config_file is not None:
        config_kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**config_kwargs)
    storage = RustFSManager.from_config(config.environment.rustfs)
    bronze_bucket = config.storage.bronze_bucket
    enrichment_bucket = config.storage.enrichment_bucket
    global BRONZE_CURRENT_KEY
    BRONZE_CURRENT_KEY = config.storage.bronze_current_key
    storage.create_bucket(enrichment_bucket)

    bronze_current, bronze_manifest, bronze_manifest_sha256 = load_bronze_state(
        storage=storage,
        bucket=bronze_bucket,
    )
    bronze_artifact_id = str(bronze_current["bronze_artifact_id"])

    model_digest = (
        f"{config.enrichment.provider}:{config.enrichment.region}:{config.enrichment.model}"
        if config.enrichment.provider in {"bedrock", "bedrock-mantle"} else resolve_ollama_model_digest(
            base_url=config.generation.base_url, model=config.enrichment.model,
        )
    )
    prompt_hash = prompt_sha256()
    schema_hash = structured_output_schema_sha256()
    signature, config_hash, dataset_id = build_signature(
        config=config,
        bronze_artifact_id=bronze_artifact_id,
        bronze_manifest_sha256=bronze_manifest_sha256,
        model_digest=model_digest,
        prompt_hash=prompt_hash,
        schema_hash=schema_hash,
    )

    dataset_prefix = f"{ENRICHMENT_DATASETS_PREFIX}/{dataset_id}"
    manifest_key = f"{dataset_prefix}/manifest.parquet"
    artifact_key = f"{dataset_prefix}/artifact.json"
    created_at_utc = utc_now()

    manifest_exists = storage.object_exists(enrichment_bucket, manifest_key)
    artifact_exists = storage.object_exists(enrichment_bucket, artifact_key)

    if manifest_exists or artifact_exists:
        if not args.resume:
            raise RuntimeError(
                "O enrichment artifact já existe. Use --resume para retomá-lo.\n"
                f"Artifact: {dataset_id}"
            )
        if not (manifest_exists and artifact_exists):
            raise RuntimeError(
                "Artifact inconsistente: manifest.parquet e artifact.json "
                "não existem em conjunto."
            )

        manifest = load_parquet_bytes(
            storage.download_bytes(bucket=enrichment_bucket, key=manifest_key)
        )
        artifact = json.loads(
            storage.download_bytes(bucket=enrichment_bucket, key=artifact_key)
            .decode("utf-8")
        )
        if str(artifact.get("config_hash", "")) != config_hash:
            raise RuntimeError(
                "config_hash do artifact existente não corresponde ao config atual."
            )
        created_at_utc = str(artifact.get("created_at_utc", created_at_utc))

        # Não confia em success sem o objeto correspondente.
        for index, row in manifest[
            manifest["status_enrichment"].eq("success")
        ].iterrows():
            object_key = str(row["enrichment_object_key"])
            if not storage.object_exists(enrichment_bucket, object_key):
                manifest.at[index, "status_enrichment"] = "pending"
                manifest.at[index, "erro_enrichment"] = (
                    "Objeto de enrichment ausente; marcado para reconstrução."
                )
    else:
        manifest = build_initial_manifest(
            bronze_manifest=bronze_manifest,
            dataset_id=dataset_id,
            bronze_artifact_id=bronze_artifact_id,
            enrichment_bucket=enrichment_bucket,
            dataset_prefix=dataset_prefix,
            config=config,
            model_digest=model_digest,
        )
        sync_manifest_and_artifact(
            storage=storage,
            bucket=enrichment_bucket,
            manifest_key=manifest_key,
            artifact_key=artifact_key,
            manifest=manifest,
            dataset_id=dataset_id,
            config_hash=config_hash,
            signature=signature,
            config=config,
            bronze_current=bronze_current,
            bronze_manifest_sha256=bronze_manifest_sha256,
            dataset_prefix=dataset_prefix,
            status="running",
            created_at_utc=created_at_utc,
        )

    manifest = ensure_retry_columns(manifest)
    execution_rows = select_execution_rows(
        manifest=manifest,
        title=args.title,
        limit=args.limit,
    )

    print(f"[+] Enrichment dataset: {dataset_id}")
    print(f"[+] Bronze source: {bronze_artifact_id}")
    print(
        f"[+] Model: {config.enrichment.model}"
        + (f" | digest={model_digest[:12]}" if model_digest else " | digest=unavailable")
    )
    print(
        f"[+] Pipeline: {config.enrichment.prompt_version} | "
        f"{config.enrichment.strategy}"
    )
    print(f"[+] Obras no manifest: {len(manifest)}")
    print(f"[+] Obras nesta execução: {len(execution_rows)}")

    if not execution_rows:
        print("[+] Nenhuma obra pendente.")

    service = WorkEnrichmentService(config)
    successes_run = 0
    errors_run = 0

    try:
        with tqdm(
            total=len(execution_rows),
            desc="Enrichment · obras",
            unit="obra",
            dynamic_ncols=True,
        ) as progress:
            for index in execution_rows:
                row = manifest.loc[index]
                title = str(row["titulo"])
                progress.set_postfix(
                    obra=title[:32],
                    ok=successes_run,
                    erro=errors_run,
                    refresh=True,
                )

                work_started = perf_counter()
                manifest.at[index, "tentativas_enrichment"] = int(manifest.at[index, "tentativas_enrichment"]) + 1
                before_usage = service.provider.usage_snapshot() if service.provider else {}
                batch_paused = False

                def record_usage():
                    usage = service.provider.usage_snapshot() if service.provider else {}
                    for name, column in {"calls": "llm_calls", "input_tokens": "llm_input_tokens", "output_tokens": "llm_output_tokens"}.items():
                        manifest.at[index, column] = int(manifest.at[index, column]) + int(usage.get(name, 0) - before_usage.get(name, 0))

                try:
                    bronze_object_key = str(row["source_bronze_object_key"])
                    text = storage.download_bytes(
                        bucket=bronze_bucket,
                        key=bronze_object_key,
                    ).decode("utf-8", errors="replace")
                    if not text.strip():
                        raise RuntimeError("Texto Bronze vazio.")

                    source_text_sha = hashlib.sha256(
                        text.encode("utf-8")
                    ).hexdigest()
                    source = WorkSource(
                        title=title,
                        author=str(row.get("autor", "") or "Machado de Assis"),
                        category=str(row.get("categoria", "") or ""),
                        source_document_id=str(row["source_document_id"]),
                        bronze_artifact_id=bronze_artifact_id,
                        bronze_object_key=bronze_object_key,
                        text=text,
                    )
                    generator_signature = hashlib.sha256(json.dumps({
                        "enrichment": config.enrichment.model_dump(exclude=ENRICHMENT_RUNTIME_FIELDS), "prompt": prompt_hash, "schema": schema_hash,
                    }, sort_keys=True).encode()).hexdigest()
                    result = service.enrich(source, checkpoint_prefix=f"checkpoints/{generator_signature}/{source_text_sha}")

                    work_payload = {
                        "enrichment_dataset_id": dataset_id,
                        "config_hash": config_hash,
                        "usage": result.usage,
                        "created_at_utc": utc_now(),
                        "source": {
                            "bronze_artifact_id": bronze_artifact_id,
                            "source_document_id": source.source_document_id,
                            "titulo": source.title,
                            "autor": source.author,
                            "categoria": source.category,
                            "bronze_bucket": bronze_bucket,
                            "bronze_object_key": bronze_object_key,
                            "text_sha256": source_text_sha,
                            "chars": len(text),
                            "words": len(text.split()),
                        },
                        "generation": {
                            "model": config.enrichment.model,
                            "model_digest": model_digest,
                            "prompt_version": config.enrichment.prompt_version,
                            "prompt_sha256": prompt_hash,
                            "schema_version": config.enrichment.schema_version,
                            "schema_sha256": schema_hash,
                            "max_tokens": config.enrichment.max_tokens,
                            "response_retries": config.enrichment.response_retries,
                            "response_retry_max_tokens": config.enrichment.response_retry_max_tokens,
                        },
                        "plan": result.plan,
                        "blocks": result.blocks,
                        "summary": result.summary,
                        "metrics": {
                            "timings_seconds": result.timings_seconds,
                            "reduce_calls": result.reduce_calls,
                        },
                    }
                    work_data = json_bytes(work_payload)
                    object_key = str(row["enrichment_object_key"])

                    # Primeiro o objeto da obra, depois o manifest.
                    storage.upload_bytes(
                        bucket=enrichment_bucket,
                        key=object_key,
                        data=work_data,
                        content_type="application/json",
                    )
                    work_sha = sha256_bytes(work_data)

                    manifest.at[index, "effective_strategy"] = str(
                        result.plan["effective_strategy_from_chunker"]
                    )
                    manifest.at[index, "plan_mode"] = str(result.plan["plan_mode"])
                    manifest.at[index, "blocks"] = int(result.plan["planned_units"])
                    manifest.at[index, "source_chars"] = int(len(text))
                    manifest.at[index, "output_json_bytes"] = int(len(work_data))
                    manifest.at[index, "elapsed_seconds"] = float(
                        result.timings_seconds["total"]
                    )
                    manifest.at[index, "reduce_calls"] = int(result.reduce_calls)
                    manifest.at[index, "source_text_sha256"] = source_text_sha
                    manifest.at[index, "work_sha256"] = work_sha
                    manifest.at[index, "status_enrichment"] = "success"
                    manifest.at[index, "processed_at_utc"] = utc_now()
                    manifest.at[index, "erro_enrichment"] = ""
                    manifest.at[index, "erro_tipo"] = ""
                    manifest.at[index, "erro_detalhes_json"] = ""
                    successes_run += 1

                except Exception as exc:
                    manifest.at[index, "elapsed_seconds"] = round(
                        perf_counter() - work_started,
                        4,
                    )
                    manifest.at[index, "status_enrichment"] = "error"
                    manifest.at[index, "processed_at_utc"] = utc_now()
                    manifest.at[index, "erro_enrichment"] = str(exc)[:2000]
                    details = getattr(exc, "diagnostics", {})
                    manifest.at[index, "erro_tipo"] = details.get("reason", type(exc).__name__)
                    manifest.at[index, "erro_detalhes_json"] = json.dumps(details, ensure_ascii=False)
                    record_usage()
                    errors_run += 1

                    sync_manifest_and_artifact(
                        storage=storage,
                        bucket=enrichment_bucket,
                        manifest_key=manifest_key,
                        artifact_key=artifact_key,
                        manifest=manifest,
                        dataset_id=dataset_id,
                        config_hash=config_hash,
                        signature=signature,
                        config=config,
                        bronze_current=bronze_current,
                        bronze_manifest_sha256=bronze_manifest_sha256,
                        dataset_prefix=dataset_prefix,
                        status="running",
                        created_at_utc=created_at_utc,
                    )

                    if args.fail_fast:
                        raise
                    if isinstance(exc, GenerationServiceError) and exc.batch_fatal:
                        print(f"[pausa] Serviço indisponível: {manifest.at[index, 'erro_tipo']}; manifest salvo, demais obras preservadas como pendentes.", flush=True)
                        batch_paused = True

                else:
                    record_usage()
                    # Persistência imediata após cada obra concluída.
                    sync_manifest_and_artifact(
                        storage=storage,
                        bucket=enrichment_bucket,
                        manifest_key=manifest_key,
                        artifact_key=artifact_key,
                        manifest=manifest,
                        dataset_id=dataset_id,
                        config_hash=config_hash,
                        signature=signature,
                        config=config,
                        bronze_current=bronze_current,
                        bronze_manifest_sha256=bronze_manifest_sha256,
                        dataset_prefix=dataset_prefix,
                        status="running",
                        created_at_utc=created_at_utc,
                    )

                progress.update(1)
                progress.set_postfix(
                    obra=title[:32],
                    ok=successes_run,
                    erro=errors_run,
                    refresh=True,
                )
                if batch_paused:
                    break

    finally:
        service.unload_model()

    final_status = final_status_from_manifest(manifest)
    manifest_sha, artifact_sha = sync_manifest_and_artifact(
        storage=storage,
        bucket=enrichment_bucket,
        manifest_key=manifest_key,
        artifact_key=artifact_key,
        manifest=manifest,
        dataset_id=dataset_id,
        config_hash=config_hash,
        signature=signature,
        config=config,
        bronze_current=bronze_current,
        bronze_manifest_sha256=bronze_manifest_sha256,
        dataset_prefix=dataset_prefix,
        status=final_status,
        created_at_utc=created_at_utc,
    )

    # Segue o padrão Bronze: current só muda após artifact completo.
    if final_status == "complete":
        upload_json(
            storage,
            bucket=enrichment_bucket,
            key=ENRICHMENT_CURRENT_KEY,
            payload={
                "enrichment_dataset_id": dataset_id,
                "manifest_key": manifest_key,
                "artifact_key": artifact_key,
                "manifest_sha256": manifest_sha,
                "artifact_sha256": artifact_sha,
                "source_bronze_artifact_id": bronze_artifact_id,
                "updated_at_utc": utc_now(),
            },
        )

    counts = manifest["status_enrichment"].value_counts().to_dict()
    print()
    print("[OK] Enrichment batch finalizado.")
    print(f"     Dataset: {dataset_id}")
    print(f"     Status: {final_status}")
    print(f"     Bucket: s3://{enrichment_bucket}/{dataset_prefix}/")
    print(f"     Manifest: s3://{enrichment_bucket}/{manifest_key}")
    if final_status == "complete":
        print(f"     Current: s3://{enrichment_bucket}/{ENRICHMENT_CURRENT_KEY}")
    else:
        print("     Current: não atualizado (artifact ainda não está completo)")
    print()
    print(f"[+] Success: {int(counts.get('success', 0))}")
    print(f"[+] Errors:  {int(counts.get('error', 0))}")
    print(f"[+] Pending: {int(counts.get('pending', 0))}")
    if final_status != "complete":
        print("[pendente] Manifest salvo. Retome com --retry; índice auxiliar e promoção aguardam todas as obras.", flush=True)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
