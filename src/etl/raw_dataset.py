"""Contrato do Raw coletado no notebook e consumido pelo Glue."""

from contextlib import contextmanager
import hashlib
import io
import json
import re

import pandas as pd
from botocore.exceptions import ClientError


def raw_prefix(raw_id):
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,100}", raw_id):
        raise ValueError(
            "raw-id inválido; use letras, números, _ e -, até 100 caracteres."
        )
    return f"production/raw/{raw_id}"


@contextmanager
def crawl_lock(storage, bucket, raw_id):
    key = raw_prefix(raw_id) + "/crawl.lock"
    try:
        storage.client.put_object(
            Bucket=bucket, Key=key, Body=raw_id.encode(), IfNoneMatch="*"
        )
    except ClientError as exc:
        if exc.response["Error"]["Code"] in {
            "PreconditionFailed",
            "412",
            "ConditionalRequestConflict",
        }:
            raise RuntimeError(
                "Este Raw já está em coleta ou tem lock de uma interrupção. Confira o processo antes de liberar o lock."
            ) from exc
        raise
    try:
        yield
    finally:
        storage.delete_object(bucket, key)


def pdf_metadata(storage, bucket, key):
    response = storage.client.get_object(Bucket=bucket, Key=key)
    digest, length, signature = hashlib.sha256(), 0, b""
    body = response["Body"]
    try:
        for chunk in body.iter_chunks(chunk_size=1024 * 1024):
            signature = (signature + chunk)[:5]
            digest.update(chunk)
            length += len(chunk)
    finally:
        body.close()
    if signature != b"%PDF-":
        raise ValueError(f"Objeto Raw não é PDF: {key}.")
    return digest.hexdigest(), length


def validate_raw_manifest(storage, bucket, raw_id, payload, allow_unavailable=False):
    """Verifica cada PDF no S3; não considera um HTTP 200 HTML como documento."""
    frame = pd.read_parquet(io.BytesIO(payload))
    required = {
        "document_id",
        "titulo",
        "url_detalhe",
        "raw_bucket",
        "raw_object_key",
        "sha256",
        "size_bytes",
        "status_ingestao",
    }
    if frame.empty or not required.issubset(frame.columns):
        raise ValueError("Manifest Raw vazio ou sem as colunas obrigatórias.")
    identity = required - {"sha256", "size_bytes"}
    if frame[list(identity)].isna().any().any():
        raise ValueError("Manifest Raw contém metadados ausentes.")
    for column in ("document_id", "url_detalhe", "raw_object_key"):
        if frame[column].duplicated().any():
            raise ValueError(f"Manifest Raw contém valores duplicados: {column}.")
    allowed = ["uploaded", "existing", *(["unavailable"] if allow_unavailable else [])]
    if not frame["status_ingestao"].isin(allowed).all():
        raise ValueError("Raw incompleto: há obras sem download ou com erro.")
    available = frame["status_ingestao"].isin(["uploaded", "existing"])
    if not available.any():
        raise ValueError("Raw sem PDFs disponíveis para processamento.")
    if frame.loc[available, ["sha256", "size_bytes"]].isna().any().any():
        raise ValueError("Manifest Raw contém metadados ausentes nos PDFs disponíveis.")
    prefix = raw_prefix(raw_id) + "/documents/"
    for row in frame.to_dict("records"):
        key = row["raw_object_key"]
        if (
            row["raw_bucket"] != bucket
            or not isinstance(key, str)
            or not key.startswith(prefix)
            or ".." in key.split("/")
        ):
            raise ValueError("Documento Raw fora do bucket ou prefixo selecionado.")
        if row["status_ingestao"] == "unavailable":
            for name in ("unavailable_reason", "unavailable_at_utc"):
                if not isinstance(row.get(name), str) or not row[name].strip():
                    raise ValueError(
                        "Indisponibilidade sem motivo ou data de confirmação."
                    )
            continue
        if not re.fullmatch(r"[0-9a-f]{64}", str(row["sha256"])):
            raise ValueError("Documento Raw sem SHA-256 válido.")
        checksum, length = pdf_metadata(storage, bucket, key)
        if length != int(row["size_bytes"]) or checksum != row["sha256"]:
            raise ValueError(f"PDF ausente, inválido ou alterado: {key}.")
    return int(available.sum())


def load_raw_dataset(storage, bucket, raw_id):
    prefix = raw_prefix(raw_id)
    if not storage.object_exists(bucket, prefix + "/ready.json"):
        raise ValueError(
            "Raw ainda não está pronto. Conclua o crawler no notebook antes de iniciar o Glue."
        )
    receipt = json.loads(storage.download_bytes(bucket, prefix + "/ready.json"))
    version = receipt.get("version")
    if version not in {1, 2} or (
        receipt.get("status"),
        receipt.get("raw_id"),
        receipt.get("raw_bucket"),
        receipt.get("manifest_key"),
    ) != ("ready", raw_id, bucket, prefix + "/manifest.parquet"):
        raise ValueError("Recibo Raw inválido ou de outro destino.")
    request_fields = ("raw_id", "source_query", "first_page", "last_page")
    if any(name not in receipt for name in request_fields):
        raise ValueError("Recibo Raw sem a consulta ou o intervalo de páginas.")
    request = json.loads(storage.download_bytes(bucket, prefix + "/request.json"))
    if request != {name: receipt[name] for name in request_fields}:
        raise ValueError("Recibo Raw diverge da consulta original.")
    payload = storage.download_bytes(bucket, receipt["manifest_key"])
    if hashlib.sha256(payload).hexdigest() != receipt.get("manifest_sha256"):
        raise ValueError("Manifest Raw diverge do SHA-256 publicado.")
    count = validate_raw_manifest(
        storage, bucket, raw_id, payload, allow_unavailable=version == 2
    )
    if count != receipt.get("documents"):
        raise ValueError("Contagem Raw diverge do recibo publicado.")
    if version == 2:
        frame = pd.read_parquet(io.BytesIO(payload))
        excluded = sorted(
            frame.loc[frame["status_ingestao"].eq("unavailable"), "document_id"].astype(
                str
            )
        )
        if (
            receipt.get("catalog_records"),
            receipt.get("unavailable"),
            receipt.get("excluded_document_ids"),
        ) != (len(frame), len(excluded), excluded):
            raise ValueError("Exclusões do Raw divergem do manifest.")
    return receipt, payload
