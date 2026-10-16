"""Raw no notebook, cinco etapas no Glue; sem portal, modelos ou AWS reais."""

import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import zipfile

import boto3
from moto import mock_aws
import pandas as pd
import pytest
import requests

from src.config.config_manager import ConfigManager
from src.etl import crawler_ingestao_raw_rustfs as crawler
from src.etl import production
from src.etl.raw_dataset import load_raw_dataset, raw_prefix
from src.storage.rustfs_manager import RustFSManager

ROOT = Path(__file__).resolve().parents[1]
RAW = "crawler-raw"
BUCKETS = {
    "RAW_BUCKET": RAW,
    "BRONZE_BUCKET": "crawler-bronze",
    "SILVER_BUCKET": "crawler-silver",
    "EMBEDDING_INDEXES_BUCKET": "crawler-indexes",
    "ENRICHMENT_BUCKET": "crawler-enrichment",
    "BENCHMARKS_BUCKET": "crawler-benchmarks",
}
PDF = b"%PDF-1.4\nfixture\n%%EOF\n"


@pytest.fixture
def aws(monkeypatch):
    original = dict(os.environ)
    for key in list(os.environ):
        if key.startswith(
            ("AWS_", "RUSTFS_", "QDRANT_", "DATA_RELEASE", "LLM_", "ENRICHMENT_")
        ) or key in {
            "STORAGE_PROVIDER",
            "CONFIG_FILE",
            "ENV_FILE",
            "BRONZE_CURRENT_KEY",
            "SILVER_CURRENT_KEY",
            "RAW_MANIFEST_KEY",
            "COLIBRI_REVISION",
        }:
            monkeypatch.delenv(key, raising=False)
    for key, value in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_DEFAULT_REGION": "us-east-1",
        "AWS_EC2_METADATA_DISABLED": "true",
    }.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(crawler.time, "sleep", lambda _: None)
    with mock_aws():
        storage = RustFSManager()
        for bucket in [*BUCKETS.values(), "crawler-ops"]:
            storage.create_bucket(bucket)
        try:
            yield storage
        finally:
            os.environ.clear()
            os.environ.update(original)


def book(number=1):
    return {
        "titulo": f"Obra {number}",
        "autor": "Machado de Assis",
        "fonte": "MEC",
        "formato": "pdf",
        "url_detalhe": f"{crawler.BASE_URL}pesquisa/DetalheObraForm.do?co_obra={number}",
        "url_download": f"{crawler.BASE_URL}download/obra-{number}.pdf",
        "pagina_origem": 1,
    }


def fake_source(monkeypatch, rows=None):
    rows = rows if rows is not None else [book()]
    pages = Mock(return_value=rows)
    downloads = Mock()

    def download(url, destino):
        downloads(url)
        destino.write_bytes(PDF)
        return len(PDF), "application/pdf"

    monkeypatch.setattr(crawler, "get_obras_machado", pages)
    monkeypatch.setattr(crawler, "baixar_pdf_temporario", download)
    return pages, downloads


def crawl(monkeypatch, raw_id="machado-v1", last_page=1, extra=()):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "crawler",
            "--raw-bucket",
            RAW,
            "--raw-id",
            raw_id,
            "--pagina-final",
            str(last_page),
            "--delay",
            "0",
            *extra,
        ],
    )
    crawler.main()


def ready(aws, monkeypatch, raw_id="machado-v1"):
    fake_source(monkeypatch)
    crawl(monkeypatch, raw_id)
    return load_raw_dataset(aws, RAW, raw_id)


def script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_local_crawler_writes_only_raw_without_loading_services(aws, monkeypatch):
    monkeypatch.setattr(
        ConfigManager,
        "load",
        Mock(side_effect=AssertionError("não carregar .env/serviços")),
    )
    aws.upload_bytes(RAW, "manifests/obras_raw.parquet", b"old-failed-cloud-manifest")
    pages, downloads = fake_source(monkeypatch, [book(1), book(1), book(2)])
    crawl(monkeypatch)
    receipt, payload = load_raw_dataset(aws, RAW, "machado-v1")
    assert receipt["documents"] == 2
    assert receipt["first_page"] == receipt["last_page"] == 1
    assert receipt["manifest_sha256"] == hashlib.sha256(payload).hexdigest()
    frame = pd.read_parquet(io.BytesIO(payload))
    assert list(frame.document_id) == ["000001", "000002"]
    assert downloads.call_count == 2
    assert all(
        "production/raw/machado-v1/documents/" in key for key in frame.raw_object_key
    )
    assert (
        aws.download_bytes(RAW, "manifests/obras_raw.parquet")
        == b"old-failed-cloud-manifest"
    )
    assert all(
        not aws.list_objects(bucket)
        for name, bucket in BUCKETS.items()
        if name != "RAW_BUCKET"
    )
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/crawl.lock")
    before = aws.download_bytes(RAW, receipt["manifest_key"])
    pages.side_effect = AssertionError("não consultar portal novamente")
    crawl(monkeypatch)
    assert downloads.call_count == 2
    assert aws.download_bytes(RAW, receipt["manifest_key"]) == before


def test_page_403_saves_partial_but_glue_rejects_until_resume(aws, monkeypatch):
    pages, downloads = fake_source(monkeypatch)
    pages.side_effect = [[book()], requests.HTTPError("403 Forbidden")]
    with pytest.raises(SystemExit) as error:
        crawl(monkeypatch, last_page=2)
    assert error.value.code == 2
    assert aws.object_exists(RAW, raw_prefix("machado-v1") + "/manifest.parquet")
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/ready.json")
    with pytest.raises(ValueError, match="ainda não está pronto"):
        load_raw_dataset(aws, RAW, "machado-v1")
    pages.side_effect = [[book()], [book(2)]]
    crawl(monkeypatch, last_page=2)
    receipt, _ = load_raw_dataset(aws, RAW, "machado-v1")
    assert receipt["documents"] == 2
    assert downloads.call_count == 2  # primeiro PDF não foi baixado de novo


@pytest.mark.parametrize(
    "failure", ["empty", "missing-link", "invalid-pdf", "detail-403"]
)
def test_incomplete_crawl_never_publishes_ready(aws, monkeypatch, failure):
    if failure == "detail-403":
        html = '<table><tr><td><a href="DetalheObraForm.do?co_obra=1">Obra</a></td><td>Machado</td><td>MEC</td><td>pdf</td><td>1 KB</td><td>10</td></tr></table>'
        response = Mock(text=html)
        response.raise_for_status.return_value = None
        monkeypatch.setattr(crawler.session, "get", Mock(return_value=response))
        monkeypatch.setattr(
            crawler,
            "get_download_links",
            Mock(side_effect=requests.HTTPError("403 detalhe")),
        )
    else:
        row = book()
        if failure == "missing-link":
            row["url_download"] = None
        fake_source(monkeypatch, [] if failure == "empty" else [row])
        if failure == "invalid-pdf":
            monkeypatch.setattr(
                crawler,
                "baixar_pdf_temporario",
                Mock(side_effect=ValueError("HTML não é PDF")),
            )
    with pytest.raises(SystemExit) as error:
        crawl(monkeypatch)
    assert error.value.code == 2
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/ready.json")
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/crawl.lock")


def test_http_parser_and_download_use_existing_crawler(aws, monkeypatch):
    class Response:
        headers = {"Content-Type": "application/pdf"}

        def __init__(self, text=""):
            self.text = text

        def raise_for_status(self):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def iter_content(self, **kwargs):
            return iter([PDF])

    def get(url, **kwargs):
        if "ResultadoPesquisaObraForm" in url:
            return Response(
                '<table><tr><td><a href="DetalheObraForm.do?co_obra=1">A Cartomante</a></td><td>Machado</td><td>MEC</td><td>pdf</td><td>1 KB</td><td>10</td></tr></table>'
            )
        if "DetalheObraForm" in url:
            return Response('<a href="/download/cartomante.pdf">Baixar</a>')
        assert url.endswith("/download/cartomante.pdf")
        return Response()

    monkeypatch.setattr(crawler.session, "get", get)
    crawl(monkeypatch)
    receipt, payload = load_raw_dataset(aws, RAW, "machado-v1")
    assert receipt["documents"] == 1
    assert pd.read_parquet(io.BytesIO(payload)).iloc[0]["titulo"] == "A Cartomante"


def test_lock_and_changed_page_range_preserve_existing_objects(aws, monkeypatch):
    fake_source(monkeypatch)
    key = raw_prefix("machado-v1") + "/crawl.lock"
    aws.upload_bytes(RAW, key, b"other-process")
    with pytest.raises(RuntimeError, match="lock"):
        crawl(monkeypatch)
    assert aws.download_bytes(RAW, key) == b"other-process"
    aws.delete_object(RAW, key)
    ready(aws, monkeypatch)
    old = aws.download_bytes(RAW, raw_prefix("machado-v1") + "/ready.json")
    with pytest.raises(ValueError, match="outra consulta ou intervalo"):
        crawl(monkeypatch, last_page=2)
    assert aws.download_bytes(RAW, raw_prefix("machado-v1") + "/ready.json") == old


@pytest.mark.parametrize(
    "tamper", ["manifest", "pdf", "receipt", "missing-pdf", "bucket"]
)
def test_glue_checks_raw_receipt_manifest_and_pdf_integrity(aws, monkeypatch, tamper):
    receipt, payload = ready(aws, monkeypatch)
    if tamper == "manifest":
        aws.upload_bytes(RAW, receipt["manifest_key"], b"changed")
    elif tamper in {"pdf", "missing-pdf"}:
        key = pd.read_parquet(io.BytesIO(payload)).iloc[0]["raw_object_key"]
        if tamper == "pdf":
            aws.upload_bytes(RAW, key, b"%PDF-changed")
        else:
            aws.delete_object(RAW, key)
    elif tamper == "receipt":
        receipt["raw_bucket"] = "other-bucket"
        aws.upload_bytes(
            RAW, raw_prefix("machado-v1") + "/ready.json", json.dumps(receipt).encode()
        )
    with pytest.raises((ValueError, aws.client.exceptions.NoSuchKey)):
        load_raw_dataset(
            aws, BUCKETS["BRONZE_BUCKET"] if tamper == "bucket" else RAW, "machado-v1"
        )


def test_resume_recovers_pdf_uploaded_before_manifest_checkpoint(aws, monkeypatch):
    _, downloads = fake_source(monkeypatch)
    url = book()["url_detalhe"]
    key = (
        raw_prefix("machado-v1")
        + "/documents/"
        + hashlib.sha256(url.encode()).hexdigest()
        + ".pdf"
    )
    aws.upload_bytes(RAW, key, PDF)
    crawl(monkeypatch)
    receipt, _ = load_raw_dataset(aws, RAW, "machado-v1")
    assert receipt["documents"] == 1
    downloads.assert_not_called()


@pytest.mark.parametrize(
    "extra",
    [
        ["--raw-id", "../bad"],
        ["--delay", "nan"],
        ["--config-file", "config/config.yaml"],
    ],
)
def test_invalid_local_arguments_do_not_write_raw(aws, monkeypatch, extra):
    fake_source(monkeypatch)
    with pytest.raises((SystemExit, ValueError)):
        crawl(monkeypatch, extra=extra)
    assert not aws.list_objects(RAW)


def production_config(monkeypatch):
    for key, value in {
        **BUCKETS,
        "STORAGE_PROVIDER": "s3",
        "QDRANT_URL": "https://cluster.example.org",
        "QDRANT_API_KEY": "testing",
    }.items():
        monkeypatch.setenv(key, value)
    return ConfigManager.load(
        env_file=ROOT / "absent.env", yaml_file=ROOT / "config/config.aws.yaml"
    )


def partial_corpus(aws, monkeypatch, legacy=False):
    fake_source(monkeypatch, [book(number) for number in range(1, 5)])

    def download(url, destino):
        if not url.endswith("obra-1.pdf"):
            raise ValueError("Conteúdo baixado não possui assinatura de arquivo PDF.")
        destino.write_bytes(PDF)
        return len(PDF), "application/pdf"

    monkeypatch.setattr(crawler, "baixar_pdf_temporario", download)
    with pytest.raises(SystemExit) as error:
        crawl(monkeypatch)
    assert error.value.code == 2
    key = raw_prefix("machado-v1") + "/manifest.parquet"
    if legacy:
        frame = pd.read_parquet(io.BytesIO(aws.download_bytes(RAW, key)))
        frame = frame.drop(columns=["attempt_count", "last_attempt_at_utc"])
        aws.upload_bytes(RAW, key, frame.to_parquet(index=False))
        aws.delete_object(RAW, raw_prefix("machado-v1") + "/crawl-status.json")
    return key


@pytest.mark.parametrize("legacy", [False, True])
def test_retry_downloads_only_three_pending_records_and_keeps_success(
    aws, monkeypatch, legacy
):
    key = partial_corpus(aws, monkeypatch, legacy)
    before = pd.read_parquet(io.BytesIO(aws.download_bytes(RAW, key))).iloc[0].to_dict()
    listing = Mock(return_value=[book(number) for number in range(1, 5)])
    if not legacy:
        listing.side_effect = AssertionError("paginação já confirmada")
    monkeypatch.setattr(crawler, "get_obras_machado", listing)
    details = Mock(
        side_effect=lambda url: [book(int(url.rsplit("=", 1)[-1]))["url_download"]]
    )
    monkeypatch.setattr(crawler, "get_download_links", details)
    downloads = []

    def recovered(url, destino):
        downloads.append(url)
        destino.write_bytes(PDF)
        return len(PDF), "application/pdf"

    monkeypatch.setattr(crawler, "baixar_pdf_temporario", recovered)
    crawl(monkeypatch, extra=["--retry"])
    assert downloads == [book(number)["url_download"] for number in (2, 3, 4)]
    assert details.call_count == 3
    if legacy:
        listing.assert_called_once_with(1, fetch_downloads=False)
    else:
        listing.assert_not_called()
    receipt, payload = load_raw_dataset(aws, RAW, "machado-v1")
    after = pd.read_parquet(io.BytesIO(payload))
    assert receipt["documents"] == 4
    assert list(after.document_id) == ["000001", "000002", "000003", "000004"]
    assert after.iloc[0]["sha256"] == before["sha256"]
    assert after.iloc[0]["status_ingestao"] == "uploaded"
    assert aws.download_bytes(RAW, before["raw_object_key"]) == PDF
    assert list(after.iloc[1:]["attempt_count"]) == ([1, 1, 1] if legacy else [2, 2, 2])
    assert after.iloc[1:]["last_attempt_at_utc"].notna().all()


def test_retry_persists_remaining_failures_and_exits_without_traceback(
    aws, monkeypatch, capsys
):
    key = partial_corpus(aws, monkeypatch)
    monkeypatch.setattr(crawler, "get_download_links", Mock(return_value=[]))
    for _ in range(2):
        with pytest.raises(SystemExit) as error:
            crawl(monkeypatch, extra=["--retry"])
        assert error.value.code == 2
    frame = pd.read_parquet(io.BytesIO(aws.download_bytes(RAW, key)))
    assert list(frame.iloc[1:]["attempt_count"]) == [3, 3, 3]
    assert frame.iloc[1:]["status_ingestao"].eq("erro").all()
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/ready.json")
    output = capsys.readouterr()
    assert "Pendências: 3" in output.out
    assert "Traceback" not in output.out + output.err


@pytest.mark.parametrize("failure", ["sem_download", "detail-403"])
def test_retry_refreshes_links_and_persists_detail_failure(aws, monkeypatch, failure):
    key = partial_corpus(aws, monkeypatch)
    frame = pd.read_parquet(io.BytesIO(aws.download_bytes(RAW, key)))
    if failure == "sem_download":
        frame.loc[frame.document_id.eq("000002"), "status_ingestao"] = "sem_download"
        frame.loc[frame.document_id.eq("000002"), "url_download"] = None
        aws.upload_bytes(RAW, key, frame.to_parquet(index=False))
    details = (
        Mock(side_effect=requests.HTTPError("403 detalhe"))
        if failure == "detail-403"
        else Mock(return_value=[])
    )
    monkeypatch.setattr(crawler, "get_download_links", details)
    with pytest.raises(SystemExit) as error:
        crawl(monkeypatch, extra=["--retry"])
    assert error.value.code == 2
    after = pd.read_parquet(io.BytesIO(aws.download_bytes(RAW, key)))
    assert list(after.iloc[1:].attempt_count) == [2, 2, 2]
    if failure == "detail-403":
        assert after.iloc[1:].erro_ingestao.str.contains("403 detalhe").all()
    else:
        assert after.iloc[1].status_ingestao == "sem_download"


def test_confirmed_unavailable_records_are_audited_and_bronze_uses_only_pdf(
    aws, monkeypatch
):
    # Exercita a seleção real da Bronze sem executar o parser de PDFs.
    with monkeypatch.context() as patch:
        patch.setitem(sys.modules, "pymupdf", ModuleType("pymupdf"))
        bronze = script(
            "retry_bronze_selection", "src/etl/preparar_camada_bronze_rustfs.py"
        )
    key = partial_corpus(aws, monkeypatch, legacy=True)
    pages = Mock(return_value=[book(number) for number in range(1, 5)])
    details = Mock(side_effect=AssertionError("não consultar detalhes"))
    downloads = Mock(side_effect=AssertionError("não baixar PDFs"))
    monkeypatch.setattr(crawler, "get_obras_machado", pages)
    monkeypatch.setattr(crawler, "get_download_links", details)
    monkeypatch.setattr(crawler, "baixar_pdf_temporario", downloads)
    crawl(
        monkeypatch,
        extra=[
            "--mark-unavailable",
            "2",
            "000003",
            "000004",
            "--reason",
            "Fonte conferida manualmente",
        ],
    )
    pages.assert_called_once_with(1, fetch_downloads=False)
    details.assert_not_called()
    downloads.assert_not_called()
    receipt, payload = load_raw_dataset(aws, RAW, "machado-v1")
    assert receipt["version"] == 2
    assert (
        receipt["documents"],
        receipt["catalog_records"],
        receipt["unavailable"],
    ) == (1, 4, 3)
    assert receipt["excluded_document_ids"] == ["000002", "000003", "000004"]
    frame = pd.read_parquet(io.BytesIO(payload))
    excluded = frame[frame.status_ingestao.eq("unavailable")]
    assert excluded.unavailable_reason.eq("Fonte conferida manualmente").all()
    assert excluded.unavailable_at_utc.notna().all()
    assert excluded.erro_ingestao.str.contains("assinatura").all()
    assert set(bronze.selecionar_versoes_canonicas(frame)["document_id"]) == {"000001"}
    before = aws.download_bytes(RAW, key)
    pages.side_effect = AssertionError("não refazer catálogo concluído")
    crawl(monkeypatch, extra=["--retry"])
    assert aws.download_bytes(RAW, key) == before
    # Uma exclusão sem justificativa não pode passar pela validação do Glue.
    changed = frame.copy()
    changed.loc[changed.status_ingestao.eq("unavailable"), "unavailable_reason"] = None
    data = changed.to_parquet(index=False)
    aws.upload_bytes(RAW, key, data)
    receipt["manifest_sha256"] = hashlib.sha256(data).hexdigest()
    aws.upload_bytes(
        RAW, raw_prefix("machado-v1") + "/ready.json", json.dumps(receipt).encode()
    )
    with pytest.raises(ValueError, match="sem motivo"):
        load_raw_dataset(aws, RAW, "machado-v1")


@pytest.mark.parametrize("ids", [["1"], ["2", "999"], ["invalido"]])
def test_unavailable_rejects_good_pdf_and_unknown_ids_without_mutating_manifest(
    aws, monkeypatch, ids
):
    key = partial_corpus(aws, monkeypatch)
    before = aws.download_bytes(RAW, key)
    with pytest.raises(SystemExit) as error:
        crawl(monkeypatch, extra=["--mark-unavailable", *ids, "--reason", "Conferido"])
    assert error.value.code == 2
    assert aws.download_bytes(RAW, key) == before
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/ready.json")


def test_retry_of_legacy_manifest_cannot_publish_if_listing_does_not_match(
    aws, monkeypatch
):
    partial_corpus(aws, monkeypatch, legacy=True)
    monkeypatch.setattr(crawler, "get_obras_machado", Mock(return_value=[book(1)]))
    monkeypatch.setattr(crawler, "get_download_links", Mock(return_value=[]))
    with pytest.raises(SystemExit) as error:
        crawl(
            monkeypatch,
            extra=["--mark-unavailable", "2", "3", "4", "--reason", "Conferido"],
        )
    assert error.value.code == 2
    assert not aws.object_exists(RAW, raw_prefix("machado-v1") + "/ready.json")


def test_listing_only_mode_does_not_request_download_details(monkeypatch):
    html = '<table><tr><td><a href="DetalheObraForm.do?co_obra=1">Obra</a></td><td>Machado</td><td>MEC</td><td>pdf</td><td>1 KB</td><td>10</td></tr></table>'
    response = Mock(text=html)
    response.raise_for_status.return_value = None
    request = Mock(return_value=response)
    details = Mock(side_effect=AssertionError("não consultar detalhe"))
    monkeypatch.setattr(crawler.session, "get", request)
    monkeypatch.setattr(crawler, "get_download_links", details)
    rows = crawler.get_obras_machado(1, fetch_downloads=False)
    assert len(rows) == 1 and rows[0]["url_download"] is None
    request.assert_called_once()
    details.assert_not_called()


@pytest.mark.parametrize("with_exclusions", [False, True])
def test_glue_runs_all_five_stages_resumes_and_promotes_same_raw(
    aws, monkeypatch, with_exclusions
):
    if with_exclusions:
        partial_corpus(aws, monkeypatch)
        crawl(
            monkeypatch,
            extra=[
                "--mark-unavailable",
                "2",
                "3",
                "4",
                "--reason",
                "Conferido manualmente",
            ],
        )
        receipt, payload = load_raw_dataset(aws, RAW, "machado-v1")
        assert receipt["version"] == 2
    else:
        receipt, payload = ready(aws, monkeypatch)
    conf = production_config(monkeypatch)
    monkeypatch.setattr(production.ConfigManager, "load", lambda: conf)
    monkeypatch.setattr("src.generation.provider.create_generation_provider", Mock())
    monkeypatch.setenv("SOURCE_REVISION", "a" * 40)
    import src.storage.qdrant_manager as qdrant_module

    qdrant = Mock()
    qdrant.count.return_value = 1
    monkeypatch.setattr(qdrant_module.QdrantManager, "from_config", lambda _: qdrant)
    executions = []

    def write(bucket, key, value):
        production.write_json(aws, bucket, key, value)

    class Index:
        def __init__(self, *args):
            pass

        def build(self, silver_id, alias, **kwargs):
            executions.append("text-index")
            assert (silver_id, alias) == ("s1", "colibri")
            manifest = {
                "status": "success",
                "index_id": "text",
                "source": {"silver_dataset_id": "s1"},
                "qdrant": {"collection": "text"},
                "validation": {"silver_chunks": 1},
            }
            write(BUCKETS["EMBEDDING_INDEXES_BUCKET"], "text/manifest.json", manifest)
            return manifest

    embeddings = ModuleType("src.embeddings")
    embeddings.__path__ = []
    registry = ModuleType("src.embeddings.registry")
    registry.EmbedderRegistry = Mock()
    indexing = ModuleType("src.indexing.embedding_index")
    indexing.EmbeddingIndex = Index
    for name, module in {
        "src.embeddings": embeddings,
        "src.embeddings.registry": registry,
        "src.indexing.embedding_index": indexing,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    failed = False

    def run_stage(name, arguments=()):
        nonlocal failed
        assert "crawler" not in name
        if "preparar_camada_bronze" in name:
            executions.append("bronze")
            assert aws.download_bytes(RAW, conf.storage.raw_manifest_key) == payload
            write(
                conf.storage.bronze_bucket,
                conf.storage.bronze_current_key,
                {"bronze_artifact_id": "b1"},
            )
        elif "preparar_camada_silver" in name:
            executions.append("silver")
            if not failed:
                failed = True
                raise RuntimeError("interrupção Silver")
            write(
                conf.storage.silver_bucket,
                conf.storage.silver_current_key,
                {"silver_dataset_id": "s1", "source_bronze_artifact_id": "b1"},
            )
        elif name == "src.enrichment.build_enrichment_dataset":
            executions.append("enrichment")
            write(
                conf.storage.enrichment_bucket,
                os.environ["ENRICHMENT_DATASET_POINTER"],
                {"enrichment_dataset_id": "e1", "source_bronze_artifact_id": "b1"},
            )
        elif name == "src.indexing.build_enrichment_embedding_indexes":
            executions.append("enrichment-index")
            write(
                conf.storage.embedding_indexes_bucket,
                "aux/manifest.json",
                {"validation": {"qdrant_vectors": 1}},
            )
            write(
                conf.storage.embedding_indexes_bucket,
                os.environ["ENRICHMENT_INDEX_POINTER"],
                {
                    "status": "success",
                    "source_enrichment_dataset_id": "e1",
                    "indexes": {
                        "colibri": {
                            "manifest_key": "aux/manifest.json",
                            "collection": "aux",
                        }
                    },
                },
            )
        else:
            pytest.fail("Etapa inesperada: " + name)

    monkeypatch.setattr(production, "run_module", run_stage)
    args = SimpleNamespace(
        run_id="cloud-v1",
        raw_id="machado-v1",
        steps=list(production.RAW_STEPS),
        reuse_index_id=None,
        first_page=1,
        last_page=7,
        promote=False,
    )
    with pytest.raises(RuntimeError, match="interrupção Silver"):
        production.execute(args)
    checkpoint = production.read_json(
        aws, conf.storage.embedding_indexes_bucket, "production/runs/cloud-v1.json"
    )
    assert checkpoint["completed"] == ["bronze"]
    assert not aws.object_exists(
        conf.storage.embedding_indexes_bucket, "production/etl.lock"
    )
    result = production.execute(args)
    assert result["completed"] == list(production.RAW_STEPS)
    assert executions == [
        "bronze",
        "silver",
        "silver",
        "text-index",
        "enrichment",
        "enrichment-index",
    ]
    assert not aws.object_exists(
        conf.storage.embedding_indexes_bucket, "production/current.json"
    )
    args.promote = True
    production.execute(args)
    assert len(executions) == 6
    release = production.read_json(
        aws, conf.storage.embedding_indexes_bucket, "production/current.json"
    )
    assert release["raw_id"] == "machado-v1"
    assert release["raw_manifest_sha256"] == receipt["manifest_sha256"]
    assert release["index_id"] == "text"
    # Outro Raw válido ou outra revisão não podem tomar o checkpoint deste run_id.
    ready(aws, monkeypatch, "machado-v2")
    args.raw_id = "machado-v2"
    with pytest.raises(ValueError, match="configuração mudou"):
        production.execute(args)
    args.raw_id = "machado-v1"
    monkeypatch.setenv("SOURCE_REVISION", "b" * 40)
    with pytest.raises(ValueError, match="configuração mudou"):
        production.execute(args)
    args.steps = list(production.STEPS)
    with pytest.raises(ValueError, match="Raw exige"):
        production.execute(args)


@pytest.mark.parametrize(
    "raw_id,steps,smoke",
    [
        ("../bad", production.RAW_STEPS, False),
        ("raw-v1", production.STEPS, False),
        ("raw-v1", production.RAW_STEPS, True),
    ],
)
def test_controller_rejects_raw_with_crawler_or_smoke(raw_id, steps, smoke):
    cloud = script("raw_cloud_invalid", "scripts/cloud.py")
    with pytest.raises(ValueError, match="raw-id"):
        cloud.glue_arguments(
            "a" * 40, "ops", "corpus-v1", ",".join(steps), raw_id=raw_id, smoke=smoke
        )


def test_job_forwards_raw_and_five_steps_without_empty_optional_values(
    aws, monkeypatch
):
    cloud = script("raw_cloud", "scripts/cloud.py")
    client = Mock()
    client.start_job_run.return_value = {"JobRunId": "jr-fake"}
    monkeypatch.setattr(
        cloud, "outputs", lambda _: {"ops_bucket": "crawler-ops", "glue_job": "etl"}
    )
    monkeypatch.setattr(cloud, "bundle_manifest", lambda *args: {"revision": "a" * 40})
    factory = cloud.boto3.client
    monkeypatch.setattr(
        cloud.boto3,
        "client",
        lambda name, **kwargs: client if name == "glue" else factory(name, **kwargs),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "cloud.py",
            "glue",
            "--revision",
            "a" * 40,
            "--run-id",
            "cloud-v1",
            "--raw-id",
            "machado-v1",
            "--steps",
            ",".join(production.RAW_STEPS),
        ],
    )
    cloud.main()
    arguments = client.start_job_run.call_args.kwargs["Arguments"]
    assert arguments["--raw-id"] == "machado-v1"
    assert (
        arguments["--steps"] == "bronze,silver,text-index,enrichment,enrichment-index"
    )
    assert "--reuse-index-id" not in arguments
    assert all(arguments.values())
    source = io.BytesIO()
    with zipfile.ZipFile(source, "w") as archive:
        archive.writestr("config/source_revision.txt", "a" * 40)
    aws.upload_bytes(
        "crawler-ops", "glue/" + "a" * 40 + "/source.zip", source.getvalue()
    )
    aws.upload_bytes("crawler-ops", "config/buckets.json", json.dumps(BUCKETS).encode())
    boto3.client("ssm", region_name="us-east-1").put_parameter(
        Name="/test/etl",
        Type="SecureString",
        Value=json.dumps(
            {"QDRANT_URL": "https://example.org", "QDRANT_API_KEY": "testing"}
        ),
    )
    job = script("raw_glue_job", "infra/glue/job.py")
    argv = [
        "job.py",
        "--assets-bucket",
        "crawler-ops",
        "--settings-parameter",
        "/test/etl",
    ]
    for name, value in arguments.items():
        argv.extend([name, value])
    argv.extend(["--JOB_NAME", "etl", "--JOB_RUN_ID", "jr-fake"])
    monkeypatch.setattr(sys, "argv", argv)
    calls = []
    monkeypatch.setattr(
        job.runpy,
        "run_module",
        lambda name, **kwargs: calls.append((name, list(sys.argv))),
    )
    previous_path = list(sys.path)
    try:
        job.main()
    finally:
        sys.path[:] = previous_path
    assert calls == [
        (
            "src.etl.production",
            [
                "src.etl.production",
                "--run-id",
                "cloud-v1",
                "--steps",
                *production.RAW_STEPS,
                "--raw-id",
                "machado-v1",
            ],
        )
    ]
