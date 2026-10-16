"""Contrato HTTP/IAM e migração de checkpoint, sem inferência ou AWS reais."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import boto3
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.credentials import Credentials
import httpx
from moto import mock_aws
import pytest

from src.config.config_manager import ConfigManager, GenerationConfig
from src.etl import production
from src.generation.provider import BedrockGenerationProvider, GenerationServiceError, _MantleClient
from src.storage.rustfs_manager import RustFSManager

ROOT = Path(__file__).resolve().parents[1]
OLD = "a" * 40
NEW = "b" * 40


def completion(text="Resposta", stop="stop"):
    return {"choices": [{"message": {"content": text}, "finish_reason": stop}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def mantle(handler, **changes):
    config = GenerationConfig(provider="bedrock-mantle", model="qwen.qwen3-32b",
                              model_prices={"qwen.qwen3-32b": {"input": .15, "output": .6}}, **changes)
    session = Mock()
    session.get_credentials.return_value = Credentials("TESTKEY", "test-secret", "test-token")
    http = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = _MantleClient(config, session=session, http_client=http)
    return BedrockGenerationProvider(config, client=adapter), session


def test_signed_chat_maps_messages_model_tokens_cost_and_session_token():
    def serve(request):
        assert str(request.url) == "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
        assert request.headers["x-amz-security-token"] == "test-token"
        assert "/us-east-1/bedrock-mantle/aws4_request" in request.headers["authorization"]
        # A assinatura cobre exatamente os bytes UTF-8 enviados.
        signed = AWSRequest(method="POST", url=str(request.url), data=request.content,
                            headers={"Content-Type": "application/json", "X-Amz-Date": request.headers["x-amz-date"]})
        SigV4Auth(Credentials("TESTKEY", "test-secret", "test-token"), "bedrock-mantle", "us-east-1").add_auth(signed)
        assert signed.headers["Authorization"] == request.headers["authorization"]
        payload = json.loads(request.content)
        assert payload["model"] == "qwen.qwen3-32b"
        assert payload["messages"][0] == {"role": "system", "content": "Sistema\n\n/no_think"}
        assert payload["messages"][1]["content"] == "Questão\nComplemento"
        assert payload["max_tokens"] == 512 and payload["stream"] is False
        assert "Authorization" not in payload
        return httpx.Response(200, json=completion("<think>privado</think>Resposta"), headers={"x-request-id": "req-1"})
    provider, _ = mantle(serve)
    result = provider.invoke([("system", "Sistema"), ("human", "Questão"), ("human", "Complemento")])
    assert result.content == "Resposta"
    assert result.usage_metadata["total_tokens"] == 110
    assert result.response_metadata["provider"] == "bedrock-mantle"
    assert result.response_metadata["request_id"] == "req-1"
    assert result.response_metadata["estimated_cost_usd"] == pytest.approx(.000021)


def test_retry_refreshes_credentials_and_honors_bounded_retry_after(monkeypatch):
    seen, delays = [], []
    def serve(request):
        seen.append(request.headers["x-amz-security-token"])
        return httpx.Response(429, json={"error": {"type": "throttled"}}, headers={"retry-after": "2"}) if len(seen) < 3 else httpx.Response(200, json=completion())
    provider, session = mantle(serve)
    session.get_credentials.side_effect = [Credentials("KEY", "secret", f"token-{i}") for i in range(3)]
    monkeypatch.setattr("src.generation.provider.sleep", delays.append)
    assert provider.invoke([("human", "Teste")]).response_metadata["retry_attempts"] == 2
    assert seen == ["token-0", "token-1", "token-2"] and delays == [2, 2]


@pytest.mark.parametrize("status,attempts", [(403, 1), (429, 3), (503, 3)])
def test_errors_are_bounded_redacted_and_release_capacity(monkeypatch, caplog, status, attempts):
    calls = []
    def serve(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"type": "AccessDenied", "message": "SECRET_PROMPT"}}, headers={"x-request-id": "req-safe"})
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    provider, _ = mantle(serve)
    with pytest.raises(GenerationServiceError) as error:
        provider.invoke([("human", "SECRET_PROMPT")])
    assert len(calls) == attempts
    assert "SECRET_PROMPT" not in str(error.value) + caplog.text
    assert "req-safe" in caplog.text
    assert provider._slots.acquire(blocking=False)


@pytest.mark.parametrize("text,stop", [("", "stop"), ("parcial", "length"), ("bloqueado", "content_filter")])
def test_empty_truncated_or_filtered_response_is_rejected_and_accounted(text, stop):
    provider, _ = mantle(lambda _: httpx.Response(200, json=completion(text, stop)))
    with pytest.raises(GenerationServiceError):
        provider.invoke([("human", "Teste")])
    assert provider.usage_snapshot()["output_tokens"] == 10


def test_catalog_health_does_not_generate_and_preflight_limits_output():
    seen = []
    def serve(request):
        seen.append((request.method, request.url.path))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "qwen.qwen3-32b"})
        assert json.loads(request.content)["max_tokens"] == 64
        return httpx.Response(200, json=completion("OK."))
    provider, _ = mantle(serve)
    provider.healthcheck()
    provider.preflight()
    assert seen == [("GET", "/v1/models/qwen.qwen3-32b"), ("POST", "/v1/chat/completions")]
    assert provider.usage_snapshot()["calls"] == 1


@pytest.fixture
def checkpoint(monkeypatch):
    for name in list(os.environ):
        if name.startswith(("AWS_", "RUSTFS_", "QDRANT_", "LLM_", "ENRICHMENT_", "DATA_RELEASE")) or name in {"CONFIG_FILE", "ENV_FILE", "COLIBRI_REVISION"}:
            monkeypatch.delenv(name, raising=False)
    for name, value in {"AWS_ACCESS_KEY_ID": "testing", "AWS_SECRET_ACCESS_KEY": "testing", "AWS_DEFAULT_REGION": "us-east-1", "AWS_EC2_METADATA_DISABLED": "true", "STORAGE_PROVIDER": "s3", "QDRANT_URL": "https://qdrant.example.org", "QDRANT_API_KEY": "testing", "SOURCE_REVISION": NEW}.items():
        monkeypatch.setenv(name, value)
    config = ConfigManager.load(env_file=ROOT / "absent.env", yaml_file=ROOT / "config/config.aws.yaml")
    config.storage.raw_manifest_key = "production/candidates/corpus/raw-manifest.parquet"
    config.storage.bronze_current_key = "production/candidates/corpus/bronze.json"
    config.storage.silver_current_key = "production/candidates/corpus/silver.json"
    raw = b"unchanged raw manifest"
    receipt = {"raw_id": "machado-v1", "manifest_sha256": hashlib.sha256(raw).hexdigest(), "first_page": 1, "last_page": 7, "source_query": "Machado"}
    inputs = {"source_revision": OLD, "bronze": config.bronze.model_dump(), "pages": [1, 7], "silver": config.silver.model_dump(), "embedding": config.embedding.model_dump(exclude={"execution"}), "enrichment": config.enrichment.model_dump(exclude={"max_parallel", "timeout_seconds", "keep_alive", "num_batch"}), "reuse_index_id": None, "raw": receipt}
    inputs["enrichment"].update(provider="bedrock", model="qwen.qwen3-32b-v1:0")
    previous = {"run_id": "corpus", "signature": production.checkpoint_signature(inputs), "completed": ["bronze", "silver", "text-index"], "outputs": {"index_id": "existing-text"}, "raw_id": "machado-v1", "raw_manifest_sha256": receipt["manifest_sha256"], "status": "running", "current_step": "enrichment"}
    args = SimpleNamespace(run_id="corpus", raw_id="machado-v1", steps=list(production.RAW_STEPS), reuse_index_id=None, first_page=1, last_page=7, promote=False, resume_from_revision=OLD)
    with mock_aws():
        storage = RustFSManager()
        for name in type(config.storage).model_fields:
            if name.endswith("_bucket"):
                storage.create_bucket(getattr(config.storage, name))
        write = lambda bucket, key, value: production.write_json(storage, bucket, key, value)
        write(config.storage.embedding_indexes_bucket, "production/runs/corpus.json", previous)
        storage.upload_bytes(config.storage.raw_bucket, config.storage.raw_manifest_key, raw)
        write(config.storage.bronze_bucket, config.storage.bronze_current_key, {"bronze_artifact_id": "bronze_production_corpus"})
        write(config.storage.silver_bucket, config.storage.silver_current_key, {"source_bronze_artifact_id": "bronze_production_corpus", "silver_dataset_id": "s1"})
        index = {"status": "success", "index_id": "existing-text", "source": {"silver_dataset_id": "s1"}, "embedding": {"model_alias": "colibri", "resolved_revision": config.embedding.models["colibri"].revision}, "qdrant": {"collection": "existing-text"}, "validation": {"silver_chunks": 15402}}
        write(config.storage.embedding_indexes_bucket, "existing-text/manifest.json", index)
        qdrant = Mock(); qdrant.count.return_value = 15402
        probe = Mock()
        monkeypatch.setattr("src.storage.qdrant_manager.QdrantManager.from_config", lambda _: qdrant)
        monkeypatch.setattr("src.generation.provider.create_generation_provider", lambda _: probe)
        monkeypatch.setattr(production.ConfigManager, "load", lambda: config)
        monkeypatch.setattr("src.etl.raw_dataset.load_raw_dataset", lambda *_: (receipt, raw))
        executions = []
        def stage(name, arguments=()):
            executions.append(name)
            if name == "src.enrichment.build_enrichment_dataset":
                write(config.storage.enrichment_bucket, "production/candidates/corpus/enrichment-dataset.json", {"enrichment_dataset_id": "new-mantle"})
            elif name != "src.indexing.build_enrichment_embedding_indexes":
                pytest.fail("Não deve refazer Bronze, Silver ou embeddings textuais")
        monkeypatch.setattr(production, "run_module", stage)
        yield SimpleNamespace(config=config, storage=storage, previous=previous, args=args, qdrant=qdrant, probe=probe, executions=executions, index=index, write=write)


def test_explicit_migration_preserves_completed_stages_backup_and_retries(checkpoint):
    c = checkpoint
    state = production.execute(c.args)
    assert state["outputs"]["index_id"] == "existing-text"
    assert state["completed"] == list(production.RAW_STEPS)
    assert c.executions == ["src.enrichment.build_enrichment_dataset", "src.indexing.build_enrichment_embedding_indexes"]
    migration = state["migrations"][0]
    assert migration["from_revision"] == OLD and migration["to_revision"] == NEW
    assert production.read_json(c.storage, c.config.storage.embedding_indexes_bucket, migration["backup_key"]) == c.previous
    c.probe.preflight.assert_called_once()
    assert not c.storage.object_exists(c.config.storage.embedding_indexes_bucket, "production/current.json")
    production.execute(c.args)
    c.args.resume_from_revision = ""
    production.execute(c.args)
    assert len(c.executions) == 2 and c.probe.preflight.call_count == 1


@pytest.mark.parametrize("change", ["no-flag", "wrong-sha", "changed-chunking", "changed-raw", "wrong-count", "wrong-revision", "completed-enrichment", "missing-checkpoint", "probe-failed"])
def test_unsafe_migrations_leave_original_checkpoint_and_lock_clean(checkpoint, change):
    c = checkpoint
    bucket = c.config.storage.embedding_indexes_bucket
    key = "production/runs/corpus.json"
    if change == "no-flag": c.args.resume_from_revision = ""
    elif change == "wrong-sha": c.args.resume_from_revision = "c" * 40
    elif change == "changed-chunking": c.config.silver.semantic.min_capitulos += 1
    elif change == "changed-raw": c.previous["raw_manifest_sha256"] = "different"; c.write(bucket, key, c.previous)
    elif change == "wrong-count": c.qdrant.count.return_value = 15401
    elif change == "wrong-revision": c.index["embedding"]["resolved_revision"] = "different"; c.write(bucket, "existing-text/manifest.json", c.index)
    elif change == "completed-enrichment": c.previous["completed"].append("enrichment"); c.write(bucket, key, c.previous)
    elif change == "missing-checkpoint": c.storage.delete_object(bucket, key)
    elif change == "probe-failed": c.probe.preflight.side_effect = GenerationServiceError("quota indisponível")
    with pytest.raises((ValueError, GenerationServiceError)):
        production.execute(c.args)
    if change != "missing-checkpoint":
        assert production.read_json(c.storage, bucket, key) == c.previous
    assert not c.storage.object_exists(bucket, "production/etl.lock")
    assert c.executions == []


def test_enrichment_uses_mantle_factory_and_never_ollama(checkpoint, monkeypatch):
    from src.enrichment.service import WorkEnrichmentService
    factory = Mock(return_value=Mock())
    monkeypatch.setattr("src.enrichment.service.create_generation_provider", factory)
    service = WorkEnrichmentService(checkpoint.config)
    service.unload_model()
    assert factory.call_args.args[0].provider == "bedrock-mantle"
    assert factory.call_args.args[0].model == "qwen.qwen3-32b"


def test_enrichment_signature_keeps_provider_and_region(checkpoint):
    from src.enrichment.build_enrichment_dataset import build_signature
    signature, _, _ = build_signature(config=checkpoint.config, bronze_artifact_id="b1", bronze_manifest_sha256="hash", model_digest="bedrock-mantle:us-east-1:qwen.qwen3-32b", prompt_hash="p", schema_hash="s")
    assert signature["provider"] == "bedrock-mantle" and signature["region"] == "us-east-1"


def test_legacy_signature_matches_the_checkpoint_reported_in_production(checkpoint):
    from src.etl.crawler_ingestao_raw_rustfs import LIST_URL
    config = checkpoint.config
    enrichment = config.enrichment.model_dump(exclude={"max_parallel", "timeout_seconds", "keep_alive", "num_batch"})
    enrichment.update(provider="bedrock", model="qwen.qwen3-32b-v1:0")
    inputs = {"source_revision": "de3ac95e91500f26dcdf43212526fb52f33a2916", "bronze": config.bronze.model_dump(), "pages": [1, 7], "silver": config.silver.model_dump(), "embedding": config.embedding.model_dump(exclude={"execution"}), "enrichment": enrichment, "reuse_index_id": None,
              "raw": {"raw_id": "machado-v1", "manifest_sha256": "292171fb07fceee7f820ed16b0c40e5ccb3a73438461d4ce25025773d670402f", "first_page": 1, "last_page": 7, "source_query": LIST_URL}}
    assert production.checkpoint_signature(inputs) == "cf0951c3f2655efa1eb58523ebfe1e13c53ba8733ad3f0218ab79a327c8e66b8"


def test_controller_forwards_migration_and_rejects_smoke(monkeypatch):
    spec = importlib.util.spec_from_file_location("mantle_cloud", ROOT / "scripts/cloud.py")
    cloud = importlib.util.module_from_spec(spec); spec.loader.exec_module(cloud)
    clients = {name: Mock() for name in ("s3", "ssm", "glue")}
    clients["glue"].start_job_run.return_value = {"JobRunId": "jr_test"}
    monkeypatch.setattr(cloud.boto3, "client", lambda name, **_: clients[name])
    monkeypatch.setattr(cloud, "outputs", lambda _: {"ops_bucket": "ops", "glue_job": "etl"})
    monkeypatch.setattr(cloud, "bundle_manifest", lambda *_: {"revision": NEW})
    argv = ["cloud.py", "glue", "--revision", NEW, "--run-id", "corpus", "--raw-id", "machado-v1", "--steps", ",".join(production.RAW_STEPS), "--resume-from-revision", OLD]
    monkeypatch.setattr("sys.argv", argv)
    cloud.main()
    assert clients["glue"].start_job_run.call_args.kwargs["Arguments"]["--resume-from-revision"] == OLD
    monkeypatch.setattr("sys.argv", [*argv, "--smoke"])
    with pytest.raises(SystemExit): cloud.main()
    assert clients["glue"].start_job_run.call_count == 1
