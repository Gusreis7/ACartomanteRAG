"""Contrato HTTP/IAM e migração de checkpoint, sem inferência ou AWS reais."""
import hashlib
import importlib.util
import io
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
import pandas as pd
from moto import mock_aws
from pydantic import BaseModel
import pytest

from src.config.config_manager import ConfigManager, GenerationConfig, EnrichmentConfig, ENRICHMENT_RUNTIME_FIELDS
from src.etl import production
from src.generation.provider import BedrockGenerationProvider, GenerationServiceError, GenerationResponseError, _MantleClient
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
    inputs = {"source_revision": OLD, "bronze": config.bronze.model_dump(), "pages": [1, 7], "silver": config.silver.model_dump(), "embedding": config.embedding.model_dump(exclude={"execution"}), "enrichment": config.enrichment.model_dump(exclude=ENRICHMENT_RUNTIME_FIELDS), "reuse_index_id": None, "raw": receipt}
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
    enrichment = config.enrichment.model_dump(exclude=ENRICHMENT_RUNTIME_FIELDS)
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


class Answer(BaseModel):
    option: str


@pytest.mark.parametrize("first_stop", ["stop", None])
def test_three_response_retries_cover_empty_truncated_and_invalid_json(monkeypatch, first_stop):
    payloads = [completion("", first_stop), completion("parcial", "length"), completion("{}"), completion('{"option":"OK"}')]
    requests, delays = [], []
    def serve(request):
        requests.append(request)
        return httpx.Response(200, json=payloads[len(requests)-1], headers={"x-request-id": f"request-{len(requests)}"})
    provider, _ = mantle(serve, response_retries=3)
    monkeypatch.setattr("src.generation.provider.sleep", delays.append)
    assert provider.invoke_structured([("human", "Teste")], Answer).option == "OK"
    assert len(requests) == 4 and delays == [10, 20, 40]
    assert provider.usage_snapshot() == {"calls": 4, "input_tokens": 400, "output_tokens": 40}


def test_exhausted_response_preserves_finish_tokens_reasoning_and_request(monkeypatch, caplog):
    payload = completion("SECRET_BODY", "length")
    payload["usage"].update(completion_tokens=512, completion_tokens_details={"reasoning_tokens": 500})
    requests = []
    def serve(request):
        requests.append(request)
        return httpx.Response(200, json=payload, headers={"x-request-id": "request-last"})
    provider, _ = mantle(serve, response_retries=3)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    with pytest.raises(GenerationResponseError) as error:
        provider.invoke_structured([("human", "SECRET_PROMPT")], Answer)
    details = error.value.diagnostics
    assert details["reason"] == "truncated" and details["finish_reason"] == "length"
    assert details["output_limit_reached"] and details["reasoning_tokens"] == 500
    assert details["response_attempts"] == 4 and details["request_id"] == "request-last"
    assert len(requests) == 4 and provider.usage_snapshot()["output_tokens"] == 2048
    assert "SECRET_BODY" not in caplog.text and "SECRET_PROMPT" not in caplog.text


@pytest.mark.parametrize("daily,expected", [(False, 3), (True, 1)])
def test_quota_is_not_multiplied_by_response_retries(monkeypatch, daily, expected):
    requests = []
    def serve(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"type": "rate_limit_exceeded", "message": "Too many tokens per day" if daily else "Too many tokens per minute"}})
    provider, _ = mantle(serve, response_retries=3)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    with pytest.raises(GenerationServiceError) as error:
        provider.invoke_structured([("human", "Teste")], Answer)
    assert len(requests) == expected
    assert error.value.batch_fatal and error.value.diagnostics["reason"] == "quota"
    assert error.value.diagnostics["quota_scope"] == ("daily" if daily else "minute")


def test_filtered_response_is_identified_and_not_retried(monkeypatch):
    requests = []
    def serve(request):
        requests.append(request)
        return httpx.Response(200, json=completion("Texto bloqueado", "content_filter"))
    provider, _ = mantle(serve, response_retries=3)
    with pytest.raises(GenerationResponseError) as error:
        provider.invoke_structured([("human", "Teste")], Answer)
    assert len(requests) == 1
    assert error.value.diagnostics["finish_reason"] == "content_filter"
    assert error.value.diagnostics["reason"] == "filtered"


def test_preflight_retries_response_with_same_64_token_budget(monkeypatch):
    requests = []
    def serve(request):
        requests.append(request)
        assert json.loads(request.content)["max_tokens"] == 64
        return httpx.Response(200, json=completion("" if len(requests)<4 else "OK"))
    provider, _ = mantle(serve, response_retries=3, response_retry_max_tokens=4096)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    provider.preflight()
    assert len(requests) == 4


def test_truncated_json_grows_budget_within_three_retries_and_resets_next_call(monkeypatch):
    budgets = []
    def serve(request):
        budget = json.loads(request.content)["max_tokens"]
        budgets.append(budget)
        payload = completion("parcial" if len(budgets) < 4 else '{"option":"OK"}', "length" if len(budgets) < 4 else "stop")
        payload["usage"]["completion_tokens"] = budget if len(budgets) < 4 else 10
        return httpx.Response(200, json=payload)
    provider, _ = mantle(serve, max_tokens=1024, response_retries=3, response_retry_max_tokens=4096)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    assert provider.invoke_structured([("human", "Teste")], Answer).option == "OK"
    assert budgets == [1024, 2048, 4096, 4096]
    assert provider.usage_snapshot() == {"calls": 4, "input_tokens": 400, "output_tokens": 7178}
    assert provider.config.max_tokens == 1024
    assert provider.invoke_structured([("human", "Outra unidade")], Answer).option == "OK"
    assert budgets[-1] == 1024


@pytest.mark.parametrize("stop,tokens,text", [("length", 100, "parcial"), ("model_context_window_exceeded", 1024, "parcial"), ("stop", 1024, "{}")])
def test_budget_does_not_grow_for_other_response_failures(monkeypatch, stop, tokens, text):
    budgets = []
    def serve(request):
        budgets.append(json.loads(request.content)["max_tokens"])
        payload = completion(text, stop)
        payload["usage"]["completion_tokens"] = tokens
        return httpx.Response(200, json=payload)
    provider, _ = mantle(serve, max_tokens=1024, response_retries=3, response_retry_max_tokens=4096)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    with pytest.raises(GenerationResponseError):
        provider.invoke_structured([("human", "Teste")], Answer)
    assert budgets == [1024] * 4


def test_truncation_still_fails_at_the_configured_cap(monkeypatch):
    budgets = []
    def serve(request):
        budget = json.loads(request.content)["max_tokens"]
        budgets.append(budget)
        payload = completion("parcial", "length")
        payload["usage"]["completion_tokens"] = budget
        return httpx.Response(200, json=payload, headers={"x-request-id": "last-request"})
    provider, _ = mantle(serve, max_tokens=1024, response_retries=3, response_retry_max_tokens=3000)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    with pytest.raises(GenerationResponseError) as error:
        provider.invoke_structured([("human", "Teste")], Answer)
    assert budgets == [1024, 2048, 3000, 3000]
    assert error.value.diagnostics["max_tokens"] == 3000
    assert error.value.diagnostics["response_attempts"] == 4
    assert error.value.diagnostics["request_id"] == "last-request"


@pytest.mark.parametrize("model", [GenerationConfig, EnrichmentConfig])
@pytest.mark.parametrize("cap", [512, 8193])
def test_retry_budget_rejects_invalid_limits(model, cap):
    with pytest.raises(ValueError):
        model(max_tokens=1024, response_retry_max_tokens=cap)


@pytest.fixture
def enrichment_batch(checkpoint, monkeypatch):
    from src.enrichment import build_enrichment_dataset as builder
    c = checkpoint
    titles = ["A falha", "B boa", "C boa"]
    bronze = pd.DataFrame([{"source_document_id": str(i), "titulo": title, "autor": "Machado", "categoria": "Conto", "bronze_object_key": f"texts/{i}.txt"} for i,title in enumerate(titles)])
    for key in bronze["bronze_object_key"]:
        c.storage.upload_bytes(c.config.storage.bronze_bucket, key, b"Texto de teste.")
    monkeypatch.setattr(builder, "load_bronze_state", lambda **_: ({"bronze_artifact_id": "bronze_production_corpus", "manifest_key": "bronze.parquet"}, bronze, "source-hash"))
    monkeypatch.setattr(builder, "ENRICHMENT_CURRENT_KEY", "production/candidates/corpus/enrichment-dataset.json")
    args = SimpleNamespace(resume=True, limit=None, title=None, fail_fast=False, env_file=None, config_file=None)
    monkeypatch.setattr(builder, "parse_args", lambda: args)
    calls, prefixes, failures = [], [], {"A falha": GenerationResponseError("Falhou após retries", diagnostics={"reason": "truncated", "finish_reason": "length", "request_id": "req-1", "response_attempts": 4})}
    class Service:
        def __init__(self, config):
            self.provider = self
            self.usage = {"calls": 0, "input_tokens": 0, "output_tokens": 0}
        def usage_snapshot(self): return dict(self.usage)
        def enrich(self, source, *, checkpoint_prefix):
            calls.append(source.title); prefixes.append(checkpoint_prefix)
            failure = failures.get(source.title)
            if failure and not isinstance(failure, GenerationResponseError): raise failure
            count = 4 if failure else 1
            for key,value in {"calls": count,"input_tokens": 100*count,"output_tokens": 10*count}.items():self.usage[key]+=value
            if source.title in failures: raise failures[source.title]
            return SimpleNamespace(usage={"calls": 1}, plan={"effective_strategy_from_chunker": "fixed", "plan_mode": "window", "planned_units": 1}, blocks=[], summary={"resumo_geral": "OK", "eventos_principais": []}, timings_seconds={"total": .1}, reduce_calls=0)
        def unload_model(self): pass
    monkeypatch.setattr(builder, "WorkEnrichmentService", Service)
    def read_manifest():
        keys=[x["Key"] for x in c.storage.client.list_objects_v2(Bucket=c.config.storage.enrichment_bucket).get("Contents", []) if x["Key"].endswith("/manifest.parquet")]
        assert len(keys)==1
        key=keys[0]
        frame=pd.read_parquet(io.BytesIO(c.storage.download_bytes(c.config.storage.enrichment_bucket,key)))
        artifact=production.read_json(c.storage,c.config.storage.enrichment_bucket,key.replace("manifest.parquet","artifact.json"))
        return frame,artifact
    return SimpleNamespace(builder=builder, context=c, calls=calls, prefixes=prefixes, failures=failures, read_manifest=read_manifest)


def test_manifest_continues_after_work_error_and_retries_only_failed(enrichment_batch):
    b = enrichment_batch
    b.context.config.enrichment.response_retry_max_tokens = 0
    with pytest.raises(SystemExit) as error: b.builder.main()
    assert error.value.code == 2
    frame, artifact = b.read_manifest()
    assert list(frame.status_enrichment) == ["error","success","success"]
    assert list(frame.tentativas_enrichment) == [1,1,1]
    assert frame.iloc[0].erro_tipo == "truncated"
    assert json.loads(frame.iloc[0].erro_detalhes_json)["request_id"] == "req-1"
    assert artifact["status"] == "complete_with_errors"
    assert not b.context.storage.object_exists(b.context.config.storage.enrichment_bucket,b.builder.ENRICHMENT_CURRENT_KEY)
    b.failures.clear()
    b.context.config.enrichment.response_retries = 1
    b.context.config.enrichment.response_retry_max_tokens = 4096
    b.builder.main()
    frame, artifact = b.read_manifest()
    assert b.calls == ["A falha","B boa","C boa","A falha"]
    assert b.prefixes[0] == b.prefixes[-1]
    assert list(frame.tentativas_enrichment)==[2,1,1]
    assert list(frame.llm_calls)==[5,1,1]
    assert list(frame.llm_output_tokens)==[50,10,10]
    assert frame.status_enrichment.eq("success").all()
    assert artifact["status"] == "complete"
    assert b.context.storage.object_exists(b.context.config.storage.enrichment_bucket,b.builder.ENRICHMENT_CURRENT_KEY)


def test_service_quota_saves_error_and_leaves_unattempted_works_pending(enrichment_batch):
    b = enrichment_batch
    b.failures["A falha"] = GenerationServiceError("quota", diagnostics={"reason":"quota","http_status":429,"quota_scope":"daily"}, batch_fatal=True)
    with pytest.raises(SystemExit) as error: b.builder.main()
    assert error.value.code == 2 and b.calls == ["A falha"]
    frame, artifact = b.read_manifest()
    assert list(frame.status_enrichment)==["error","pending","pending"]
    assert artifact["documents_error"]==1 and artifact["documents_pending"]==2


def test_retry_alias_and_old_manifest_upgrade(monkeypatch):
    from src.enrichment import build_enrichment_dataset as builder
    monkeypatch.setattr("sys.argv",["builder","--retry"])
    assert builder.parse_args().resume
    original = pd.DataFrame([{"status_enrichment":"success","titulo":"Já pronto"}])
    upgraded = builder.ensure_retry_columns(original.copy())
    assert upgraded.iloc[0].status_enrichment=="success" and upgraded.iloc[0].tentativas_enrichment==0


def test_pending_enrichment_blocks_auxiliary_index_and_promotion(checkpoint, monkeypatch):
    calls = []
    def interrupted(name, arguments=()):
        calls.append(name)
        assert arguments == ["--retry"]
        raise SystemExit(2)
    monkeypatch.setattr(production,"run_module",interrupted)
    promotion=Mock();monkeypatch.setattr(production,"promote",promotion)
    checkpoint.args.promote=True
    with pytest.raises(SystemExit) as error: production.execute(checkpoint.args)
    assert error.value.code==2 and calls==["src.enrichment.build_enrichment_dataset"]
    state=production.read_json(checkpoint.storage,checkpoint.config.storage.embedding_indexes_bucket,"production/runs/corpus.json")
    assert state["status"]=="pending" and state["completed"]==["bronze","silver","text-index"]
    promotion.assert_not_called()


def test_mantle_code_update_preserves_checkpoint_and_runtime_only_settings(checkpoint, monkeypatch):
    # Simula a primeira run Mantle interrompida e a adoção explícita do patch seguinte.
    stages=production.run_module
    checkpoint.config.enrichment.response_retry_max_tokens=0
    monkeypatch.setattr(production,"run_module",lambda *args: (_ for _ in ()).throw(SystemExit(2)))
    with pytest.raises(SystemExit): production.execute(checkpoint.args)
    monkeypatch.setenv("SOURCE_REVISION","c"*40)
    checkpoint.args.resume_from_revision=NEW
    checkpoint.config.enrichment.response_retries=0
    checkpoint.config.enrichment.response_retry_max_tokens=4096
    monkeypatch.setattr(production,"run_module",stages)
    result=production.execute(checkpoint.args)
    assert result["outputs"]["index_id"]=="existing-text"
    assert result["migrations"][-1]["from_revision"]==NEW
    assert result["migrations"][-1]["to_revision"]=="c"*40
    assert result["source_revision"]=="c"*40
    assert checkpoint.executions==["src.enrichment.build_enrichment_dataset","src.indexing.build_enrichment_embedding_indexes"]


@pytest.mark.parametrize("text,kind,location", [
    ('{"resumo":', "json_syntax", []),
    ('{"resumo":"Texto","eventos":"Evento"}', "schema_validation", ["eventos"]),
    ('{"resumo":"","eventos":[]}', "empty_structure", None),
    ('{"summary":"Campo desconhecido"}', "empty_structure", None),
])
def test_structured_diagnostics_distinguish_syntax_schema_and_empty(text, kind, location, caplog):
    from src.enrichment.service import UnitSummary
    provider, _ = mantle(lambda _: httpx.Response(200, json=completion(text)), response_retries=0)
    with pytest.raises(GenerationResponseError) as error:
        provider.invoke_structured([("human", "PRIVATE_PROMPT")], UnitSummary)
    details = error.value.diagnostics
    assert details["reason"] == "invalid_json" and details["validation_kind"] == kind
    assert details["schema_name"] == "UnitSummary"
    assert details["response_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert error.value.response_text == text
    if location is not None:
        assert details["validation_errors"][0]["loc"] == location
    assert text not in caplog.text and "PRIVATE_PROMPT" not in caplog.text


def test_rejected_response_is_saved_with_context_and_never_as_success(checkpoint, monkeypatch, caplog):
    from src.enrichment.service import WorkEnrichmentService, WorkSource
    provider, _ = mantle(lambda _: httpx.Response(200, json=completion('{"resumo":"","eventos":[]}'), headers={"x-request-id":"request-json-error"}), response_retries=0)
    monkeypatch.setattr("src.enrichment.service.create_generation_provider", lambda _: provider)
    service=WorkEnrichmentService(checkpoint.config)
    source=WorkSource("O almada", "Machado", "Poema", "doc-123", "bronze", "obra.txt", "PRIVATE_SOURCE_TEXT")
    with pytest.raises(GenerationResponseError) as error:
        service.enrich(source, checkpoint_prefix="checkpoints/test-json")
    details=error.value.diagnostics
    assert details["enrichment_context"]["phase"] == "unit"
    assert details["enrichment_context"]["title"] == "O almada"
    assert details["enrichment_context"]["source_document_id"] == "doc-123"
    assert details["enrichment_context"]["unit_id"]
    assert details["enrichment_context"]["end_char"] > details["enrichment_context"]["start_char"]
    payload=production.read_json(checkpoint.storage,details["response_error_bucket"],details["response_error_key"])
    assert payload["response_text"] == '{"resumo":"","eventos":[]}'
    assert payload["expected_schema"]["title"] == "UnitSummary"
    assert not payload["response_capture_truncated"]
    assert "PRIVATE_SOURCE_TEXT" not in json.dumps(payload) and "PRIVATE_SOURCE_TEXT" not in caplog.text
    assert payload["response_text"] not in caplog.text
    objects=checkpoint.storage.client.list_objects_v2(Bucket=details["response_error_bucket"],Prefix="checkpoints/test-json")["Contents"]
    assert len(objects)==1 and "/diagnostics/UnitSummary/" in objects[0]["Key"]


def test_failed_diagnostic_upload_preserves_the_original_validation_error(checkpoint, monkeypatch):
    from src.enrichment.service import WorkEnrichmentService, UnitSummary
    provider, _ = mantle(lambda _: httpx.Response(200, json=completion('{"resumo":"","eventos":[]}')), response_retries=0)
    monkeypatch.setattr("src.enrichment.service.create_generation_provider", lambda _: provider)
    service=WorkEnrichmentService(checkpoint.config)
    service._checkpoint_prefix="checkpoints/test-json"
    monkeypatch.setattr(service.storage,"upload_bytes",Mock(side_effect=RuntimeError("upload-failed")))
    with pytest.raises(GenerationResponseError) as error:
        service._cached_summary(service.unit_llm,[("human","PRIVATE_PROMPT")],UnitSummary)
    assert error.value.diagnostics["validation_kind"]=="empty_structure"
    assert error.value.diagnostics["response_capture_error"]=="RuntimeError"
    assert "response_error_key" not in error.value.diagnostics


def test_failed_response_capture_is_bounded_and_does_not_change_unit_cache_keys(checkpoint, monkeypatch):
    from src.enrichment.service import WorkEnrichmentService, UnitSummary
    monkeypatch.setattr("src.enrichment.service.create_generation_provider",lambda _: Mock())
    service=WorkEnrichmentService(checkpoint.config)
    service._checkpoint_prefix="checkpoints/test-json"
    long_text="x"*70000
    failure=GenerationResponseError("invalid",diagnostics={"reason":"invalid_json","request_id":"r","response_sha256":hashlib.sha256(long_text.encode()).hexdigest()},response_text=long_text)
    llm=Mock();llm.invoke.side_effect=failure
    messages=[("human","PRIVATE_PROMPT")]
    with pytest.raises(GenerationResponseError):service._cached_summary(llm,messages,UnitSummary)
    data=production.read_json(checkpoint.storage,failure.diagnostics["response_error_bucket"],failure.diagnostics["response_error_key"])
    assert len(data["response_text"])==65536 and data["response_capture_truncated"]
    digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()
    key=f"checkpoints/test-json/UnitSummary/{digest}.json"
    checkpoint.storage.upload_bytes(checkpoint.config.storage.enrichment_bucket,key,UnitSummary(resumo="Cached",eventos=[]).model_dump_json().encode())
    assert service._cached_summary(llm,messages,UnitSummary).resumo=="Cached"
    assert llm.invoke.call_count==1


@pytest.mark.parametrize("text,expected", [
    ("I\n" + "."*65 + "\n\n", True),
    ("I\n" + "."*37 + "\n" + "."*37 + "\n", True),
    ("\r\n IV \r\n ........ \r\n", True),
    ("I", False),
    ("", False),
    ("........\n........", False),
    ("I\nCamilo leu a carta.", False),
    ("I\n........\nCamilo partiu.", False),
    ("I\n...\n— Não!", False),
    ("CAPÍTULO I\n........", False),
    ("I\n..", False),
])
def test_only_roman_marker_with_dotted_lines_is_ornamental(text, expected):
    from src.enrichment.service import is_ornamental_unit
    assert is_ornamental_unit(text) is expected


@pytest.fixture
def ornamental_work(checkpoint, monkeypatch):
    from src.enrichment import service as module
    first="I\n"+"."*65+"\n\n"
    second="I\n"+"."*37+"\n"+"."*37+"\n"
    lengths=[3311]*3+[len(first)]+[694]*11+[697]+[len(second)]
    sections=[];start=0
    for position,length in enumerate(lengths,start=1):
        text=first if position==4 else second if position==17 else (f"CAPÍTULO {position}\n"+"Camilo leu a carta e partiu. "*150)[:length]
        sections.append({"section_index":position-1,"section_type":"chapter","section_header":"I" if position in {4,17} else f"CAPÍTULO {position}","section_title":"","start_char":start,"end_char":start+len(text),"text":text})
        start+=len(text)
    assert (sections[3]["start_char"],sections[3]["end_char"])==(9933,10002)
    assert (sections[16]["start_char"],sections[16]["end_char"])==(18333,18411)
    source=module.WorkSource("O almada","Machado de Assis","Poema","000204","bronze","almada.txt","".join(section["text"] for section in sections))
    monkeypatch.setattr(module,"build_sections",lambda **_: (sections,"hybrid"))
    class Provider:
        def __init__(self):self.calls=[]
        def usage_snapshot(self):return {"calls":len(self.calls)}
        def invoke_structured(self,messages,schema):
            self.calls.append(schema.__name__)
            if schema is module.UnitSummary:raise AssertionError("Unidade narrativa deveria usar checkpoint; ornamento não deve chamar LLM.")
            return module.FinalNarrativeSummary(resumo_geral="Camilo leu a carta e partiu.",eventos_principais=["Camilo partiu."])
    provider=Provider()
    monkeypatch.setattr(module,"create_generation_provider",lambda _:provider)
    service=module.WorkEnrichmentService(checkpoint.config)
    return SimpleNamespace(module=module,service=service,provider=provider,source=source,sections=sections,storage=checkpoint.storage,config=checkpoint.config)


def test_real_ornamental_units_are_audited_without_renumbering(ornamental_work):
    c=ornamental_work
    units,plan=c.service._plan_units(c.source)
    assert [unit.unit_id for unit in units]==[f"s{i:03d}_p01" for i in range(1,18) if i not in {4,17}]
    assert plan["detected_units"]==17 and plan["planned_units"]==15
    assert plan["skipped_units_count"]==2
    skipped=plan["skipped_units"]
    assert [item["unit_id"] for item in skipped]==["s004_p01","s017_p01"]
    assert [(item["start_char"],item["end_char"]) for item in skipped]==[(9933,10002),(18333,18411)]
    assert all(item["status"]=="skipped" and item["reason"]=="roman_marker_and_dotted_lines" for item in skipped)
    for item,index in zip(skipped,[3,16]):
        assert item["text_sha256"]==hashlib.sha256(c.sections[index]["text"].encode()).hexdigest()
    assert not c.provider.calls


def test_ornamental_fix_reuses_legacy_unit_checkpoints_and_only_generates_final(ornamental_work,capsys):
    c=ornamental_work
    prefix="checkpoints/existing-generator-signature/existing-source-hash"
    bucket=c.config.storage.enrichment_bucket
    cached={}
    for position,section in enumerate(c.sections,start=1):
        if position in {4,17}:continue
        messages=[("system",c.module.UNIT_SYSTEM_PROMPT),("human",c.module.UNIT_PROMPT_TEMPLATE.format(title=c.source.title,author=c.source.author,max_events=c.module.MAX_EVENTS_PER_BLOCK,unit_id=f"s{position:03d}_p01",section_label=c.module.section_label(section),part_index=1,part_count=1,text=section["text"]))]
        digest=hashlib.sha256(json.dumps(messages,ensure_ascii=False).encode()).hexdigest()
        key=f"{prefix}/UnitSummary/{digest}.json"
        payload=c.module.UnitSummary(resumo=f"Resumo já salvo {position}",eventos=["Camilo partiu."]).model_dump_json().encode()
        c.storage.upload_bytes(bucket,key,payload);cached[key]=payload
    result=c.service.enrich(c.source,checkpoint_prefix=prefix)
    assert c.provider.calls==["FinalNarrativeSummary"]
    assert len(result.blocks)==15 and result.plan["skipped_units_count"]==2
    assert [block["unit_id"] for block in result.blocks]==[f"s{i:03d}_p01" for i in range(1,18) if i not in {4,17}]
    assert all(block["resumo"].startswith("Resumo já salvo") for block in result.blocks)
    for key,payload in cached.items():assert c.storage.download_bytes(bucket,key)==payload
    assert result.summary["resumo_geral"]=="Camilo leu a carta e partiu."
    assert result.usage["calls"]==1
    output=capsys.readouterr().out
    assert "[skip] Enrichment:" in output and "s004_p01" in output and "s017_p01" in output


def test_work_with_only_ornamental_units_still_fails_without_calling_llm(ornamental_work):
    c=ornamental_work
    c.sections[:]=[c.sections[3],c.sections[16]]
    with pytest.raises(RuntimeError,match="sem unidades processáveis"):
        c.service.enrich(c.source,checkpoint_prefix="checkpoints/only-ornaments")
    assert not c.provider.calls
