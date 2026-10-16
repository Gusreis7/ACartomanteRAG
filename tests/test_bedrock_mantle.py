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

from src.config.config_manager import ConfigManager, GenerationConfig
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
    inputs = {"source_revision": OLD, "bronze": config.bronze.model_dump(), "pages": [1, 7], "silver": config.silver.model_dump(), "embedding": config.embedding.model_dump(exclude={"execution"}), "enrichment": config.enrichment.model_dump(exclude={"max_parallel", "timeout_seconds", "keep_alive", "num_batch", "response_retries"}), "reuse_index_id": None, "raw": receipt}
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
    enrichment = config.enrichment.model_dump(exclude={"max_parallel", "timeout_seconds", "keep_alive", "num_batch", "response_retries"})
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
    provider, _ = mantle(serve, response_retries=3)
    monkeypatch.setattr("src.generation.provider.sleep", lambda _: None)
    provider.preflight()
    assert len(requests) == 4


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
    monkeypatch.setattr(production,"run_module",lambda *args: (_ for _ in ()).throw(SystemExit(2)))
    with pytest.raises(SystemExit): production.execute(checkpoint.args)
    monkeypatch.setenv("SOURCE_REVISION","c"*40)
    checkpoint.args.resume_from_revision=NEW
    checkpoint.config.enrichment.response_retries=0
    monkeypatch.setattr(production,"run_module",stages)
    result=production.execute(checkpoint.args)
    assert result["outputs"]["index_id"]=="existing-text"
    assert result["migrations"][-1]["from_revision"]==NEW
    assert result["migrations"][-1]["to_revision"]=="c"*40
    assert result["source_revision"]=="c"*40
    assert checkpoint.executions==["src.enrichment.build_enrichment_dataset","src.indexing.build_enrichment_embedding_indexes"]
