from __future__ import annotations
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from uuid import uuid4
import boto3
from moto import mock_aws
import pytest
from pydantic import BaseModel
import fakeredis
from src.config.config_manager import ConfigManager, GenerationConfig, RustFSConfig
from src.generation.provider import BedrockGenerationProvider, GenerationServiceError, GenerationCapacityError, GenerationQuotaError, OllamaGenerationProvider
from src.storage.rustfs_manager import RustFSManager

ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def environment(monkeypatch):
    for key in list(os.environ):
        if key.startswith(("RUSTFS_", "QDRANT_", "REDIS_", "AWS_", "LLM_", "ENRICHMENT_", "DATA_RELEASE", "ACARTOMANTE_")) or key in {"STORAGE_PROVIDER", "CONFIG_FILE", "ENV_FILE", "BRONZE_CURRENT_KEY", "SILVER_CURRENT_KEY", "RAW_MANIFEST_KEY", "COLIBRI_REVISION"}:
            monkeypatch.delenv(key, raising=False)
    for key, value in {"AWS_ACCESS_KEY_ID":"testing", "AWS_SECRET_ACCESS_KEY":"testing", "AWS_DEFAULT_REGION":"us-east-1", "AWS_EC2_METADATA_DISABLED":"true", "QDRANT_URL":"https://qdrant.example.org", "QDRANT_API_KEY":"testing", "STORAGE_PROVIDER":"s3"}.items():
        monkeypatch.setenv(key, value)

def config():
    return ConfigManager.load(env_file=ROOT / "absent.env", yaml_file=ROOT / "config/config.aws.yaml")

def response(text="Resposta", stop="end_turn"):
    return {"output":{"message":{"content":[{"text":text}]}}, "stopReason":stop, "usage":{"inputTokens":100,"outputTokens":10}}

def provider(*responses):
    client = Mock()
    client.converse.side_effect = responses or [response()]
    return BedrockGenerationProvider(config().generation, client=client, control_client=Mock())

def test_aws_iam_and_enrichment_independent_of_chat(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "us.meta.llama4-scout-17b-instruct-v1:0")
    value = config()
    assert value.environment.rustfs.provider == "s3"
    assert value.environment.rustfs.access_key is None
    assert value.generation.model.startswith("us.meta")
    assert value.enrichment.model == "qwen.qwen3-32b-v1:0"
    assert value.enrichment_retrieval.fallback_embedder == "colibri"

def test_local_credentials_and_model_preserved(monkeypatch):
    monkeypatch.setenv("STORAGE_PROVIDER", "rustfs")
    with pytest.raises(ValueError, match="RUSTFS_ENDPOINT"):
        config()
    for key,value in {"RUSTFS_ENDPOINT":"http://rustfs:9000", "RUSTFS_ACCESS_KEY":"testing", "RUSTFS_SECRET_KEY":"testing"}.items():
        monkeypatch.setenv(key,value)
    value = ConfigManager.load(env_file=ROOT/"absent.env", yaml_file=ROOT/"config/config.yaml")
    assert value.generation.provider == "ollama"
    assert value.retrieval_index.index_id == "idx_colibri-embed-ptbr_2b279b323307"

def test_s3_manager_does_not_pass_static_credentials():
    with patch("src.storage.rustfs_manager.boto3.client") as factory:
        RustFSManager.from_config(RustFSConfig(provider="s3"))
    assert not {"aws_access_key_id","aws_secret_access_key","endpoint_url"} & factory.call_args.kwargs.keys()

def test_rustfs_manager_preserves_endpoint():
    with patch("src.storage.rustfs_manager.boto3.client") as factory:
        RustFSManager("http://rustfs:9000","access","secret")
    assert factory.call_args.kwargs["endpoint_url"] == "http://rustfs:9000"
    assert factory.call_args.kwargs["aws_secret_access_key"] == "secret"

def test_converse_system_history_usage_and_no_reasoning_leak():
    p=provider(response("<think>oculto</think>Resposta"))
    answer=p.invoke([("system","Sistema"),("human","Pergunta"),("ai","Antes"),("human","Outra")])
    call=p.client.converse.call_args.kwargs
    assert call["system"][0] == {"text":"Sistema"}
    assert call["system"][-1] == {"text":"/no_think"}
    assert [x["role"] for x in call["messages"]] == ["user","assistant","user"]
    assert answer.content == "Resposta"
    assert answer.usage_metadata["total_tokens"] == 110
    assert p.usage_snapshot()["calls"] == 1

def test_provider_errors_hide_payload_and_release_slot():
    p=provider(RuntimeError("sensitive provider detail"))
    with pytest.raises(GenerationServiceError) as error:
        p.invoke([("human","Teste")])
    assert "sensitive" not in str(error.value)
    assert p._slots.acquire(blocking=False)

@pytest.mark.parametrize("text,stop", [("","end_turn"),("parcial","max_tokens")])
def test_invalid_responses_rejected_and_accounted(text,stop):
    p=provider(response(text,stop))
    with pytest.raises(GenerationServiceError):
        p.invoke([("human","Teste")])
    assert p.usage_snapshot()["input_tokens"] == 100

def test_capacity_limit_prevents_call():
    p=provider();p.config.acquire_timeout_seconds=.001
    p._slots.acquire();p._slots.acquire()
    with pytest.raises(GenerationCapacityError):
        p.invoke([("human","Teste")])
    p.client.converse.assert_not_called()

class Selected(BaseModel):
    option: str

def test_structured_retries_record_both_calls():
    p=provider(response("não é JSON"),response('```json\n{"option":"B"}\n```'))
    parsed=p.invoke_structured([("human","Escolha")],Selected)
    assert parsed.option == "B"
    assert p.usage_snapshot()["input_tokens"] == 200
    assert p.client.converse.call_count == 2

def test_structured_failure_stops_after_two_calls():
    p=provider(response("{}"),response("{}"))
    with pytest.raises(GenerationServiceError):
        p.invoke_structured([("human","Escolha")],Selected)
    assert p.client.converse.call_count == 2

def test_readiness_and_warmup_never_generate():
    p=provider();p.healthcheck();p.warmup()
    p.client.converse.assert_not_called()
    assert p.control_client.get_foundation_model.call_count == 2

def test_scout_profile_and_price():
    p=provider();p.config.model="us.meta.llama4-scout-17b-instruct-v1:0"
    p.healthcheck();p.control_client.get_inference_profile.assert_called_once()
    answer=p.invoke([("human","Teste")])
    assert answer.response_metadata["estimated_cost_usd"] == pytest.approx((100*.17+10*.66)/1e6)

def test_ollama_readiness_inventory_only():
    p=object.__new__(OllamaGenerationProvider);p.config=GenerationConfig();p.client=Mock()
    with patch("src.generation.provider.httpx.get") as get:
        get.return_value.json.return_value={"models":[{"name":p.config.model}]}
        p.healthcheck()
    p.client.invoke.assert_not_called()

@mock_aws
def test_release_single_manifest_pins_all_datasets(monkeypatch):
    s3=boto3.client("s3");s3.create_bucket(Bucket="acartomante-indexes")
    s3.put_object(Bucket="acartomante-indexes",Key="production/current.json",Body=json.dumps({
        "status":"success","index_id":"selected-text","enrichment_index_pointer":"candidate/enrichment.json",
        "bronze_pointer":"candidate/bronze.json","silver_pointer":"candidate/silver.json"}).encode())
    monkeypatch.setenv("DATA_RELEASE_KEY","production/current.json")
    value=config()
    assert value.retrieval_index.index_id == "selected-text"
    assert value.storage.bronze_current_key == "candidate/bronze.json"
    assert value.enrichment_retrieval.index_pointer_key == "candidate/enrichment.json"

@mock_aws
def test_lock_refuses_concurrent_etl_and_releases_after_error():
    from src.etl.production import execution_lock
    storage=RustFSManager();storage.create_bucket("test-indexes")
    with pytest.raises(ValueError):
        with execution_lock(storage,"test-indexes","first"):
            with pytest.raises(RuntimeError,match="ETL"):
                with execution_lock(storage,"test-indexes","second"):
                    pytest.fail("Lock concorrente aceito")
            raise ValueError("falha de etapa")
    assert not storage.object_exists("test-indexes","production/etl.lock")

@mock_aws
def test_migration_reuses_identical_bytes_and_refuses_overwrite():
    from src.etl.migrate_storage import copy_verified
    source,target=RustFSManager(),RustFSManager()
    source.create_bucket("source-test");target.create_bucket("target-test")
    source.upload_bytes("source-test","data.txt",b"original")
    assert copy_verified(source,target,"source-test","target-test","data.txt")
    assert not copy_verified(source,target,"source-test","target-test","data.txt")
    source.upload_bytes("source-test","data.txt",b"changed")
    with pytest.raises(ValueError,match="diferentes"):
        copy_verified(source,target,"source-test","target-test","data.txt")
    assert target.download_bytes("target-test","data.txt") == b"original"

def test_redis_lua_session_and_global_limits_and_metadata():
    from src.services.chat import ChatService
    from src.storage.redis_message_repository import RedisMessageRepository
    redis=fakeredis.FakeRedis(decode_responses=True)
    service=object.__new__(ChatService)
    service.repository=SimpleNamespace(redis=redis)
    service.rag=SimpleNamespace(config=config())
    service.rag.config.generation.daily_request_limit=2
    service.rag.config.generation.session_requests_per_minute=1
    user=uuid4();service._check_quota(user)
    with pytest.raises(GenerationQuotaError):service._check_quota(user)
    service._check_quota(uuid4())
    with pytest.raises(GenerationQuotaError):service._check_quota(uuid4())
    messages=RedisMessageRepository(redis)
    session,chat,turn=uuid4(),uuid4(),uuid4()
    messages.append_assistant_message(session_id=session,chat_id=chat,turn_id=turn,content="Resposta",evidences=[],metadata={"model":"qwen","input_tokens":10})
    key=messages._messages_key(session_id=session,chat_id=chat)
    assert json.loads(redis.xrange(key)[0][1]["metadata"]) == {"model":"qwen","input_tokens":10}

@mock_aws
def test_enrichment_cached_work_never_generates_again(monkeypatch):
    from src.enrichment.service import WorkEnrichmentService,WorkSource,UnitSummary,FinalNarrativeSummary
    class FakeProvider:
        calls=0
        def usage_snapshot(self):return {"calls":self.calls}
        def invoke_structured(self,messages,schema):
            self.calls+=1
            return UnitSummary(resumo="Resumo",eventos=["Evento"]) if schema is UnitSummary else FinalNarrativeSummary(resumo_geral="Resumo final",eventos_principais=["Evento final"])
    value=config();storage=RustFSManager();storage.create_bucket(value.storage.enrichment_bucket)
    fake=FakeProvider()
    monkeypatch.setattr("src.enrichment.service.create_generation_provider",lambda *a,**k:fake)
    service=WorkEnrichmentService(value)
    source=WorkSource("A Cartomante","Machado de Assis","conto","1","bronze","text.txt","Camilo lia a carta de Vilela.")
    first=service.enrich(source,checkpoint_prefix="checkpoints/one");calls=fake.calls
    second=service.enrich(source,checkpoint_prefix="checkpoints/one")
    assert calls == 2
    assert fake.calls == calls
    assert first.blocks == second.blocks
    assert second.usage["calls"] == 0

def test_ready_uses_only_free_catalog_checks():
    from src.api.routes import ready
    p=provider()
    state=SimpleNamespace(redis=Mock(),rag=SimpleNamespace(config=config(),retrieval=SimpleNamespace(rustfs=Mock())),generation_provider=p,config=SimpleNamespace(retrieval_strategy="bm25"))
    result=ready(SimpleNamespace(app=SimpleNamespace(state=state)))
    assert result.status == "ready"
    p.client.converse.assert_not_called()

def test_ready_refuses_missing_bucket():
    from src.api.routes import ready
    from fastapi import HTTPException
    storage=Mock();storage.bucket_exists.return_value=False
    state=SimpleNamespace(redis=Mock(),rag=SimpleNamespace(config=config(),retrieval=SimpleNamespace(rustfs=storage)),generation_provider=provider(),config=SimpleNamespace(retrieval_strategy="bm25"))
    with pytest.raises(HTTPException) as error:ready(SimpleNamespace(app=SimpleNamespace(state=state)))
    assert error.value.status_code == 503

@mock_aws
def test_atomic_promotion_requires_both_validated_indices(monkeypatch):
    from src.etl.production import promote,write_json,read_json
    value=config();storage=RustFSManager()
    for bucket in [value.storage.embedding_indexes_bucket,value.storage.bronze_bucket,value.storage.silver_bucket,value.storage.enrichment_bucket]:
        storage.create_bucket(bucket)
    write_json(storage,value.storage.bronze_bucket,"current.json",{"bronze_artifact_id":"b1"})
    write_json(storage,value.storage.silver_bucket,"current.json",{"silver_dataset_id":"s1"})
    write_json(storage,value.storage.enrichment_bucket,"candidate/dataset.json",{"source_bronze_artifact_id":"b1","enrichment_dataset_id":"e1"})
    bucket=value.storage.embedding_indexes_bucket
    write_json(storage,bucket,"text/manifest.json",{"status":"success","source":{"silver_dataset_id":"s1"},"qdrant":{"collection":"text"},"validation":{"silver_chunks":3}})
    write_json(storage,bucket,"aux/manifest.json",{"validation":{"qdrant_vectors":2}})
    write_json(storage,bucket,"candidate/indexes.json",{"status":"success","source_enrichment_dataset_id":"e1","indexes":{"colibri":{"manifest_key":"aux/manifest.json","collection":"aux"}}})
    write_json(storage,bucket,"production/current.json",{"previous":"kept"})
    monkeypatch.setenv("ENRICHMENT_DATASET_POINTER","candidate/dataset.json")
    state={"outputs":{"index_id":"text","enrichment_index_pointer":"candidate/indexes.json"}}
    with patch("src.storage.qdrant_manager.QdrantManager.from_config") as factory:
        factory.return_value.count.return_value=1
        with pytest.raises(ValueError,match="diverge"):
            promote(storage,value,state,"release-v1")
        assert read_json(storage,bucket,"production/current.json") == {"previous":"kept"}
        factory.return_value.count.side_effect=[3,2,3,2]
        promote(storage,value,state,"release-v1")
        first=read_json(storage,bucket,"production/current.json")
        promote(storage,value,state,"release-v1")
        assert read_json(storage,bucket,"production/current.json") == first
        assert first["index_id"] == "text"
        assert storage.object_exists(bucket,"production/releases/release-v1.json")
