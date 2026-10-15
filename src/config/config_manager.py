from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_PATH = PROJECT_ROOT / "config" / ".env"
DEFAULT_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"


class RustFSConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str


class QdrantConfig(BaseModel):
    url: str
    api_key: str


class RedisEnvironmentConfig(BaseModel):
    url: str | None = None


class EnvironmentConfig(BaseModel):
    rustfs: RustFSConfig
    qdrant: QdrantConfig
    redis: RedisEnvironmentConfig = Field(
        default_factory=RedisEnvironmentConfig
    )

    @property
    def redis_url(self) -> str | None:
        """Compatibilidade temporária com o runtime v1 já aplicado."""
        return self.redis.url


class StorageConfig(BaseModel):
    raw_bucket: str = "acartomante-raw"
    bronze_bucket: str = "acartomante-bronze"
    silver_bucket: str = "acartomante-silver"
    enrichment_bucket: str = "acartomante-enrichment"
    embedding_indexes_bucket: str = "acartomante-indexes"
    benchmarks_bucket: str = "acartomante-benchmarks"


class ColetaneaConfig(BaseModel):
    similaridade_threshold: float = Field(default=0.10, ge=0, le=1)
    qtd_titulos_minima: int = Field(default=2, ge=1)
    tamanho_fragmento: int = Field(default=1000, gt=0)
    tamanho_minimo_fragmento: int = Field(default=100, gt=0)


class CartaConfig(BaseModel):
    min_assinaturas: int = Field(default=5, ge=0)


class PoemaConfig(BaseModel):
    max_palavras_linha: int = Field(default=10, gt=0)
    min_pct_linhas_curtas: float = Field(default=0.85, ge=0, le=1)


class TeatroConfig(BaseModel):
    min_repeticoes_locutor: int = Field(default=3, ge=1)
    min_locutores: int = Field(default=2, ge=1)
    min_turnos: int = Field(default=10, ge=1)
    min_pct_locutores: float = Field(default=0.91, ge=0, le=1)


class BronzeConfig(BaseModel):
    coletanea: ColetaneaConfig
    carta: CartaConfig
    poema: PoemaConfig
    teatro: TeatroConfig


class ChunkTokenizerConfig(BaseModel):
    model: str
    revision: str = "main"


class SilverChunkingConfig(BaseModel):
    strategy: Literal["fixed", "hybrid"] = "fixed"
    tokenizer: str
    chunk_size: int = Field(default=256, gt=0)
    overlap: int = Field(default=32, ge=0)

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                "silver.chunking.overlap deve ser menor que "
                "silver.chunking.chunk_size"
            )
        return self


class SilverSemanticConfig(BaseModel):
    min_capitulos: int = Field(default=2, ge=2)
    min_mediana_chars: int = Field(default=200, ge=0)


class SilverConfig(BaseModel):
    pipeline_version: int = Field(default=1, ge=1)
    tokenizers: dict[str, ChunkTokenizerConfig]
    chunking: SilverChunkingConfig
    semantic: SilverSemanticConfig = Field(default_factory=SilverSemanticConfig)

    @model_validator(mode="after")
    def validate_tokenizer(self):
        if self.chunking.tokenizer not in self.tokenizers:
            raise ValueError(
                "Tokenizer selecionado em silver.chunking.tokenizer "
                "não existe em silver.tokenizers."
            )
        return self


class EmbeddingModelConfig(BaseModel):
    backend: str = Field(min_length=1)
    model: str = Field(min_length=1)
    revision: str = "main"

    inference_dtype: Literal[
        "float32",
        "float16",
        "bfloat16",
    ] = "float32"

    pooling: Literal[
        "model",
        "mean",
        "cls",
        "max",
    ] = "model"

    document_prompt_name: str | None = None
    query_prompt_name: str | None = None

    truncate_dim: int | None = Field(
        default=None,
        gt=0,
    )


class EmbeddingIndexConfig(BaseModel):
    normalize_embeddings: bool = False

    distance: Literal[
        "cosine",
        "dot",
        "euclid",
    ] = "cosine"


class EmbeddingExecutionConfig(BaseModel):
    device: str = "auto"

    batch_size: int = Field(
        default=32,
        gt=0,
    )


class EmbeddingConfig(BaseModel):
    pipeline_version: int = Field(
        default=1,
        ge=1,
    )

    selected_model: str

    models: dict[
        str,
        EmbeddingModelConfig,
    ]

    index: EmbeddingIndexConfig = Field(default_factory=EmbeddingIndexConfig)

    execution: EmbeddingExecutionConfig = Field(
        default_factory=EmbeddingExecutionConfig
    )

    @model_validator(mode="after")
    def validate_selected_model(self):
        if self.selected_model not in self.models:
            raise ValueError(
                "Modelo selecionado em "
                "embedding.selected_model "
                "não existe em embedding.models."
            )

        return self


class RetrievalIndexConfig(BaseModel):
    release: str = "default"
    index_id: str | None = None


class BM25RetrievalConfig(BaseModel):
    top_k: int = Field(default=10, gt=0)


class DenseRetrievalConfig(BaseModel):
    top_k: int = Field(default=10, gt=0)


class DenseCrossEncoderRetrievalConfig(BaseModel):
    candidate_k: int = Field(default=25, gt=0)
    top_k: int = Field(default=10, gt=0)
    model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    revision: str = "main"
    device: str = "cpu"

    @model_validator(mode="after")
    def validate_candidate_k(self):
        if self.candidate_k < self.top_k:
            raise ValueError(
                "retrievals.dense_cross_encoder.candidate_k deve ser >= top_k."
            )
        return self


class RetrievalStrategiesConfig(BaseModel):
    bm25: BM25RetrievalConfig = Field(default_factory=BM25RetrievalConfig)
    dense: DenseRetrievalConfig = Field(default_factory=DenseRetrievalConfig)
    dense_cross_encoder: DenseCrossEncoderRetrievalConfig = Field(
        default_factory=DenseCrossEncoderRetrievalConfig
    )


class EnrichmentRetrievalConfig(BaseModel):
    enabled: bool = True

    top_k: int = Field(
        default=3,
        gt=0,
    )

    max_context_chars: int = Field(
        default=6000,
        gt=0,
    )

    max_context_fraction: float = Field(
        default=0.25,
        gt=0,
        le=1,
    )

    fallback_embedder: str = Field(
        default="qwen3_embedding_06b",
        min_length=1,
    )

    expose_as_evidence: bool = False

    fail_open: bool = True


class RetrievalBenchmarkConfig(BaseModel):
    evidence_min_coverage: float = Field(default=0.60, ge=0, le=1)
    warmup_runs: int = Field(default=3, ge=0)



class GenerationConfig(BaseModel):
    provider: Literal["ollama"] = "ollama"
    model: str = "qwen3.5:4b"
    base_url: str = "http://localhost:11434"

    temperature: float = Field(
        default=0.1,
        ge=0,
        le=2,
    )

    top_p: float = Field(
        default=0.9,
        gt=0,
        le=1,
    )

    max_tokens: int = Field(
        default=512,
        gt=0,
    )

    num_ctx: int = Field(
        default=8192,
        gt=0,
    )

    timeout_seconds: float = Field(
        default=120,
        gt=0,
    )

    keep_alive: str = "10m"

    max_context_chars: int = Field(
        default=18000,
        gt=0,
    )

    max_parallel: int = Field(
        default=2,
        gt=0,
    )

    acquire_timeout_seconds: float = Field(
        default=15,
        gt=0,
    )



class EnrichmentSemanticConfig(BaseModel):
    min_capitulos: int = Field(default=2, ge=2)
    min_mediana_chars: int = Field(default=200, ge=0)


class EnrichmentConfig(BaseModel):
    pipeline_version: int = Field(default=1, ge=1)
    model: str = Field(default="qwen3.5:4b", min_length=1)
    prompt_version: str = Field(default="summary_events_v1", min_length=1)
    schema_version: str = Field(default="summary_events_v1", min_length=1)
    strategy: Literal["hybrid"] = "hybrid"
    unit_chars: int = Field(default=18000, gt=0)
    char_overlap: int = Field(default=800, ge=0)
    final_context_chars: int = Field(default=24000, gt=0)
    temperature: float = Field(default=0.0, ge=0, le=2)
    top_p: float = Field(default=0.9, gt=0, le=1)
    max_tokens: int = Field(default=1024, gt=0)
    num_ctx: int = Field(default=8192, gt=0)
    num_batch: int = Field(default=256, gt=0)
    max_parallel: int = Field(default=2, gt=0)
    timeout_seconds: float = Field(default=300, gt=0)
    keep_alive: str = "10m"
    reasoning: bool = False
    semantic: EnrichmentSemanticConfig = Field(default_factory=EnrichmentSemanticConfig)

    @model_validator(mode="after")
    def validate_char_overlap(self):
        if self.char_overlap >= self.unit_chars:
            raise ValueError(
                "enrichment.char_overlap deve ser menor que enrichment.unit_chars."
            )
        return self


class ApiConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(
        default=8000,
        gt=0,
        le=65535,
    )


class ChatConfig(BaseModel):
    history_turns: int = Field(
        default=3,
        gt=0,
    )


class AppConfig(BaseModel):
    environment: EnvironmentConfig
    storage: StorageConfig
    bronze: BronzeConfig
    silver: SilverConfig
    embedding: EmbeddingConfig
    retrieval_strategy: Literal["bm25", "dense", "dense_cross_encoder"] = (
        "dense_cross_encoder"
    )
    retrieval_index: RetrievalIndexConfig = Field(default_factory=RetrievalIndexConfig)
    retrievals: RetrievalStrategiesConfig = Field(
        default_factory=RetrievalStrategiesConfig
    )
    retrieval_benchmark: RetrievalBenchmarkConfig = Field(
        default_factory=RetrievalBenchmarkConfig
    )
    enrichment_retrieval: EnrichmentRetrievalConfig = Field(
        default_factory=EnrichmentRetrievalConfig
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    enrichment: EnrichmentConfig = Field(default_factory=EnrichmentConfig)

    @model_validator(mode="after")
    def validate_enrichment_retrieval(self):
        alias = self.enrichment_retrieval.fallback_embedder

        if alias not in self.embedding.models:
            raise ValueError(
                "enrichment_retrieval.fallback_embedder "
                "não existe em embedding.models."
            )

        return self

    @property
    def retrieval(self) -> RetrievalIndexConfig:
        """Compatibilidade com IndexResolver/ReleaseManager."""
        return self.retrieval_index
    api: ApiConfig = Field(default_factory=ApiConfig)
    chat: ChatConfig = Field(default_factory=ChatConfig)


class ConfigManager:
    @classmethod
    def load(
        cls,
        env_file: str | Path = DEFAULT_ENV_PATH,
        yaml_file: str | Path = DEFAULT_YAML_PATH,
    ) -> AppConfig:
        env_file = Path(env_file)
        yaml_file = Path(yaml_file)

        if env_file.exists():
            load_dotenv(
                dotenv_path=env_file,
                override=False,
            )

        if not yaml_file.exists():
            raise FileNotFoundError(
                f"Arquivo de configuração não encontrado: {yaml_file}"
            )

        with yaml_file.open(
            "r",
            encoding="utf-8",
        ) as file:
            yaml_config = yaml.safe_load(file) or {}

        if not isinstance(
            yaml_config,
            dict,
        ):
            raise ValueError("config.yaml deve possuir um objeto YAML na raiz.")

        # Host e container enxergam endpoints diferentes. O override não
        # altera o YAML nem as configurações versionadas dos benchmarks.
        ollama_base_url = os.getenv("OLLAMA_BASE_URL")
        if ollama_base_url:
            yaml_config["generation"] = {
                **(yaml_config.get("generation") or {}),
                "base_url": ollama_base_url,
            }

        environment_values = {
            "RUSTFS_ENDPOINT": os.getenv("RUSTFS_ENDPOINT"),
            "RUSTFS_ACCESS_KEY": os.getenv("RUSTFS_ACCESS_KEY"),
            "RUSTFS_SECRET_KEY": os.getenv("RUSTFS_SECRET_KEY"),
            "QDRANT_URL": os.getenv("QDRANT_URL"),
            "QDRANT_API_KEY": os.getenv("QDRANT_API_KEY"),
            "REDIS_URL": os.getenv("REDIS_URL"),
        }

        required_environment_keys = (
            "RUSTFS_ENDPOINT",
            "RUSTFS_ACCESS_KEY",
            "RUSTFS_SECRET_KEY",
            "QDRANT_URL",
            "QDRANT_API_KEY",
        )

        missing = [
            key
            for key in required_environment_keys
            if not environment_values[key]
        ]

        if missing:
            raise ValueError(
                "Variáveis de ambiente " "obrigatórias ausentes: " + ", ".join(missing)
            )

        environment = {
            "rustfs": {
                "endpoint": (environment_values["RUSTFS_ENDPOINT"]),
                "access_key": (environment_values["RUSTFS_ACCESS_KEY"]),
                "secret_key": (environment_values["RUSTFS_SECRET_KEY"]),
            },
            "qdrant": {
                "url": (environment_values["QDRANT_URL"]),
                "api_key": (environment_values["QDRANT_API_KEY"]),
            },
            "redis": {
                "url": environment_values["REDIS_URL"],
            },
        }

        return AppConfig.model_validate(
            {
                **yaml_config,
                "environment": environment,
            }
        )
