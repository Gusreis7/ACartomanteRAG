from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import (
    BaseModel,
    Field,
    model_validator,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_ENV_PATH = PROJECT_ROOT / "config" / ".env"

DEFAULT_YAML_PATH = PROJECT_ROOT / "config" / "config.yaml"


class RustFSConfig(BaseModel):
    endpoint: str
    access_key: str
    secret_key: str


class QdrantConfig(BaseModel):
    url: str
    api_key: str | None = None


class EnvironmentConfig(BaseModel):
    rustfs: RustFSConfig
    qdrant: QdrantConfig | None = None


class StorageConfig(BaseModel):
    raw_bucket: str = "acartomante-raw"
    bronze_bucket: str = "acartomante-bronze"
    silver_bucket: str = "acartomante-silver"


class ColetaneaConfig(BaseModel):
    similaridade_threshold: float = Field(
        default=0.10,
        ge=0,
        le=1,
    )
    qtd_titulos_minima: int = Field(
        default=2,
        ge=1,
    )
    tamanho_fragmento: int = Field(
        default=1000,
        gt=0,
    )
    tamanho_minimo_fragmento: int = Field(
        default=100,
        gt=0,
    )


class CartaConfig(BaseModel):
    min_assinaturas: int = Field(
        default=5,
        ge=0,
    )


class PoemaConfig(BaseModel):
    max_palavras_linha: int = Field(
        default=10,
        gt=0,
    )
    min_pct_linhas_curtas: float = Field(
        default=0.85,
        ge=0,
        le=1,
    )


class TeatroConfig(BaseModel):
    min_repeticoes_locutor: int = Field(
        default=3,
        ge=1,
    )
    min_locutores: int = Field(
        default=2,
        ge=1,
    )
    min_turnos: int = Field(
        default=10,
        ge=1,
    )
    min_pct_locutores: float = Field(
        default=0.91,
        ge=0,
        le=1,
    )


class BronzeConfig(BaseModel):
    coletanea: ColetaneaConfig
    carta: CartaConfig
    poema: PoemaConfig
    teatro: TeatroConfig


class ChunkTokenizerConfig(BaseModel):
    model: str
    revision: str = "main"


class SilverChunkingConfig(BaseModel):
    strategy: Literal[
        "fixed",
        "hybrid",
    ] = "hybrid"

    tokenizer: str

    chunk_size: int = Field(
        default=512,
        gt=0,
    )

    overlap: int = Field(
        default=64,
        ge=0,
    )

    @model_validator(mode="after")
    def validate_overlap(self):
        if self.overlap >= self.chunk_size:
            raise ValueError(
                "silver.chunking.overlap "
                "deve ser menor que "
                "silver.chunking.chunk_size"
            )

        return self


class SilverSemanticConfig(BaseModel):
    min_capitulos: int = Field(
        default=2,
        ge=2,
    )

    min_mediana_chars: int = Field(
        default=200,
        ge=0,
    )


class SilverConfig(BaseModel):
    pipeline_version: int = Field(
        default=1,
        ge=1,
    )

    tokenizers: dict[
        str,
        ChunkTokenizerConfig,
    ]

    chunking: SilverChunkingConfig

    semantic: SilverSemanticConfig = SilverSemanticConfig()

    @model_validator(mode="after")
    def validate_tokenizer(self):
        if self.chunking.tokenizer not in self.tokenizers:
            raise ValueError(
                "Tokenizer selecionado em "
                "silver.chunking.tokenizer "
                "não existe em "
                "silver.tokenizers."
            )

        return self


class AppConfig(BaseModel):
    environment: EnvironmentConfig
    storage: StorageConfig
    bronze: BronzeConfig
    silver: SilverConfig


class ConfigManager:
    @classmethod
    def load(
        cls,
        env_file: str | Path = (DEFAULT_ENV_PATH),
        yaml_file: str | Path = (DEFAULT_YAML_PATH),
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
                "Arquivo de configuração " "não encontrado: " f"{yaml_file}"
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
            raise ValueError("config.yaml deve possuir " "um objeto YAML na raiz.")

        rustfs_endpoint = os.getenv("RUSTFS_ENDPOINT")
        rustfs_access_key = os.getenv("RUSTFS_ACCESS_KEY")
        rustfs_secret_key = os.getenv("RUSTFS_SECRET_KEY")

        required = {
            "RUSTFS_ENDPOINT": (rustfs_endpoint),
            "RUSTFS_ACCESS_KEY": (rustfs_access_key),
            "RUSTFS_SECRET_KEY": (rustfs_secret_key),
        }

        missing = [key for key, value in required.items() if not value]

        if missing:
            raise ValueError(
                "Variáveis de ambiente " "obrigatórias ausentes: " + ", ".join(missing)
            )

        environment = {
            "rustfs": {
                "endpoint": (rustfs_endpoint),
                "access_key": (rustfs_access_key),
                "secret_key": (rustfs_secret_key),
            }
        }

        qdrant_url = os.getenv("QDRANT_URL")

        if qdrant_url:
            environment["qdrant"] = {
                "url": qdrant_url,
                "api_key": os.getenv("QDRANT_API_KEY"),
            }

        return AppConfig.model_validate(
            {
                **yaml_config,
                "environment": environment,
            }
        )
