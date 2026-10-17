from __future__ import annotations

BENCHMARK_PREFIX = "benchmark_completo"
BENCHMARK_SOURCE_KEY = "benchmark_questions_final.csv"

SPLIT_SEED = 20260930
DEV_RATIO = 0.60
HOLDOUT_RATIO = 0.40

CHUNK_VARIANTS = (
    {"strategy": "fixed", "chunk_size": 256, "overlap": 32},
    {"strategy": "fixed", "chunk_size": 512, "overlap": 64},
    {"strategy": "fixed", "chunk_size": 1024, "overlap": 128},
    {"strategy": "hybrid", "chunk_size": 256, "overlap": 32},
    {"strategy": "hybrid", "chunk_size": 512, "overlap": 64},
    {"strategy": "hybrid", "chunk_size": 1024, "overlap": 128},
)

EMBEDDERS = (
    "colibri",
    "qwen3_embedding_06b",
)

RETRIEVALS = (
    {"strategy": "bm25", "embedder": None},
    {"strategy": "dense", "embedder": "colibri"},
    {"strategy": "dense", "embedder": "qwen3_embedding_06b"},
    {"strategy": "dense_cross_encoder", "embedder": "colibri"},
    {"strategy": "dense_cross_encoder", "embedder": "qwen3_embedding_06b"},
)

OLLAMA_MODELS = (
    {
        "name": "llama3:latest",
        "ollama_id": "365c0bd3c000",
    },
    {
        "name": "qwen3.5:4b",
        "ollama_id": "2a654d98e6fb",
    },
    {
        "name": "ministral-3:3b",
        "ollama_id": "f04aa1c738f6",
    },
    {
        "name": "mistral:7b-instruct-v0.3-q4_K_M",
        "ollama_id": "6577803aa9a0",
    },
)

# Alternativas do MCQ na AWS; os demais parâmetros vêm da configuração publicada.
AWS_MCQ_MODELS = {
    "qwen3-32b": {
        "label": "Qwen3 32B",
        "provider": "bedrock-mantle",
        "model": "qwen.qwen3-32b",
    },
    "llama4-scout": {
        "label": "Llama 4 Scout",
        "provider": "bedrock",
        "model": "us.meta.llama4-scout-17b-instruct-v1:0",
    },
}
