from __future__ import annotations

import argparse
from pathlib import Path

from src.config.config_manager import ConfigManager
from src.generation.graph import MachadoRAG


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Consulta o RAG local de Machado de Assis."
    )
    parser.add_argument("query", help="Pergunta sobre a coletânea.")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument(
        "--persona",
        choices=("neutral", "cosmo", "damiao"),
        default="neutral",
        help="Persona interpretativa do chat.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kwargs = {}
    if args.env_file is not None:
        kwargs["env_file"] = args.env_file
    if args.config_file is not None:
        kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**kwargs)
    rag = MachadoRAG(config)
    result = rag.ask(
        args.query,
        persona=args.persona,
    )

    print()
    print("=" * 80)
    print("strategy:", result["retrieval_strategy"])
    print("silver:", result["silver_dataset_id"])
    print("index:", result["index_id"])
    print("=" * 80)
    print()
    print(result["answer"])
    print()
    print("EVIDÊNCIAS ENVIADAS AO LLM")

    for evidence in result["evidences"]:
        location = evidence.get("obra") or "obra não informada"
        if evidence.get("secao"):
            location += f" · {evidence['secao']}"
        print(
            f"[{evidence['evidence_id']}] {location} | "
            f"chunk={evidence['chunk_id']}"
        )

    print()
    print("timings_ms:", result["timings_ms"])


if __name__ == "__main__":
    main()
