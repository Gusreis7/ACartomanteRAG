from __future__ import annotations

import argparse
from pathlib import Path

from src.config.config_manager import ConfigManager
from src.retrieval.benchmark import (
    DEFAULT_BENCHMARK_KEY,
    run_retrieval_benchmark,
)
from src.retrieval.service import RetrievalService


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Executa benchmark BM25 vs Dense " "vs Dense + CrossEncoder.")
    )
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)
    parser.add_argument(
        "--index-id",
        default=None,
        help=(
            "Pin opcional do índice. Se omitido, usa a política " "do IndexResolver."
        ),
    )
    parser.add_argument(
        "--benchmark-key",
        default=DEFAULT_BENCHMARK_KEY,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_kwargs = {}
    if args.env_file is not None:
        config_kwargs["env_file"] = args.env_file
    if args.config_file is not None:
        config_kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**config_kwargs)
    service = RetrievalService(config, index_id=args.index_id)

    print(f"[+] Retrieval selection: {service.resolved.selection}")
    print(f"[+] Silver: {service.resolved.silver_dataset_id}")
    print(f"[+] Index: {service.resolved.index_id}")
    print(f"[+] Collection: {service.resolved.collection}")
    print(f"[+] BM25 build: {service.bm25.build_ms:.2f} ms")

    result = run_retrieval_benchmark(
        service,
        benchmark_key=args.benchmark_key,
    )

    print()
    print(result["summary"].to_string(index=False))
    print()
    print(
        "[OK] Benchmark salvo em: "
        f"s3://{config.storage.benchmarks_bucket}/{result['prefix']}/"
    )


if __name__ == "__main__":
    main()
