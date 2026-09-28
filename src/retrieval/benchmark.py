from __future__ import annotations

import io
import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.retrieval.evidence import build_evidence_matches
from src.retrieval.metrics import aggregate_metrics, evaluate_response

DEFAULT_BENCHMARK_KEY = "retrieval/v1/benchmark_questions.csv"


def run_retrieval_benchmark(
    service,
    *,
    benchmark_key: str = DEFAULT_BENCHMARK_KEY,
    methods: tuple[str, ...] = ("bm25", "dense", "dense_ce"),
) -> dict[str, Any]:
    config = service.config
    rustfs = service.rustfs
    benchmark_bucket = config.storage.benchmarks_bucket

    payload = rustfs.download_bytes(benchmark_bucket, benchmark_key)
    benchmark = pd.read_csv(io.BytesIO(payload))

    evidence_matches = build_evidence_matches(
        benchmark,
        service.chunks,
        min_coverage=float(
            config.retrieval_benchmark.evidence_min_coverage
        ),
    )

    relevant_by_query = {}
    valid_matches = evidence_matches[evidence_matches["chunk_id"].notna()]
    for query_id, group in valid_matches.groupby("query_id"):
        relevant_by_query[str(query_id)] = set(
            group["chunk_id"].astype(str)
        )

    warmup = service.warmup()
    print(
        "[+] Warm-up descartado: "
        f"{warmup['runs']} execuções | {warmup['total_ms']:.2f} ms"
    )

    rows = []
    for index, item in benchmark.iterrows():
        query_id = str(item["query_id"])
        query = str(item["query"])
        relevant = relevant_by_query.get(query_id, set())

        if not relevant:
            print(
                "[AVISO] Evidência não mapeada para a Silver: "
                f"{query_id}"
            )

        for method in methods:
            response = service.search(query, method=method)
            metrics = evaluate_response(response, relevant)
            row = {
                "query_id": query_id,
                "query": query,
                "canonical_work": item.get("canonical_work"),
                "query_type": item.get("query_type"),
                "difficulty": item.get("difficulty"),
                "method": method,
                "silver_dataset_id": response.silver_dataset_id,
                "index_id": response.index_id,
                "collection": response.collection,
                "relevant_chunks": len(relevant),
                **metrics,
            }
            for timing_name, timing_value in response.timings_ms.items():
                row[timing_name] = timing_value
            row["retrieved_chunk_ids"] = json.dumps(
                [hit.chunk_id for hit in response.hits],
                ensure_ascii=False,
            )
            rows.append(row)

        print(f"[{index + 1}/{len(benchmark)}] {query_id}")

    results = pd.DataFrame(rows)
    scored_results = results[results["relevant_chunks"].gt(0)].copy()
    summary = aggregate_metrics(scored_results)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prefix = f"retrieval/v1/runs/{run_id}_{service.resolved.index_id}"

    _upload_csv(
        rustfs,
        benchmark_bucket,
        f"{prefix}/evidence_matches.csv",
        evidence_matches,
    )
    _upload_csv(
        rustfs,
        benchmark_bucket,
        f"{prefix}/results.csv",
        results,
    )
    _upload_csv(
        rustfs,
        benchmark_bucket,
        f"{prefix}/summary.csv",
        summary,
    )

    ce_cfg = config.retrievals.dense_cross_encoder
    manifest = {
        "run_id": run_id,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark_key": benchmark_key,
        "benchmark_rows": int(len(benchmark)),
        "scored_queries": int(scored_results["query_id"].nunique()),
        "unmatched_queries": int(
            benchmark["query_id"].astype(str).nunique()
            - scored_results["query_id"].astype(str).nunique()
        ),
        "methods": list(methods),
        "top_k": {
            "bm25": int(config.retrievals.bm25.top_k),
            "dense": int(config.retrievals.dense.top_k),
            "dense_cross_encoder": int(ce_cfg.top_k),
        },
        "dense_candidate_k": int(ce_cfg.candidate_k),
        "evidence_min_coverage": float(
            config.retrieval_benchmark.evidence_min_coverage
        ),
        "silver_dataset_id": service.resolved.silver_dataset_id,
        "index_id": service.resolved.index_id,
        "collection": service.resolved.collection,
        "selection": service.resolved.selection,
        "release": service.resolved.release,
        "reranker": {
            "model": ce_cfg.model,
            "revision": ce_cfg.revision,
        },
        "initialization_ms": {
            "retrieval_service": service.init_ms,
            "bm25_build": service.bm25.build_ms,
            "dense_model": service.dense_init_ms,
            "reranker_model": service.reranker_init_ms,
        },
        "warmup": warmup,
    }

    rustfs.upload_bytes(
        benchmark_bucket,
        f"{prefix}/manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2).encode("utf-8"),
        content_type="application/json",
    )

    return {
        "prefix": prefix,
        "manifest": manifest,
        "results": results,
        "summary": summary,
        "evidence_matches": evidence_matches,
    }


def _upload_csv(rustfs, bucket: str, key: str, frame: pd.DataFrame) -> None:
    payload = frame.to_csv(index=False).encode("utf-8")
    rustfs.upload_bytes(
        bucket,
        key,
        payload,
        content_type="text/csv; charset=utf-8",
    )
