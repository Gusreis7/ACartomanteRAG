#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.matrix import BENCHMARK_PREFIX
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

PAIR_KEYS = ["query_id", "config_id", "model"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compara de forma pareada MCQ baseline vs MCQ com enrichment."
    )
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--enriched-run-id", required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-upload", action="store_true")
    return parser.parse_args()


def read_json(storage: RustFSManager, bucket: str, key: str) -> dict[str, Any]:
    return json.loads(storage.download_bytes(bucket, key).decode("utf-8"))


def read_parquet(storage: RustFSManager, bucket: str, key: str) -> pd.DataFrame:
    return pd.read_parquet(io.BytesIO(storage.download_bytes(bucket, key)))


def parquet_bytes(frame: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    frame.to_parquet(buf, index=False)
    return buf.getvalue()


def load_run(storage: RustFSManager, bucket: str, run_id: str):
    prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"
    config = read_json(storage, bucket, f"{prefix}/config.json")
    manifest = read_json(storage, bucket, f"{prefix}/manifest.json")
    if manifest.get("status") != "success":
        raise RuntimeError(f"Run {run_id} não está success.")
    results = read_parquet(storage, bucket, f"{prefix}/mcq/results.parquet")
    return config, manifest, results


def status(row: pd.Series, suffix: str) -> str:
    for name in ("correct", "ambiguous", "wrong", "invalid"):
        if bool(row[f"is_{name}_{suffix}"]):
            return name
    return "invalid"


def bootstrap_query_delta(
    paired: pd.DataFrame, *, runs: int, seed: int
) -> dict[str, float]:
    per_query = (
        paired.groupby("query_id", sort=False)["accuracy_delta"].mean().astype(float)
    )
    observed = float(per_query.mean()) if len(per_query) else 0.0
    if runs <= 0 or len(per_query) <= 1:
        return {"delta": observed, "ci95_low": observed, "ci95_high": observed}

    rng = np.random.default_rng(seed)
    values = per_query.to_numpy(dtype=float)
    samples = np.empty(runs, dtype=float)
    for i in range(runs):
        draw = rng.choice(values, size=len(values), replace=True)
        samples[i] = float(draw.mean())
    return {
        "delta": observed,
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
    }


def aggregate(paired: pd.DataFrame, scope: str, **keys: Any) -> dict[str, Any]:
    boot = bootstrap_query_delta(
        paired,
        runs=int(keys.pop("bootstrap_runs")),
        seed=int(keys.pop("seed")),
    )
    return {
        "scope": scope,
        **keys,
        "n_rows": int(len(paired)),
        "queries": int(paired["query_id"].nunique()),
        "baseline_accuracy": float(paired["is_correct_baseline"].astype(float).mean()),
        "enriched_accuracy": float(paired["is_correct_enriched"].astype(float).mean()),
        "accuracy_delta": boot["delta"],
        "accuracy_delta_ci95_low": boot["ci95_low"],
        "accuracy_delta_ci95_high": boot["ci95_high"],
        "improved_rows": int((paired["accuracy_delta"] > 0).sum()),
        "regressed_rows": int((paired["accuracy_delta"] < 0).sum()),
        "unchanged_rows": int((paired["accuracy_delta"] == 0).sum()),
        "baseline_input_tokens_mean": float(
            pd.to_numeric(paired.get("input_tokens_baseline"), errors="coerce").mean()
        ),
        "enriched_input_tokens_mean": float(
            pd.to_numeric(paired.get("input_tokens_enriched"), errors="coerce").mean()
        ),
        "enrichment_context_chars_mean": float(
            pd.to_numeric(
                paired.get("enrichment_context_chars_enriched"), errors="coerce"
            ).mean()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.bootstrap_runs < 0:
        raise SystemExit("--bootstrap-runs deve ser >= 0.")

    config = ConfigManager.load()
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.benchmarks_bucket

    base_cfg, base_manifest, baseline = load_run(storage, bucket, args.baseline_run_id)
    enr_cfg, enr_manifest, enriched = load_run(storage, bucket, args.enriched_run_id)

    if bool((base_cfg.get("enrichment") or {}).get("enabled", False)):
        raise RuntimeError("O run baseline informado já possui enrichment.")
    if not bool((enr_cfg.get("enrichment") or {}).get("enabled", False)):
        raise RuntimeError("O run enriched informado não possui enrichment.")
    if base_cfg.get("source_retrieval_run_id") != enr_cfg.get(
        "source_retrieval_run_id"
    ):
        raise RuntimeError("Runs usam source_retrieval_run_id diferentes.")
    if base_cfg.get("benchmark_sha256") != enr_cfg.get("benchmark_sha256"):
        raise RuntimeError("Runs usam benchmark_sha256 diferentes.")

    required = set(PAIR_KEYS + ["is_correct", "selected_class", "correct_option"])
    for label, frame in (("baseline", baseline), ("enriched", enriched)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise RuntimeError(f"{label} sem colunas: {missing}")
        if frame.duplicated(PAIR_KEYS).any():
            raise RuntimeError(f"{label} possui pares duplicados em {PAIR_KEYS}.")

    paired = baseline.merge(
        enriched,
        on=PAIR_KEYS,
        how="inner",
        suffixes=("_baseline", "_enriched"),
        validate="one_to_one",
    )
    if len(paired) != len(baseline) or len(paired) != len(enriched):
        raise RuntimeError(
            "Escopos baseline/enriched não são idênticos. "
            f"baseline={len(baseline)} enriched={len(enriched)} paired={len(paired)}"
        )

    paired["accuracy_delta"] = paired["is_correct_enriched"].astype(int) - paired[
        "is_correct_baseline"
    ].astype(int)
    paired["transition"] = paired.apply(
        lambda row: f"{str(row['selected_class_baseline'])}->{str(row['selected_class_enriched'])}",
        axis=1,
    )

    rows: list[dict[str, Any]] = []
    rows.append(
        aggregate(
            paired,
            "overall",
            bootstrap_runs=args.bootstrap_runs,
            seed=args.seed,
        )
    )
    for model, group in paired.groupby("model", sort=False):
        rows.append(
            aggregate(
                group,
                "model",
                model=str(model),
                bootstrap_runs=args.bootstrap_runs,
                seed=args.seed,
            )
        )
    for config_id, group in paired.groupby("config_id", sort=False):
        rows.append(
            aggregate(
                group,
                "config",
                config_id=str(config_id),
                bootstrap_runs=args.bootstrap_runs,
                seed=args.seed,
            )
        )
    if "difficulty_baseline" in paired.columns:
        for difficulty, group in paired.groupby("difficulty_baseline", sort=False):
            rows.append(
                aggregate(
                    group,
                    "difficulty",
                    difficulty=str(difficulty),
                    bootstrap_runs=args.bootstrap_runs,
                    seed=args.seed,
                )
            )
    if "query_type_baseline" in paired.columns:
        for query_type, group in paired.groupby("query_type_baseline", sort=False):
            rows.append(
                aggregate(
                    group,
                    "query_type",
                    query_type=str(query_type),
                    bootstrap_runs=args.bootstrap_runs,
                    seed=args.seed,
                )
            )

    summary = pd.DataFrame(rows)
    transitions = (
        paired["transition"]
        .value_counts()
        .rename_axis("transition")
        .reset_index(name="n")
    )
    overall = summary[summary["scope"].eq("overall")].iloc[0].to_dict()

    payload = {
        "baseline_run_id": args.baseline_run_id,
        "enriched_run_id": args.enriched_run_id,
        "source_retrieval_run_id": base_cfg.get("source_retrieval_run_id"),
        "benchmark_sha256": base_cfg.get("benchmark_sha256"),
        "paired_keys": PAIR_KEYS,
        "bootstrap": {
            "unit": "query_id",
            "runs": args.bootstrap_runs,
            "seed": args.seed,
        },
        "overall": overall,
        "transitions": transitions.to_dict(orient="records"),
        "enrichment": enr_cfg.get("enrichment") or {},
    }

    output = args.output or Path("reports") / (
        f"compare_{args.baseline_run_id}__{args.enriched_run_id}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )

    if not args.no_upload:
        prefix = (
            f"{BENCHMARK_PREFIX}/runs/{args.enriched_run_id}/mcq/enrichment_comparison"
        )
        storage.upload_bytes(
            bucket,
            f"{prefix}/comparison.json",
            json.dumps(payload, ensure_ascii=False, indent=2, default=str).encode(
                "utf-8"
            ),
            content_type="application/json",
        )
        storage.upload_bytes(
            bucket,
            f"{prefix}/summary.parquet",
            parquet_bytes(summary),
            content_type="application/vnd.apache.parquet",
        )
        storage.upload_bytes(
            bucket,
            f"{prefix}/paired_rows.parquet",
            parquet_bytes(paired),
            content_type="application/vnd.apache.parquet",
        )

    print("=== MCQ ENRICHMENT COMPARISON OK ===")
    print(f"baseline: {args.baseline_run_id}")
    print(f"enriched: {args.enriched_run_id}")
    print(f"paired rows: {len(paired)}")
    print(f"queries: {paired['query_id'].nunique()}")
    print(f"baseline accuracy: {overall['baseline_accuracy']:.3f}")
    print(f"enriched accuracy: {overall['enriched_accuracy']:.3f}")
    print(
        "delta: "
        f"{overall['accuracy_delta']:+.3f} "
        f"CI95 [{overall['accuracy_delta_ci95_low']:+.3f}, "
        f"{overall['accuracy_delta_ci95_high']:+.3f}]"
    )
    print(f"local: {output.resolve()}")


if __name__ == "__main__":
    main()
