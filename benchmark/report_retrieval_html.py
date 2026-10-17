#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import pandas as pd

from benchmark.matrix import BENCHMARK_PREFIX
from benchmark.prepare import load_json
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager

DEFAULT_METRIC = "hit_at_10"
REPORT_KEY_NAME = "retrieval_dashboard.html"

QUERY_COLUMNS = [
    "query_id",
    "query_variant",
    "query",
    "canonical_work",
    "difficulty",
    "query_type",
    "config_id",
    "experiment_id",
    "chunk_strategy",
    "chunk_size",
    "overlap",
    "retrieval_strategy",
    "embedder",
    "first_hit_rank",
    "mrr_at_10",
    "ndcg_at_10",
    "coverage_auc_at_10",
    "evidence_coverage_at_1",
    "evidence_coverage_at_3",
    "evidence_coverage_at_5",
    "evidence_coverage_at_10",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "correct_work_at_1",
    "correct_work_at_3",
    "correct_work_at_5",
    "correct_work_at_10",
    "retrieval_latency_ms",
    "retrieval_core_latency_ms",
    "qdrant_latency_ms",
    "rerank_latency_ms",
]

SUMMARY_COLUMNS = [
    "config_id",
    "experiment_id",
    "chunk_strategy",
    "chunk_size",
    "overlap",
    "retrieval_strategy",
    "embedder",
    "query_variant",
    "queries",
    "evidence_coverage_at_1",
    "evidence_coverage_at_3",
    "evidence_coverage_at_5",
    "evidence_coverage_at_10",
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_10",
    "correct_work_at_1",
    "correct_work_at_3",
    "correct_work_at_5",
    "correct_work_at_10",
    "mrr_at_10",
    "ndcg_at_10",
    "coverage_auc_at_10",
    "retrieval_latency_ms_mean",
    "retrieval_latency_ms_p50",
    "retrieval_latency_ms_p95",
    "qdrant_latency_ms_mean",
    "qdrant_latency_ms_p50",
    "qdrant_latency_ms_p95",
    "rerank_latency_ms_mean",
    "rerank_latency_ms_p50",
    "rerank_latency_ms_p95",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Gera um dashboard HTML self-contained para uma execução do "
            "benchmark de retrieval armazenada no RustFS."
        )
    )
    parser.add_argument(
        "--run-id",
        required=True,
        help="Ex.: retrieval_dev_20261001T042919Z",
    )
    parser.add_argument(
        "--holdout-run-id",
        default=None,
        help=(
            "Run HOLDOUT opcional. Quando informado, gera um único HTML com "
            "GERAL / DEV / HOLDOUT. Sem ele, HOLDOUT fica desabilitado."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=("HTML local. Padrão: reports/retrieval_<run-id>.html"),
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Não envia o HTML gerado de volta ao RustFS.",
    )
    parser.add_argument(
        "--open",
        action="store_true",
        dest="open_browser",
        help="Abre o HTML no navegador após gerar.",
    )
    parser.add_argument(
        "--bootstrap-runs",
        type=int,
        default=2000,
        help=(
            "Reamostragens por query_id para IC95%% dos efeitos emparelhados. "
            "Padrão: 2000."
        ),
    )
    return parser.parse_args()


def read_parquet(storage: RustFSManager, bucket: str, key: str) -> pd.DataFrame:
    payload = storage.download_bytes(bucket, key)
    return pd.read_parquet(io.BytesIO(payload))


def json_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def records(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    present = [column for column in columns if column in frame.columns]
    output: list[dict[str, Any]] = []
    for row in frame[present].to_dict(orient="records"):
        output.append({key: json_scalar(value) for key, value in row.items()})
    return output


def safe_script_json(value: Any) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


METRIC_SPECS = {
    "hit_at_10": {"label": "Hit@10", "type": "pct"},
    "hit_at_5": {"label": "Hit@5", "type": "pct"},
    "hit_at_3": {"label": "Hit@3", "type": "pct"},
    "evidence_coverage_at_10": {"label": "Evidence Coverage@10", "type": "pct"},
    "evidence_coverage_at_5": {"label": "Evidence Coverage@5", "type": "pct"},
    "mrr_at_10": {"label": "MRR@10", "type": "num"},
    "ndcg_at_10": {"label": "nDCG@10", "type": "num"},
    "coverage_auc_at_10": {"label": "Coverage AUC@10", "type": "num"},
    "correct_work_at_10": {"label": "Correct work@10", "type": "pct"},
    "correct_work_at_1": {"label": "Correct work@1", "type": "pct"},
}

# A seleção final não mistura valores brutos de métricas com escalas diferentes.
# Cada métrica vira um percentil de rank (1 = melhor, 0 = pior) entre todas as
# configurações. Quatro famílias recebem o mesmo peso para evitar redundância.
CONSENSUS_FAMILIES = {
    "hits": ["hit_at_10", "hit_at_5", "hit_at_3"],
    "coverage": [
        "evidence_coverage_at_10",
        "evidence_coverage_at_5",
        "coverage_auc_at_10",
    ],
    "ranking": ["mrr_at_10", "ndcg_at_10"],
    "work": ["correct_work_at_10", "correct_work_at_1"],
}
FINAL_SELECTION_SIZE = 10


def _config_label_from_row(row: pd.Series | dict[str, Any]) -> str:
    get = row.get
    strategy = str(get("chunk_strategy", ""))
    chunk_size = str(get("chunk_size", ""))
    retrieval = str(get("retrieval_strategy", ""))
    embedder = get("embedder")
    if embedder is None or (isinstance(embedder, float) and pd.isna(embedder)):
        embedder = ""
    suffix = f" · {embedder}" if str(embedder).strip() else ""
    return f"{strategy}/{chunk_size} · {retrieval}{suffix}"


def _bootstrap_cluster_mean(
    paired: pd.DataFrame,
    *,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    if paired.empty:
        return {
            "pairs": 0,
            "query_ids": 0,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "positive_pct": None,
            "negative_pct": None,
            "tie_pct": None,
            "direction": "sem_dados",
        }

    values = pd.to_numeric(paired["delta"], errors="coerce")
    valid = paired.loc[values.notna()].copy()
    valid["delta"] = values.loc[values.notna()].astype(float)
    if valid.empty:
        return {
            "pairs": 0,
            "query_ids": 0,
            "delta": None,
            "ci_low": None,
            "ci_high": None,
            "positive_pct": None,
            "negative_pct": None,
            "tie_pct": None,
            "direction": "sem_dados",
        }

    clustered = valid.groupby("query_id", as_index=False)["delta"].mean()
    cluster_values = clustered["delta"].to_numpy(dtype=float)
    delta = float(cluster_values.mean())

    if iterations > 0 and len(cluster_values) > 1:
        import numpy as np

        rng = np.random.default_rng(seed)
        sampled = rng.choice(
            cluster_values,
            size=(iterations, len(cluster_values)),
            replace=True,
        )
        means = sampled.mean(axis=1)
        ci_low = float(np.quantile(means, 0.025))
        ci_high = float(np.quantile(means, 0.975))
    else:
        ci_low = delta
        ci_high = delta

    eps = 1e-12
    positives = int((valid["delta"] > eps).sum())
    negatives = int((valid["delta"] < -eps).sum())
    ties = int(len(valid) - positives - negatives)

    if ci_low > 0:
        direction = "right"
    elif ci_high < 0:
        direction = "left"
    else:
        direction = "inconclusive"

    return {
        "pairs": int(len(valid)),
        "query_ids": int(len(clustered)),
        "delta": delta,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "positive_pct": positives / len(valid),
        "negative_pct": negatives / len(valid),
        "tie_pct": ties / len(valid),
        "direction": direction,
    }


def _paired_factor(
    frame: pd.DataFrame,
    *,
    metric: str,
    name: str,
    left_label: str,
    right_label: str,
    left_mask: pd.Series,
    right_mask: pd.Series,
    match_columns: list[str],
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    columns = ["query_id", metric, *match_columns]
    left = frame.loc[left_mask, columns].copy()
    right = frame.loc[right_mask, columns].copy()

    left = left.rename(columns={metric: "left_value"})
    right = right.rename(columns={metric: "right_value"})
    paired = left.merge(
        right,
        on=["query_id", *match_columns],
        how="inner",
        validate="one_to_one",
    )
    # hit_at_* e correct_work_at_* chegam do Parquet como bool.
    # pd.to_numeric preserva dtype booleano; NumPy não permite bool - bool.
    # Convertemos explicitamente para float para que métricas binárias sejam
    # tratadas como 0.0/1.0 e o delta emparelhado continue significando
    # diferença de proporção/acerto.
    right_values = pd.to_numeric(paired["right_value"], errors="coerce").astype(float)
    left_values = pd.to_numeric(paired["left_value"], errors="coerce").astype(float)
    paired["delta"] = right_values - left_values
    result = _bootstrap_cluster_mean(
        paired[["query_id", "delta"]],
        iterations=iterations,
        seed=seed,
    )
    result.update(
        {
            "name": name,
            "left_label": left_label,
            "right_label": right_label,
            "delta_label": f"{right_label} − {left_label}",
        }
    )
    return result


def _stable_seed(text: str) -> int:
    import hashlib

    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def _pareto_frontier(configs: list[dict[str, Any]], metric: str) -> list[str]:
    valid = [
        row
        for row in configs
        if isinstance(row.get(metric), (int, float))
        and isinstance(row.get("latency_p95"), (int, float))
        and math.isfinite(float(row[metric]))
        and math.isfinite(float(row["latency_p95"]))
    ]
    frontier: list[str] = []
    for candidate in valid:
        dominated = False
        cq = float(candidate[metric])
        cl = float(candidate["latency_p95"])
        for other in valid:
            if other["config_id"] == candidate["config_id"]:
                continue
            oq = float(other[metric])
            ol = float(other["latency_p95"])
            if ol <= cl and oq >= cq and (ol < cl or oq > cq):
                dominated = True
                break
        if not dominated:
            frontier.append(str(candidate["config_id"]))
    return frontier


def _balanced_consensus_selection(
    configs: list[dict[str, Any]],
    *,
    size: int = FINAL_SELECTION_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Seleciona configurações por rank balanceado entre famílias de métricas.

    Cada métrica é convertida para um percentil de rank em [0, 1] entre TODAS
    as configurações. Empates recebem o rank médio. Em seguida, calculamos a
    média dentro de quatro famílias (hits, cobertura, ranking e acerto de obra)
    e damos o mesmo peso de 25% para cada família.

    A seleção não depende de frequência em Top 10. O desempate é determinístico:
    nDCG@10 desc, Coverage@10 desc, p95 asc e config_id.
    """
    metadata = {
        "name": "balanced_rank_score",
        "label": "Balanced Rank Score",
        "size": size,
        "families": CONSENSUS_FAMILIES,
        "family_weight": 1.0 / len(CONSENSUS_FAMILIES),
        "eligible_pool": "all_configs",
        "eligible_count": len(configs),
        "description": (
            "Média de percentis de rank com peso igual para hits, cobertura, "
            "ranking e acerto de obra. A seleção usa todas as configurações."
        ),
        "tie_break": [
            "ndcg_at_10_desc",
            "evidence_coverage_at_10_desc",
            "latency_p95_asc",
            "config_id_asc",
        ],
    }
    if not configs:
        return [], metadata

    frame = pd.DataFrame(configs).copy()
    n = len(frame)
    metric_percentiles: dict[str, pd.Series] = {}

    for metric in METRIC_SPECS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        ranks = values.rank(method="average", ascending=False, na_option="bottom")
        if n <= 1:
            percentiles = pd.Series(1.0, index=frame.index)
        else:
            percentiles = 1.0 - (ranks - 1.0) / float(n - 1)
        metric_percentiles[metric] = percentiles.clip(0.0, 1.0)
        frame[f"rank_pct__{metric}"] = metric_percentiles[metric]

    family_columns: list[str] = []
    for family, metrics in CONSENSUS_FAMILIES.items():
        col = f"family_score__{family}"
        frame[col] = pd.concat([metric_percentiles[m] for m in metrics], axis=1).mean(
            axis=1
        )
        family_columns.append(col)

    frame["consensus_score"] = frame[family_columns].mean(axis=1)

    output: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        config = dict(configs[int(row.name)])
        config["consensus_score"] = float(row["consensus_score"])
        config["family_scores"] = {
            family: float(row[f"family_score__{family}"])
            for family in CONSENSUS_FAMILIES
        }
        config["metric_rank_percentiles"] = {
            metric: float(row[f"rank_pct__{metric}"]) for metric in METRIC_SPECS
        }
        output.append(config)

    output.sort(
        key=lambda row: (
            -float(row["consensus_score"]),
            -float(row.get("ndcg_at_10", 0.0)),
            -float(row.get("evidence_coverage_at_10", 0.0)),
            float(row.get("latency_p95", float("inf"))),
            str(row["config_id"]),
        )
    )
    for rank, row in enumerate(output, start=1):
        row["consensus_rank"] = rank

    return output[:size], metadata


def build_automatic_insights(
    query_df: pd.DataFrame,
    *,
    bootstrap_runs: int,
) -> dict[str, Any]:
    frame = query_df.copy()
    frame["embedder_key"] = (
        frame.get("embedder", pd.Series(index=frame.index, dtype=object))
        .fillna("__none__")
        .astype(str)
    )

    configs: list[dict[str, Any]] = []
    for config_id, group in frame.groupby("config_id", sort=False):
        first = group.iloc[0]
        item: dict[str, Any] = {
            "config_id": str(config_id),
            "label": _config_label_from_row(first),
            "chunk_strategy": json_scalar(first.get("chunk_strategy")),
            "chunk_size": json_scalar(first.get("chunk_size")),
            "overlap": json_scalar(first.get("overlap")),
            "retrieval_strategy": json_scalar(first.get("retrieval_strategy")),
            "embedder": json_scalar(first.get("embedder")),
            "n": int(len(group)),
            "latency_mean": float(
                pd.to_numeric(group["retrieval_latency_ms"], errors="coerce").mean()
            ),
            "latency_p95": float(
                pd.to_numeric(group["retrieval_latency_ms"], errors="coerce").quantile(
                    0.95
                )
            ),
        }
        for metric in METRIC_SPECS:
            item[metric] = float(pd.to_numeric(group[metric], errors="coerce").mean())
        configs.append(item)

    factor_specs = [
        {
            "name": "ce_vs_dense",
            "left_label": "Dense",
            "right_label": "Dense + CrossEncoder",
            "left_mask": frame["retrieval_strategy"].eq("dense"),
            "right_mask": frame["retrieval_strategy"].eq("dense_cross_encoder"),
            "match": ["query_variant", "chunk_strategy", "chunk_size", "embedder_key"],
            "description": "Efeito do reranker mantendo query, chunking, tamanho e embedder iguais.",
        },
        {
            "name": "qwen_vs_colibri",
            "left_label": "Colibri",
            "right_label": "Qwen",
            "left_mask": frame["embedder_key"].eq("colibri")
            & frame["retrieval_strategy"].isin(["dense", "dense_cross_encoder"]),
            "right_mask": frame["embedder_key"].eq("qwen3_embedding_06b")
            & frame["retrieval_strategy"].isin(["dense", "dense_cross_encoder"]),
            "match": [
                "query_variant",
                "chunk_strategy",
                "chunk_size",
                "retrieval_strategy",
            ],
            "description": "Diferença entre embedders mantendo os demais fatores iguais.",
        },
        {
            "name": "hybrid_vs_fixed",
            "left_label": "Fixed",
            "right_label": "Hybrid",
            "left_mask": frame["chunk_strategy"].eq("fixed"),
            "right_mask": frame["chunk_strategy"].eq("hybrid"),
            "match": [
                "query_variant",
                "chunk_size",
                "retrieval_strategy",
                "embedder_key",
            ],
            "description": "Diferença entre estratégias de chunking no mesmo tamanho e retrieval.",
        },
        {
            "name": "rewritten_vs_manual",
            "left_label": "Manual",
            "right_label": "Rewritten",
            "left_mask": frame["query_variant"].eq("manual"),
            "right_mask": frame["query_variant"].eq("rewritten"),
            "match": ["config_id"],
            "description": "Efeito da reformulação da pergunta dentro da mesma configuração.",
        },
        {
            "name": "chunk512_vs_256",
            "left_label": "Chunk 256",
            "right_label": "Chunk 512",
            "left_mask": pd.to_numeric(frame["chunk_size"], errors="coerce").eq(256),
            "right_mask": pd.to_numeric(frame["chunk_size"], errors="coerce").eq(512),
            "match": [
                "query_variant",
                "chunk_strategy",
                "retrieval_strategy",
                "embedder_key",
            ],
            "description": "Efeito de 512 contra 256 mantendo os demais fatores iguais.",
        },
        {
            "name": "chunk1024_vs_256",
            "left_label": "Chunk 256",
            "right_label": "Chunk 1024",
            "left_mask": pd.to_numeric(frame["chunk_size"], errors="coerce").eq(256),
            "right_mask": pd.to_numeric(frame["chunk_size"], errors="coerce").eq(1024),
            "match": [
                "query_variant",
                "chunk_strategy",
                "retrieval_strategy",
                "embedder_key",
            ],
            "description": "Efeito de 1024 contra 256 mantendo os demais fatores iguais.",
        },
    ]

    factors: dict[str, Any] = {}
    for metric in METRIC_SPECS:
        metric_results: list[dict[str, Any]] = []
        for spec in factor_specs:
            result = _paired_factor(
                frame,
                metric=metric,
                name=spec["name"],
                left_label=spec["left_label"],
                right_label=spec["right_label"],
                left_mask=spec["left_mask"],
                right_mask=spec["right_mask"],
                match_columns=spec["match"],
                iterations=bootstrap_runs,
                seed=_stable_seed(f"{metric}|{spec['name']}|{bootstrap_runs}"),
            )
            result["description"] = spec["description"]
            metric_results.append(result)
        factors[metric] = metric_results

    pareto: dict[str, list[str]] = {
        metric: _pareto_frontier(configs, metric) for metric in METRIC_SPECS
    }

    final_selection, selection_method = _balanced_consensus_selection(
        configs,
        size=FINAL_SELECTION_SIZE,
    )

    return {
        "metric_specs": METRIC_SPECS,
        "factors": factors,
        "pareto": pareto,
        "configs": configs,
        "selection_method": selection_method,
        "final_selection": final_selection,
        "bootstrap_runs": bootstrap_runs,
    }


def build_html(
    *,
    run_id: str,
    run_config: dict[str, Any],
    run_manifest: dict[str, Any],
    query_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    insights: dict[str, Any],
) -> str:
    query_payload = records(query_df, QUERY_COLUMNS)
    summary_payload = records(summary_df, SUMMARY_COLUMNS)

    meta = {
        "run_id": run_id,
        "split": run_config.get("split") or run_manifest.get("split"),
        "benchmark_sha256": run_config.get("benchmark_sha256"),
        "evidence_min_coverage": run_config.get("evidence_min_coverage"),
        "warmup_runs": run_config.get("warmup_runs"),
        "base_queries": run_config.get("base_queries"),
        "query_variants": run_config.get("query_variants"),
        "configs": run_config.get("configs"),
        "retrieval_executions": len(query_payload),
        "status": run_manifest.get("status"),
        "created_at_utc": run_config.get("created_at_utc"),
        "completed_at_utc": run_manifest.get("completed_at_utc"),
        "bm25_tokenizer": run_config.get("bm25_tokenizer"),
        "reranker": run_config.get("reranker"),
        "top_k": run_config.get("top_k"),
    }

    template = r"""<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACartomante · Retrieval Benchmark</title>
<style>
:root {
  --bg:#f6f7fb; --card:#ffffff; --ink:#182033; --muted:#697386;
  --line:#e5e8f0; --blue:#3157d5; --purple:#7652cc; --green:#18864b;
  --red:#c43d4b; --amber:#a96b00; --shadow:0 8px 24px rgba(30,40,70,.07);
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header{padding:28px 30px 22px;background:linear-gradient(120deg,#243e9f,#6842b8);color:white}
header h1{margin:0;font-size:26px}.subtitle{margin-top:7px;opacity:.86;font-size:13px}
main{max-width:1600px;margin:0 auto;padding:22px}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}.card{background:var(--card);border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:16px}.span-12{grid-column:span 12}.span-8{grid-column:span 8}.span-7{grid-column:span 7}.span-6{grid-column:span 6}.span-5{grid-column:span 5}.span-4{grid-column:span 4}.span-3{grid-column:span 3}
.kpi-label{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}.kpi{font-size:27px;font-weight:760;margin-top:5px}.kpi-note{font-size:12px;color:var(--muted);margin-top:3px}
h2{font-size:16px;margin:0 0 12px}.filters{display:grid;grid-template-columns:repeat(4,minmax(150px,1fr));gap:10px}.filter label{display:block;font-size:11px;color:var(--muted);margin-bottom:4px}.filter select,.filter input{width:100%;border:1px solid var(--line);border-radius:9px;padding:8px 9px;background:white;color:var(--ink)}button{border:0;border-radius:9px;padding:9px 12px;background:var(--blue);color:white;cursor:pointer}button.secondary{background:#eef1fa;color:#33415f}button:hover{filter:brightness(.97)}
.toolbar{display:flex;gap:8px;flex-wrap:wrap;align-items:end;margin-top:11px}.toolbar .metric{min-width:240px}
.chart{height:350px;position:relative}.chart-tall{height:390px}.insight-layout{display:grid;gap:14px;align-items:start}.insight-layout.two-thirds{grid-template-columns:minmax(0,2fr) minmax(360px,1fr)}.final-table-wrap{max-height:410px}.heatmap-wrap{max-height:560px}.selection-card{border-top:3px solid var(--purple)}.chart svg{width:100%;height:100%;overflow:visible}.axis{stroke:#ccd2df;stroke-width:1}.gridline{stroke:#edf0f6;stroke-width:1}.tick{fill:#7a8497;font-size:11px}.bar{fill:#4e67d8}.bar.alt{fill:#7652cc}.dot{fill:#4e67d8;opacity:.78}.dot:hover{opacity:1;stroke:#17233e;stroke-width:1.2}.chart-empty{display:flex;align-items:center;justify-content:center;height:100%;color:var(--muted)}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:11px;color:var(--muted);margin-top:4px}.legend span::before{content:"";display:inline-block;width:9px;height:9px;border-radius:3px;margin-right:5px;background:#4e67d8}.legend span.purple::before{background:#7652cc}
.table-wrap{overflow:auto;max-height:680px;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:separate;border-spacing:0;font-size:12px}th,td{padding:8px 9px;border-bottom:1px solid #edf0f5;text-align:right;white-space:nowrap}th{text-align:right;position:sticky;top:0;background:#f9fafc;z-index:2;cursor:pointer;color:#4f596e}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){text-align:left}.pill{display:inline-block;padding:3px 7px;border-radius:999px;background:#eff2fa;color:#42506d;font-size:11px}.good{color:var(--green)}.bad{color:var(--red)}
.note{font-size:12px;line-height:1.55;color:var(--muted)}.mini{font-size:11px;color:var(--muted)}
.query-table td.query{text-align:left;white-space:normal;min-width:320px;max-width:560px}.query-table td.metric{text-align:right}.query-table td.work{text-align:left}.delta-pos{color:var(--green);font-weight:650}.delta-neg{color:var(--red);font-weight:650}
.footer{padding:20px 0 35px;color:var(--muted);font-size:11px;text-align:center}

.tabs{display:flex;gap:8px;max-width:1600px;margin:16px auto 0;padding:0 22px}.tab-btn{background:#e9edf7;color:#34415f;border:1px solid var(--line)}.tab-btn.active{background:var(--blue);color:#fff}.tab-panel{display:none}.tab-panel.active{display:block}.insight-summary{margin-top:12px;padding:12px 14px;border-radius:10px;background:#f8f9fd;border:1px solid var(--line);font-size:12px;line-height:1.55;color:#4f596e}.insight-summary.compact{margin:0 0 10px}.insight-summary ul{margin:6px 0 0 18px;padding:0}.heat{font-variant-numeric:tabular-nums}.pareto-dot{fill:#18864b;opacity:.95;stroke:#0d5b31;stroke-width:1.4}.dominated-dot{fill:#aeb6c8;opacity:.55}.direction-right{color:var(--green);font-weight:700}.direction-left{color:var(--red);font-weight:700}.direction-inconclusive{color:var(--amber);font-weight:700}.metric-chip{display:inline-block;margin:2px 3px 2px 0;padding:2px 6px;border-radius:999px;background:#eef1fa;color:#44516f;font-size:10px}.nowrap{white-space:nowrap}
@media(max-width:1050px){.span-8,.span-7,.span-6,.span-5,.span-4,.span-3{grid-column:span 12}.filters{grid-template-columns:repeat(2,1fr)}.insight-layout.two-thirds{grid-template-columns:1fr}}
@media(max-width:640px){main{padding:12px}.filters{grid-template-columns:1fr}header{padding:22px 16px}}
</style>
</head>
<body>
<header>
  <h1>ACartomante · Retrieval Benchmark</h1>
  <div class="subtitle" id="runSubtitle"></div>
</header>
<nav class="tabs"><button class="tab-btn active" data-tab="tab-dashboard">Dashboard</button><button class="tab-btn" data-tab="tab-insights">Insights automáticos</button></nav>
<main>
<section id="tab-dashboard" class="tab-panel active">
  <div class="grid">
    <section class="card span-12">
      <h2>Filtros e métrica</h2>
      <div class="filters" id="filters"></div>
      <div class="toolbar">
        <div class="filter metric"><label>Métrica principal</label><select id="metricSelect"></select></div>
        <button id="clearBtn" class="secondary">Limpar filtros</button>
        <button id="exportBtn" class="secondary">Exportar tabela CSV</button>
      </div>
    </section>

    <section class="card span-3"><div class="kpi-label">Execuções filtradas</div><div class="kpi" id="kpiExec">—</div><div class="kpi-note" id="kpiExecNote"></div></section>
    <section class="card span-3"><div class="kpi-label">Média da métrica</div><div class="kpi" id="kpiMetric">—</div><div class="kpi-note" id="kpiMetricNote"></div></section>
    <section class="card span-3"><div class="kpi-label">Latência p95</div><div class="kpi" id="kpiLatency">—</div><div class="kpi-note">retrieval end-to-end do estágio</div></section>
    <section class="card span-3"><div class="kpi-label">Rewrite · delta médio</div><div class="kpi" id="kpiRewrite">—</div><div class="kpi-note" id="kpiRewriteNote"></div></section>

    <section class="card span-8"><h2>Configurações · métrica selecionada</h2><div class="chart" id="barChart"></div><div class="mini">Mostra até 15 configurações; filtros de obra/dificuldade/tipo recalculam os agregados a partir de query_metrics.</div></section>
    <section class="card span-4"><h2>Qualidade × latência</h2><div class="chart" id="scatterChart"></div><div class="mini">Eixo X: p95 de retrieval. Eixo Y: métrica selecionada. Passe o mouse sobre os pontos.</div></section>

    <section class="card span-6"><h2>Efeito do query rewrite</h2><div class="chart" id="rewriteChart"></div><div class="note" id="rewriteNote"></div></section>
    <section class="card span-6"><h2>Chunking · estratégia × tamanho</h2><div class="chart" id="chunkChart"></div><div class="mini">Agregação respeita os filtros atuais e combina os retrievals restantes.</div></section>

    <section class="card span-12">
      <h2>Tabela comparativa das configurações</h2>
      <div class="table-wrap"><table id="configTable"><thead></thead><tbody></tbody></table></div>
    </section>

    <section class="card span-12">
      <h2>Casos por query · drill-down</h2>
      <div class="toolbar" style="margin:0 0 10px">
        <div class="filter" style="min-width:320px;flex:1"><label>Buscar query / obra / id</label><input id="querySearch" type="search" placeholder="Ex.: Capitu, Dom Casmurro, UUID..."></div>
        <div class="filter" style="min-width:220px"><label>Configuração para inspeção</label><select id="queryConfig"></select></div>
      </div>
      <div class="table-wrap"><table class="query-table" id="queryTable"><thead></thead><tbody></tbody></table></div>
    </section>

    <section class="card span-12">
      <h2>Notas metodológicas</h2>
      <div class="note" id="methodNotes"></div>
    </section>
  </div>
</section>

<section id="tab-insights" class="tab-panel">
  <div class="grid">
    <section class="card span-12">
      <h2>Resumo da etapa de retrieval</h2>
      <div class="note">
        Esta aba resume apenas os resultados que ajudam na decisão: seleção determinística dos 10 pipelines, efeitos emparelhados dos fatores e trade-off qualidade × latência. Todo o cálculo usa somente o split DEV; o holdout permanece intocado.
      </div>
      <div id="autoInsightSummary" class="insight-summary"></div>
    </section>

    <section class="card span-3"><div class="kpi-label">Configurações avaliadas</div><div class="kpi" id="insightConfigs">—</div><div class="kpi-note">matriz completa de retrieval</div></section>
    <section class="card span-3"><div class="kpi-label">Selecionadas</div><div class="kpi" id="insightFinal">—</div><div class="kpi-note">Balanced Rank Score</div></section>
    <section class="card span-3"><div class="kpi-label">Efeitos conclusivos · nDCG</div><div class="kpi" id="insightEffects">—</div><div class="kpi-note">IC95% não cruza zero</div></section>
    <section class="card span-3"><div class="kpi-label">Pareto · nDCG@10</div><div class="kpi" id="insightPareto">—</div><div class="kpi-note">qualidade × latência p95</div></section>

    <section class="card span-12 selection-card">
      <h2>10 configurações para a próxima etapa</h2>
      <div class="note">
        <b>Balanced Rank Score</b>: cada métrica é transformada em percentil de rank entre as 30 configurações. As métricas são agrupadas em quatro famílias com peso igual — <b>Hits</b>, <b>Cobertura</b>, <b>Ranking</b> e <b>Acerto de obra</b>. A seleção usa todas as configurações; não depende de frequência em Top 10. Empates são resolvidos por nDCG@10, Coverage@10, p95 e <code>config_id</code>.
      </div>
      <div class="insight-layout two-thirds" style="margin-top:12px">
        <div>
          <div class="chart chart-tall" id="consensusChart"></div>
          <div class="mini">Score maior = melhor posição relativa de forma balanceada entre as quatro famílias.</div>
        </div>
        <div class="table-wrap final-table-wrap"><table id="finalSelectionTable"><thead></thead><tbody></tbody></table></div>
      </div>
    </section>

    <section class="card span-12">
      <h2>Efeitos emparelhados dos fatores</h2>
      <div class="note">Cada comparação altera um fator por vez. O ponto mostra o delta médio e a linha o IC95% por <code>query_id</code>; se o intervalo cruza zero, a diferença fica como inconclusiva no DEV.</div>
      <div class="toolbar" style="margin-bottom:10px"><div class="filter metric"><label>Métrica analisada</label><select id="factorMetric"></select></div></div>
      <div class="chart chart-tall" id="factorForest"></div>
      <div id="factorExplanation" class="insight-summary compact" style="margin-top:10px"></div>
      <div class="table-wrap" style="margin-top:12px"><table id="factorTable"><thead></thead><tbody></tbody></table></div>
    </section>

    <section class="card span-8">
      <h2>Fronteira de Pareto · qualidade × latência</h2>
      <div class="note">Uma configuração entra no Pareto quando nenhuma outra é simultaneamente tão boa ou melhor na métrica e tão rápida ou mais rápida no p95.</div>
      <div class="toolbar" style="margin-bottom:10px"><div class="filter metric"><label>Métrica de qualidade</label><select id="paretoMetric"></select></div></div>
      <div class="chart chart-tall" id="paretoChart"></div>
    </section>

    <section class="card span-4">
      <h2>Configurações no Pareto</h2>
      <div class="table-wrap"><table id="paretoTable"><thead></thead><tbody></tbody></table></div>
    </section>

    <section class="card span-12">
      <h2>Como ler esta aba</h2>
      <div class="note">
        <b>Balanced Rank Score</b> define de forma reprodutível quais 10 pipelines seguem para geração. <b>Efeitos emparelhados</b> mostram quais componentes apresentaram diferença consistente no DEV. <b>Pareto</b> mostra alternativas eficientes quando ganho de qualidade custa mais latência. A escolha é feita no DEV e o holdout continua reservado para validação final.
      </div>
    </section>
  </div>
</section>
  <div class="footer">Dashboard self-contained · sem dependências JS externas · artefato derivado do benchmark congelado no RustFS</div>
</main>
<script>
const META = __META__;
const ROWS = __ROWS__;
const SUMMARY = __SUMMARY__;
const INSIGHTS = __INSIGHTS__;

const METRICS = {
  hit_at_10:{label:"Hit@10", type:"pct", higher:true},
  hit_at_5:{label:"Hit@5", type:"pct", higher:true},
  hit_at_3:{label:"Hit@3", type:"pct", higher:true},
  evidence_coverage_at_10:{label:"Evidence Coverage@10", type:"pct", higher:true},
  evidence_coverage_at_5:{label:"Evidence Coverage@5", type:"pct", higher:true},
  mrr_at_10:{label:"MRR@10", type:"num", higher:true},
  ndcg_at_10:{label:"nDCG@10", type:"num", higher:true},
  coverage_auc_at_10:{label:"Coverage AUC@10", type:"num", higher:true},
  correct_work_at_10:{label:"Correct work@10", type:"pct", higher:true},
  correct_work_at_1:{label:"Correct work@1", type:"pct", higher:true}
};

const FILTERS = [
  ["query_variant","Query"],
  ["chunk_strategy","Chunk strategy"],
  ["chunk_size","Chunk size"],
  ["retrieval_strategy","Retrieval"],
  ["embedder","Embedder"],
  ["difficulty","Dificuldade"],
  ["query_type","Tipo"],
  ["canonical_work","Obra"]
];

const state = {filters:{}, metric:"__DEFAULT_METRIC__", sort:{key:"metric",dir:-1}, queryConfig:""};
const $ = id => document.getElementById(id);
const uniq = arr => [...new Set(arr.filter(v => v !== null && v !== undefined && String(v) !== ""))].sort((a,b)=>String(a).localeCompare(String(b),"pt-BR",{numeric:true}));
const avg = arr => { const x=arr.filter(Number.isFinite); return x.length?x.reduce((a,b)=>a+b,0)/x.length:0; };
const quantile = (arr,q) => { const x=arr.filter(Number.isFinite).sort((a,b)=>a-b); if(!x.length)return 0; const p=(x.length-1)*q, lo=Math.floor(p), hi=Math.ceil(p); return lo===hi?x[lo]:x[lo]+(x[hi]-x[lo])*(p-lo); };
const esc = s => String(s??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#039;"}[c]));
const fmtNum = (v,d=3) => Number.isFinite(v)?v.toFixed(d):"—";
const fmtPct = v => Number.isFinite(v)?(v*100).toFixed(1)+"%":"—";
const fmtMs = v => Number.isFinite(v)?(v<1000?v.toFixed(1)+" ms":(v/1000).toFixed(2)+" s"):"—";
const fmtMetric = (key,v) => METRICS[key]?.type==="pct"?fmtPct(v):fmtNum(v);

function buildFilters(){
  const box=$("filters");
  FILTERS.forEach(([key,label])=>{
    const wrap=document.createElement("div"); wrap.className="filter";
    const lab=document.createElement("label"); lab.textContent=label;
    const sel=document.createElement("select"); sel.dataset.key=key;
    const all=document.createElement("option"); all.value=""; all.textContent="Todos"; sel.appendChild(all);
    uniq(ROWS.map(r=>r[key])).forEach(v=>{ const op=document.createElement("option"); op.value=String(v); op.textContent=String(v); sel.appendChild(op); });
    sel.addEventListener("change",()=>{state.filters[key]=sel.value;render();});
    wrap.append(lab,sel); box.appendChild(wrap);
  });
  const ms=$("metricSelect"); Object.entries(METRICS).forEach(([key,m])=>{const op=document.createElement("option");op.value=key;op.textContent=m.label; if(key===state.metric)op.selected=true;ms.appendChild(op);});
  ms.addEventListener("change",()=>{state.metric=ms.value;render();});
  $("clearBtn").addEventListener("click",()=>{state.filters={};document.querySelectorAll("#filters select").forEach(x=>x.value="");render();});
  $("exportBtn").addEventListener("click",exportCSV);
  $("querySearch").addEventListener("input",renderQueryTable);
  $("queryConfig").addEventListener("change",()=>{state.queryConfig=$("queryConfig").value;renderQueryTable();});
}

function filteredRows(){
  return ROWS.filter(r=>Object.entries(state.filters).every(([k,v])=>!v || String(r[k]??"")===v));
}

function groupConfigs(rows){
  const map=new Map();
  rows.forEach(r=>{ if(!map.has(r.config_id))map.set(r.config_id,[]); map.get(r.config_id).push(r); });
  const out=[];
  for(const [config_id,g] of map){
    const first=g[0];
    const vals=g.map(r=>Number(r[state.metric])).filter(Number.isFinite);
    const lats=g.map(r=>Number(r.retrieval_latency_ms)).filter(Number.isFinite);
    out.push({
      config_id,
      experiment_id:first.experiment_id,
      chunk_strategy:first.chunk_strategy,
      chunk_size:first.chunk_size,
      overlap:first.overlap,
      retrieval_strategy:first.retrieval_strategy,
      embedder:first.embedder||"—",
      n:g.length,
      metric:avg(vals),
      hit10:avg(g.map(r=>Number(r.hit_at_10)).filter(Number.isFinite)),
      coverage10:avg(g.map(r=>Number(r.evidence_coverage_at_10)).filter(Number.isFinite)),
      mrr10:avg(g.map(r=>Number(r.mrr_at_10)).filter(Number.isFinite)),
      ndcg10:avg(g.map(r=>Number(r.ndcg_at_10)).filter(Number.isFinite)),
      correctWork10:avg(g.map(r=>Number(r.correct_work_at_10)).filter(Number.isFinite)),
      latencyMean:avg(lats), latencyP50:quantile(lats,.5), latencyP95:quantile(lats,.95)
    });
  }
  return out;
}

function configLabel(c){return `${c.chunk_strategy}/${c.chunk_size} · ${c.retrieval_strategy}${c.embedder&&c.embedder!=="—"?" · "+c.embedder:""}`;}

function updateKPIs(rows,groups){
  $("kpiExec").textContent=rows.length.toLocaleString("pt-BR");
  $("kpiExecNote").textContent=`${groups.length} configurações após filtros`;
  const m=avg(rows.map(r=>Number(r[state.metric])).filter(Number.isFinite));
  $("kpiMetric").textContent=fmtMetric(state.metric,m); $("kpiMetricNote").textContent=METRICS[state.metric].label;
  $("kpiLatency").textContent=fmtMs(quantile(rows.map(r=>Number(r.retrieval_latency_ms)).filter(Number.isFinite),.95));
  const d=rewriteStats(rows,state.metric); $("kpiRewrite").textContent=(d.pairs?((d.delta>=0?"+":"")+fmtMetric(state.metric,d.delta)):"—");
  $("kpiRewrite").className="kpi "+(d.delta>0?"good":d.delta<0?"bad":""); $("kpiRewriteNote").textContent=d.pairs?`${d.pairs} pares manual↔rewritten`:"sem pares com filtros atuais";
}

function svgBox(){return {w:900,h:320,l:62,r:20,t:18,b:70};}
function clearChart(id){const el=$(id);el.innerHTML="";return el;}
function emptyChart(id,msg="Sem dados para os filtros atuais"){clearChart(id).innerHTML=`<div class="chart-empty">${esc(msg)}</div>`;}
function makeSvg(id){const el=clearChart(id);const svg=document.createElementNS("http://www.w3.org/2000/svg","svg");svg.setAttribute("viewBox","0 0 900 320");el.appendChild(svg);return svg;}
function line(svg,x1,y1,x2,y2,cls="axis"){const e=document.createElementNS(svg.namespaceURI,"line");Object.entries({x1,y1,x2,y2,class:cls}).forEach(([k,v])=>e.setAttribute(k,v));svg.appendChild(e);return e;}
function textSvg(svg,x,y,s,cls="tick",anchor="middle"){const e=document.createElementNS(svg.namespaceURI,"text");e.setAttribute("x",x);e.setAttribute("y",y);e.setAttribute("class",cls);e.setAttribute("text-anchor",anchor);e.textContent=s;svg.appendChild(e);return e;}

function barChart(groups){
  if(!groups.length)return emptyChart("barChart");
  const svg=makeSvg("barChart"), b=svgBox(); const data=[...groups].sort((a,b)=>b.metric-a.metric).slice(0,15);
  const max=Math.max(...data.map(x=>x.metric),.0001); const innerW=b.w-b.l-b.r, innerH=b.h-b.t-b.b; const bw=innerW/data.length*.72, gap=innerW/data.length;
  for(let i=0;i<=5;i++){const y=b.t+innerH*(1-i/5);line(svg,b.l,y,b.w-b.r,y,"gridline");textSvg(svg,b.l-8,y+4,fmtMetric(state.metric,max*i/5),"tick","end");}
  line(svg,b.l,b.t,b.l,b.t+innerH); line(svg,b.l,b.t+innerH,b.w-b.r,b.t+innerH);
  data.forEach((d,i)=>{const h=innerH*d.metric/max,x=b.l+i*gap+(gap-bw)/2,y=b.t+innerH-h;const r=document.createElementNS(svg.namespaceURI,"rect");r.setAttribute("x",x);r.setAttribute("y",y);r.setAttribute("width",bw);r.setAttribute("height",Math.max(1,h));r.setAttribute("rx",3);r.setAttribute("class",d.query_variant==="rewritten"?"bar alt":"bar");r.innerHTML=`<title>${esc(configLabel(d))}\n${METRICS[state.metric].label}: ${fmtMetric(state.metric,d.metric)}\np95: ${fmtMs(d.latencyP95)}</title>`;svg.appendChild(r); const label=(d.chunk_strategy[0]+d.chunk_size+"/"+d.retrieval_strategy.replace("dense_cross_encoder","D+CE").replace("dense","D").replace("bm25","B")+(d.embedder&&d.embedder!=="—"?"/"+(d.embedder.startsWith("qwen")?"Q":"C"):""));const tx=textSvg(svg,x+bw/2,b.t+innerH+16,label,"tick");tx.setAttribute("transform",`rotate(55 ${x+bw/2} ${b.t+innerH+16})`);tx.setAttribute("text-anchor","start");});
}

function scatterChart(groups){
  const data=groups.filter(d=>Number.isFinite(d.latencyP95)&&Number.isFinite(d.metric)); if(!data.length)return emptyChart("scatterChart");
  const svg=makeSvg("scatterChart"), b=svgBox(), innerW=b.w-b.l-b.r, innerH=b.h-b.t-b.b; let xmin=Math.min(...data.map(d=>d.latencyP95)),xmax=Math.max(...data.map(d=>d.latencyP95)),ymin=Math.min(...data.map(d=>d.metric)),ymax=Math.max(...data.map(d=>d.metric)); if(xmin===xmax){xmin=0;xmax+=1} if(ymin===ymax){ymin=Math.max(0,ymin-.1);ymax+=.1}
  for(let i=0;i<=5;i++){const yy=b.t+innerH*(1-i/5),yv=ymin+(ymax-ymin)*i/5;line(svg,b.l,yy,b.w-b.r,yy,"gridline");textSvg(svg,b.l-8,yy+4,fmtMetric(state.metric,yv),"tick","end");const xx=b.l+innerW*i/5,xv=xmin+(xmax-xmin)*i/5;line(svg,xx,b.t,xx,b.t+innerH,"gridline");textSvg(svg,xx,b.t+innerH+18,fmtMs(xv),"tick");}
  data.forEach(d=>{const x=b.l+(d.latencyP95-xmin)/(xmax-xmin)*innerW,y=b.t+innerH-(d.metric-ymin)/(ymax-ymin)*innerH;const c=document.createElementNS(svg.namespaceURI,"circle");c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",6);c.setAttribute("class","dot");c.innerHTML=`<title>${esc(configLabel(d))}\n${METRICS[state.metric].label}: ${fmtMetric(state.metric,d.metric)}\np95: ${fmtMs(d.latencyP95)}</title>`;svg.appendChild(c);});
}

function rewriteStats(rows,metric){
  const m=new Map(); rows.forEach(r=>{const k=r.config_id+"|"+r.query_id;if(!m.has(k))m.set(k,{});m.get(k)[r.query_variant]=Number(r[metric]);}); const deltas=[];let improved=0,regressed=0,same=0,missHit=0,hitMiss=0;
  for(const p of m.values()){if(!Number.isFinite(p.manual)||!Number.isFinite(p.rewritten))continue;const d=p.rewritten-p.manual;deltas.push(d);if(d>1e-12)improved++;else if(d<-1e-12)regressed++;else same++; if(metric.startsWith("hit_at_")){if(p.manual<.5&&p.rewritten>=.5)missHit++;if(p.manual>=.5&&p.rewritten<.5)hitMiss++;}}
  return {pairs:deltas.length,delta:avg(deltas),improved,regressed,same,missHit,hitMiss};
}

function rewriteChart(rows){
  const map=new Map(); rows.forEach(r=>{if(!map.has(r.config_id))map.set(r.config_id,[]);map.get(r.config_id).push(r);}); const data=[]; for(const [id,g] of map){const s=rewriteStats(g,state.metric);if(s.pairs)data.push({id,label:configLabel(groupConfigs(g)[0]||{}),...s});} data.sort((a,b)=>b.delta-a.delta); if(!data.length)return emptyChart("rewriteChart","Selecione ambos query variants para comparar manual × rewritten.");
  const svg=makeSvg("rewriteChart"), b=svgBox(), innerW=b.w-b.l-b.r, innerH=b.h-b.t-b.b;const shown=data.slice(0,15);const abs=Math.max(...shown.map(d=>Math.abs(d.delta)),.0001);const zero=b.t+innerH/2;line(svg,b.l,zero,b.w-b.r,zero,"axis"); textSvg(svg,b.l-8,zero+4,"0","tick","end"); const gap=innerW/shown.length,bw=gap*.66;
  shown.forEach((d,i)=>{const mag=Math.abs(d.delta)/abs*(innerH/2-8),x=b.l+i*gap+(gap-bw)/2,y=d.delta>=0?zero-mag:zero;const r=document.createElementNS(svg.namespaceURI,"rect");r.setAttribute("x",x);r.setAttribute("y",y);r.setAttribute("width",bw);r.setAttribute("height",Math.max(1,mag));r.setAttribute("rx",3);r.setAttribute("class",d.delta>=0?"bar":"bar alt");r.innerHTML=`<title>${esc(d.label)}\nDelta: ${fmtMetric(state.metric,d.delta)}\nmelhorou=${d.improved} piorou=${d.regressed}</title>`;svg.appendChild(r);});
  const total=rewriteStats(rows,state.metric); $("rewriteNote").innerHTML=`Pares: <b>${total.pairs}</b> · melhora: <b>${total.improved}</b> · piora: <b>${total.regressed}</b> · igual: <b>${total.same}</b>`+(state.metric.startsWith("hit_at_")?` · miss→hit: <b class="good">${total.missHit}</b> · hit→miss: <b class="bad">${total.hitMiss}</b>`:"");
}

function chunkChart(rows){
  const m=new Map(); rows.forEach(r=>{const k=r.chunk_strategy+"|"+r.chunk_size;if(!m.has(k))m.set(k,[]);m.get(k).push(Number(r[state.metric]));});const data=[...m.entries()].map(([k,v])=>{const [s,z]=k.split("|");return {strategy:s,size:Number(z),value:avg(v.filter(Number.isFinite))};}).sort((a,b)=>a.size-b.size||a.strategy.localeCompare(b.strategy)); if(!data.length)return emptyChart("chunkChart");
  const svg=makeSvg("chunkChart"),b=svgBox(),innerW=b.w-b.l-b.r,innerH=b.h-b.t-b.b,max=Math.max(...data.map(x=>x.value),.0001);const sizes=uniq(data.map(x=>x.size)),groupW=innerW/sizes.length,bw=groupW*.28;
  for(let i=0;i<=5;i++){const y=b.t+innerH*(1-i/5);line(svg,b.l,y,b.w-b.r,y,"gridline");textSvg(svg,b.l-8,y+4,fmtMetric(state.metric,max*i/5),"tick","end");}
  sizes.forEach((size,si)=>{const items=data.filter(d=>d.size===size);items.forEach((d,j)=>{const x=b.l+si*groupW+groupW*.5+(j-.5)*bw,y=b.t+innerH-innerH*d.value/max,h=innerH*d.value/max;const r=document.createElementNS(svg.namespaceURI,"rect");r.setAttribute("x",x-bw/2);r.setAttribute("y",y);r.setAttribute("width",bw*.88);r.setAttribute("height",Math.max(1,h));r.setAttribute("rx",3);r.setAttribute("class",d.strategy==="hybrid"?"bar alt":"bar");r.innerHTML=`<title>${d.strategy}/${size}: ${fmtMetric(state.metric,d.value)}</title>`;svg.appendChild(r);});textSvg(svg,b.l+si*groupW+groupW*.5,b.t+innerH+20,String(size),"tick");});
  line(svg,b.l,b.t+innerH,b.w-b.r,b.t+innerH); const legend=document.createElementNS(svg.namespaceURI,"text");legend.setAttribute("x",b.w-20);legend.setAttribute("y",15);legend.setAttribute("text-anchor","end");legend.setAttribute("class","tick");legend.textContent="azul=fixed · roxo=hybrid";svg.appendChild(legend);
}

const TABLE_COLS=[
  ["config_id","Config"],["n","N"],["metric","Métrica"],["hit10","Hit@10"],["coverage10","Coverage@10"],["mrr10","MRR@10"],["ndcg10","nDCG@10"],["correctWork10","Work@10"],["latencyMean","Lat. média"],["latencyP95","Lat. p95"]
];
function renderConfigTable(groups){
  const table=$("configTable"),head=table.querySelector("thead"),body=table.querySelector("tbody"); head.innerHTML="<tr>"+TABLE_COLS.map(([k,l])=>`<th data-key="${k}">${esc(l)}${state.sort.key===k?(state.sort.dir>0?" ▲":" ▼"):""}</th>`).join("")+"</tr>";
  head.querySelectorAll("th").forEach(th=>th.onclick=()=>{const k=th.dataset.key;if(state.sort.key===k)state.sort.dir*=-1;else{state.sort.key=k;state.sort.dir=k==="config_id"?1:-1;}render();});
  const data=[...groups].sort((a,b)=>{let x=a[state.sort.key],y=b[state.sort.key];if(typeof x==="string")return state.sort.dir*x.localeCompare(y,"pt-BR",{numeric:true});return state.sort.dir*((x??0)-(y??0));});
  body.innerHTML=data.map(d=>`<tr><td><span class="pill">${esc(configLabel(d))}</span></td><td>${d.n}</td><td><b>${fmtMetric(state.metric,d.metric)}</b></td><td>${fmtPct(d.hit10)}</td><td>${fmtPct(d.coverage10)}</td><td>${fmtNum(d.mrr10)}</td><td>${fmtNum(d.ndcg10)}</td><td>${fmtPct(d.correctWork10)}</td><td>${fmtMs(d.latencyMean)}</td><td>${fmtMs(d.latencyP95)}</td></tr>`).join("");
}

function syncQueryConfigs(groups){
  const sel=$("queryConfig"),prev=state.queryConfig;sel.innerHTML="";groups.sort((a,b)=>configLabel(a).localeCompare(configLabel(b),"pt-BR",{numeric:true})).forEach(g=>{const op=document.createElement("option");op.value=g.config_id;op.textContent=configLabel(g);sel.appendChild(op);});if(groups.some(g=>g.config_id===prev))sel.value=prev;else if(groups.length){sel.value=groups[0].config_id;state.queryConfig=groups[0].config_id;}
}

function renderQueryTable(){
  const rows=filteredRows().filter(r=>!state.queryConfig||r.config_id===state.queryConfig);const q=$("querySearch").value.trim().toLowerCase();const data=rows.filter(r=>!q||[r.query_id,r.query,r.canonical_work].some(v=>String(v??"").toLowerCase().includes(q))).sort((a,b)=>Number(a[state.metric])-Number(b[state.metric])).slice(0,250);
  const table=$("queryTable"),head=table.querySelector("thead"),body=table.querySelector("tbody");head.innerHTML="<tr><th>Query</th><th>Variante</th><th>Obra</th><th>Dific.</th><th>Tipo</th><th>1st hit</th><th>Hit@10</th><th>Coverage@10</th><th>MRR</th><th>nDCG</th><th>Latência</th></tr>";
  body.innerHTML=data.map(r=>`<tr><td class="query"><div>${esc(r.query)}</div><div class="mini">${esc(r.query_id)}</div></td><td>${esc(r.query_variant)}</td><td class="work">${esc(r.canonical_work)}</td><td>${esc(r.difficulty)}</td><td>${esc(r.query_type)}</td><td>${r.first_hit_rank??"—"}</td><td>${fmtPct(Number(r.hit_at_10))}</td><td>${fmtPct(Number(r.evidence_coverage_at_10))}</td><td>${fmtNum(Number(r.mrr_at_10))}</td><td>${fmtNum(Number(r.ndcg_at_10))}</td><td>${fmtMs(Number(r.retrieval_latency_ms))}</td></tr>`).join("");
}

function exportCSV(){
  const groups=groupConfigs(filteredRows());const cols=TABLE_COLS.map(x=>x[0]);const header=TABLE_COLS.map(x=>x[1]).join(",");const lines=[header,...groups.map(d=>cols.map(k=>`"${String(d[k]??"").replaceAll('"','""')}"`).join(","))];const blob=new Blob([lines.join("\n")],{type:"text/csv;charset=utf-8"});const a=document.createElement("a");a.href=URL.createObjectURL(blob);a.download=`${META.run_id}_filtered_summary.csv`;a.click();URL.revokeObjectURL(a.href);
}

function fmtDelta(metric,v){
  if(!Number.isFinite(v))return "—";
  const sign=v>0?"+":"";
  if(METRICS[metric]?.type==="pct")return `${sign}${(v*100).toFixed(2)} pp`;
  return `${sign}${v.toFixed(4)}`;
}

function directionText(item){
  if(!item || !item.pairs)return "sem pares";
  if(item.direction==="right")return `${item.right_label} favorecido`;
  if(item.direction==="left")return `${item.left_label} favorecido`;
  return "inconclusivo";
}

function directionClass(item){
  if(!item)return "";
  if(item.direction==="right")return "direction-right";
  if(item.direction==="left")return "direction-left";
  return "direction-inconclusive";
}

function setupTabs(){
  document.querySelectorAll(".tab-btn").forEach(btn=>{
    btn.addEventListener("click",()=>{
      document.querySelectorAll(".tab-btn").forEach(x=>x.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(x=>x.classList.remove("active"));
      btn.classList.add("active");
      const target=$(btn.dataset.tab); if(target)target.classList.add("active");
    });
  });
}

function fillMetricSelect(id,initial="ndcg_at_10"){
  const sel=$(id); sel.innerHTML="";
  Object.entries(METRICS).forEach(([key,m])=>{
    const op=document.createElement("option");op.value=key;op.textContent=m.label;if(key===initial)op.selected=true;sel.appendChild(op);
  });
  return sel;
}

function insightConfigLabel(row){
  return `${row.chunk_strategy}/${row.chunk_size} · ${row.retrieval_strategy}${row.embedder?" · "+row.embedder:""}`;
}

function renderAutomaticSummary(){
  const finalSet=INSIGHTS.final_selection||[];
  const factors=(INSIGHTS.factors||{}).ndcg_at_10||[];
  const conclusive=factors.filter(x=>x.direction!=="inconclusive"&&x.direction!=="sem_dados");
  const inconclusive=factors.filter(x=>x.direction==="inconclusive");
  const lines=[];
  if(finalSet.length){
    lines.push(`O <b>Balanced Rank Score</b> selecionou <b>${finalSet.length}</b> pipelines entre <b>${(INSIGHTS.configs||[]).length}</b>. O primeiro é <b>${esc(finalSet[0].label)}</b> com score ${Number(finalSet[0].consensus_score).toFixed(3)}.`);
  }
  if(conclusive.length){
    lines.push(`No nDCG@10, os efeitos com IC95% sem cruzar zero foram: ${conclusive.map(x=>`<b>${esc(x.delta_label)}</b> ${fmtDelta("ndcg_at_10",Number(x.delta))} [${fmtDelta("ndcg_at_10",Number(x.ci_low))}, ${fmtDelta("ndcg_at_10",Number(x.ci_high))}]`).join("; ")}.`);
  }
  if(inconclusive.length){
    lines.push(`Sem diferença conclusiva no nDCG@10: ${inconclusive.map(x=>`<b>${esc(x.delta_label)}</b>`).join(", ")}.`);
  }
  lines.push(`A fronteira de Pareto de nDCG@10 contém <b>${(INSIGHTS.pareto?.ndcg_at_10||[]).length}</b> configurações não dominadas em qualidade × p95.`);
  $("autoInsightSummary").innerHTML=lines.map(x=>`<p>${x}</p>`).join("");
}

function renderInsightKPIs(){
  $("insightConfigs").textContent=String((INSIGHTS.configs||[]).length);
  $("insightFinal").textContent=String((INSIGHTS.final_selection||[]).length);
  const factors=(INSIGHTS.factors||{}).ndcg_at_10||[];
  const conclusive=factors.filter(x=>x.direction!=="inconclusive"&&x.direction!=="sem_dados").length;
  $("insightEffects").textContent=String(conclusive);
  $("insightPareto").textContent=String((INSIGHTS.pareto?.ndcg_at_10||[]).length);
}

function consensusShortLabel(row){
  const r=String(row.retrieval_strategy||"").replace("dense_cross_encoder","D+CE").replace("dense","D").replace("bm25","BM25");
  const e=row.embedder?(String(row.embedder).startsWith("qwen")?"Qwen":"Colibri"):"";
  return `${row.chunk_strategy}/${row.chunk_size} · ${r}${e?" · "+e:""}`;
}

function renderFinalSelection(){
  const rows=INSIGHTS.final_selection||[];
  const table=$("finalSelectionTable"),head=table.querySelector("thead"),body=table.querySelector("tbody");
  head.innerHTML="<tr><th>#</th><th>Configuração</th><th>Score</th><th>nDCG</th><th>Coverage@10</th><th>p95</th></tr>";
  body.innerHTML=rows.map((r,i)=>`<tr><td>${i+1}</td><td style="text-align:left;white-space:normal">${esc(r.label)}</td><td><b>${Number(r.consensus_score).toFixed(3)}</b></td><td>${fmtNum(Number(r.ndcg_at_10))}</td><td>${fmtPct(Number(r.evidence_coverage_at_10))}</td><td>${fmtMs(Number(r.latency_p95))}</td></tr>`).join("");

  if(!rows.length){emptyChart("consensusChart","Sem configurações para seleção.");return;}
  const svg=makeSvg("consensusChart"),b={w:900,h:320,l:230,r:35,t:16,b:26};
  const innerW=b.w-b.l-b.r, innerH=b.h-b.t-b.b, gap=innerH/rows.length, bh=Math.min(20,gap*.62);
  const min=Math.min(...rows.map(r=>Number(r.consensus_score))), max=Math.max(...rows.map(r=>Number(r.consensus_score)));
  const lo=Math.max(0,min-.04), hi=Math.min(1,Math.max(max+.015,lo+.05));
  for(let i=0;i<=4;i++){const x=b.l+innerW*i/4;line(svg,x,b.t,x,b.t+innerH,"gridline");textSvg(svg,x,b.h-5,(lo+(hi-lo)*i/4).toFixed(2),"tick");}
  rows.forEach((r,i)=>{
    const y=b.t+i*gap+gap/2; const score=Number(r.consensus_score); const width=Math.max(1,(score-lo)/(hi-lo)*innerW);
    const rect=document.createElementNS(svg.namespaceURI,"rect");rect.setAttribute("x",b.l);rect.setAttribute("y",y-bh/2);rect.setAttribute("width",width);rect.setAttribute("height",bh);rect.setAttribute("rx",4);rect.setAttribute("class","bar");rect.innerHTML=`<title>#${i+1} ${esc(r.label)}
Balanced Rank Score: ${score.toFixed(4)}
nDCG@10: ${Number(r.ndcg_at_10).toFixed(4)}
Coverage@10: ${fmtPct(Number(r.evidence_coverage_at_10))}
p95: ${fmtMs(Number(r.latency_p95))}</title>`;svg.appendChild(rect);
    textSvg(svg,b.l-8,y+4,`${i+1}. ${consensusShortLabel(r)}`,"tick","end");
    textSvg(svg,b.l+width+7,y+4,score.toFixed(3),"tick","start");
  });
}

function factorShort(name){
  const map={ce_vs_dense:"CE − Dense",qwen_vs_colibri:"Qwen − Colibri",hybrid_vs_fixed:"Hybrid − Fixed",rewritten_vs_manual:"Rewrite − Manual",chunk512_vs_256:"512 − 256",chunk1024_vs_256:"1024 − 256"};
  return map[name]||name;
}

function renderFactorForest(metric){
  const rows=(INSIGHTS.factors?.[metric]||[]).filter(x=>Number.isFinite(Number(x.delta))&&Number.isFinite(Number(x.ci_low))&&Number.isFinite(Number(x.ci_high)));
  if(!rows.length)return emptyChart("factorForest");
  const svg=makeSvg("factorForest"),b={w:900,h:320,l:185,r:55,t:20,b:45},innerW=b.w-b.l-b.r,innerH=b.h-b.t-b.b;
  let xmin=Math.min(0,...rows.map(x=>Number(x.ci_low))),xmax=Math.max(0,...rows.map(x=>Number(x.ci_high)));
  const pad=Math.max((xmax-xmin)*.12,.005);xmin-=pad;xmax+=pad;if(xmin===xmax){xmin-=.01;xmax+=.01}
  const xPos=v=>b.l+(v-xmin)/(xmax-xmin)*innerW;
  for(let i=0;i<=4;i++){const v=xmin+(xmax-xmin)*i/4,x=xPos(v);line(svg,x,b.t,x,b.t+innerH,"gridline");textSvg(svg,x,b.h-8,fmtDelta(metric,v),"tick");}
  const zero=xPos(0);line(svg,zero,b.t,zero,b.t+innerH,"axis");
  const gap=innerH/rows.length;
  rows.forEach((r,i)=>{const y=b.t+gap*(i+.5),lo=xPos(Number(r.ci_low)),hi=xPos(Number(r.ci_high)),mid=xPos(Number(r.delta));line(svg,lo,y,hi,y,"axis");line(svg,lo,y-5,lo,y+5,"axis");line(svg,hi,y-5,hi,y+5,"axis");const c=document.createElementNS(svg.namespaceURI,"circle");c.setAttribute("cx",mid);c.setAttribute("cy",y);c.setAttribute("r",6);c.setAttribute("class",r.direction==="inconclusive"?"dominated-dot":"pareto-dot");c.innerHTML=`<title>${esc(r.delta_label)}\nDelta ${fmtDelta(metric,Number(r.delta))}\nIC95% [${fmtDelta(metric,Number(r.ci_low))}, ${fmtDelta(metric,Number(r.ci_high))}]</title>`;svg.appendChild(c);textSvg(svg,b.l-10,y+4,factorShort(r.name),"tick","end");});
}

function renderFactors(metric){
  const rows=INSIGHTS.factors?.[metric]||[];const table=$("factorTable"),head=table.querySelector("thead"),body=table.querySelector("tbody");
  head.innerHTML="<tr><th>Comparação</th><th>Delta médio</th><th>IC95%</th><th>Direita vence</th><th>Esquerda vence</th><th>Pares</th><th>Queries</th><th>Leitura</th></tr>";
  body.innerHTML=rows.map(x=>`<tr><td>${esc(x.delta_label)}</td><td><b>${fmtDelta(metric,Number(x.delta))}</b></td><td>[${fmtDelta(metric,Number(x.ci_low))}, ${fmtDelta(metric,Number(x.ci_high))}]</td><td>${fmtPct(Number(x.positive_pct))}</td><td>${fmtPct(Number(x.negative_pct))}</td><td>${x.pairs}</td><td>${x.query_ids}</td><td><span class="${directionClass(x)}">${esc(directionText(x))}</span></td></tr>`).join("");
  $("factorExplanation").innerHTML=`<b>${esc(METRICS[metric].label)}</b> · ${INSIGHTS.bootstrap_runs} reamostragens por query_id.<ul>${rows.map(x=>`<li><b>${esc(x.delta_label)}</b>: ${esc(x.description)}</li>`).join("")}</ul>`;
  renderFactorForest(metric);
}

function renderPareto(metric){
  const configs=INSIGHTS.configs||[];const frontier=new Set(INSIGHTS.pareto?.[metric]||[]);const data=configs.filter(x=>Number.isFinite(Number(x[metric]))&&Number.isFinite(Number(x.latency_p95)));
  if(!data.length){emptyChart("paretoChart");return;}
  const svg=makeSvg("paretoChart"),b=svgBox(),innerW=b.w-b.l-b.r,innerH=b.h-b.t-b.b;let xmin=Math.min(...data.map(x=>Number(x.latency_p95))),xmax=Math.max(...data.map(x=>Number(x.latency_p95))),ymin=Math.min(...data.map(x=>Number(x[metric]))),ymax=Math.max(...data.map(x=>Number(x[metric])));if(xmin===xmax){xmin=0;xmax+=1}if(ymin===ymax){ymin=Math.max(0,ymin-.1);ymax+=.1}
  for(let i=0;i<=5;i++){const yy=b.t+innerH*(1-i/5),yv=ymin+(ymax-ymin)*i/5;line(svg,b.l,yy,b.w-b.r,yy,"gridline");textSvg(svg,b.l-8,yy+4,fmtMetric(metric,yv),"tick","end");const xx=b.l+innerW*i/5,xv=xmin+(xmax-xmin)*i/5;line(svg,xx,b.t,xx,b.t+innerH,"gridline");textSvg(svg,xx,b.t+innerH+18,fmtMs(xv),"tick");}
  data.forEach(d=>{const x=b.l+(Number(d.latency_p95)-xmin)/(xmax-xmin)*innerW,y=b.t+innerH-(Number(d[metric])-ymin)/(ymax-ymin)*innerH;const c=document.createElementNS(svg.namespaceURI,"circle");c.setAttribute("cx",x);c.setAttribute("cy",y);c.setAttribute("r",frontier.has(d.config_id)?7:5);c.setAttribute("class",frontier.has(d.config_id)?"pareto-dot":"dominated-dot");c.innerHTML=`<title>${esc(d.label)}\n${METRICS[metric].label}: ${fmtMetric(metric,Number(d[metric]))}\np95: ${fmtMs(Number(d.latency_p95))}\nPareto: ${frontier.has(d.config_id)?"sim":"não"}</title>`;svg.appendChild(c);});
  const rows=data.filter(x=>frontier.has(x.config_id)).sort((a,b)=>Number(b[metric])-Number(a[metric]));const table=$("paretoTable"),head=table.querySelector("thead"),body=table.querySelector("tbody");head.innerHTML="<tr><th>Configuração</th><th>Qualidade</th><th>p95</th></tr>";body.innerHTML=rows.map(x=>`<tr><td>${esc(x.label)}</td><td><b>${fmtMetric(metric,Number(x[metric]))}</b></td><td>${fmtMs(Number(x.latency_p95))}</td></tr>`).join("");
}

function initInsights(){
  setupTabs();
  renderInsightKPIs();
  renderAutomaticSummary();
  renderFinalSelection();
  const factorSel=fillMetricSelect("factorMetric","ndcg_at_10");
  factorSel.addEventListener("change",()=>renderFactors(factorSel.value));
  renderFactors(factorSel.value);
  const paretoSel=fillMetricSelect("paretoMetric","ndcg_at_10");
  paretoSel.addEventListener("change",()=>renderPareto(paretoSel.value));
  renderPareto(paretoSel.value);
}

function render(){
  const rows=filteredRows(),groups=groupConfigs(rows); updateKPIs(rows,groups);barChart(groups);scatterChart(groups);rewriteChart(rows);chunkChart(rows);renderConfigTable(groups);syncQueryConfigs(groups);renderQueryTable();
}

function init(){
  $("runSubtitle").textContent=`run ${META.run_id} · split=${META.split} · ${META.retrieval_executions.toLocaleString("pt-BR")} retrievals · benchmark ${String(META.benchmark_sha256||"").slice(0,12)}…`;
  buildFilters();
  initInsights();
  $("methodNotes").innerHTML=`<b>Hit@K</b>: a cobertura cumulativa da evidência gold atinge o threshold <code>${META.evidence_min_coverage}</code> até K. <b>Coverage@K</b>: união da evidência gold coberta pelos chunks até K. <b>MRR@10</b>: inverso do primeiro rank em que o threshold cumulativo é alcançado. <b>nDCG@10</b>: usa relevância contínua baseada na cobertura individual de evidência implementada no runner. Latências de Dense usam query embeddings pré-computados; portanto <b>não incluem embedding da query</b> nesta etapa. Warmups descartados: <code>${META.warmup_runs}</code>.`;
  render();
}
init();
</script>
</body>
</html>"""

    return (
        template.replace("__META__", safe_script_json(meta))
        .replace("__ROWS__", safe_script_json(query_payload))
        .replace("__SUMMARY__", safe_script_json(summary_payload))
        .replace("__INSIGHTS__", safe_script_json(insights))
        .replace("__DEFAULT_METRIC__", DEFAULT_METRIC)
    )


def _empty_insights() -> dict[str, Any]:
    return {
        "metric_specs": METRIC_SPECS,
        "factors": {},
        "pareto": {},
        "configs": [],
        "selection_method": {},
        "final_selection": [],
        "bootstrap_runs": 0,
    }


def _aggregate_retrieval(frame: pd.DataFrame) -> dict[str, float | int]:
    if frame.empty:
        return {
            "n": 0,
            "hit_at_10": 0.0,
            "evidence_coverage_at_10": 0.0,
            "mrr_at_10": 0.0,
            "ndcg_at_10": 0.0,
            "correct_work_at_10": 0.0,
            "latency_p95_ms": 0.0,
        }

    def mean(column: str) -> float:
        return float(pd.to_numeric(frame[column], errors="coerce").mean())

    latency = pd.to_numeric(frame["retrieval_latency_ms"], errors="coerce").dropna()
    return {
        "n": int(len(frame)),
        "hit_at_10": mean("hit_at_10"),
        "evidence_coverage_at_10": mean("evidence_coverage_at_10"),
        "mrr_at_10": mean("mrr_at_10"),
        "ndcg_at_10": mean("ndcg_at_10"),
        "correct_work_at_10": mean("correct_work_at_10"),
        "latency_p95_ms": float(latency.quantile(0.95)) if len(latency) else 0.0,
    }


def _load_selection(
    storage: RustFSManager, bucket: str, run_id: str
) -> dict[str, Any] | None:
    key = f"{BENCHMARK_PREFIX}/runs/{run_id}/retrieval/final_selection_10.json"
    if not storage.object_exists(bucket, key):
        return None
    return load_json(storage, bucket, key)


def load_retrieval_run(
    storage: RustFSManager,
    bucket: str,
    run_id: str,
    *,
    bootstrap_runs: int,
) -> dict[str, Any]:
    run_id = str(run_id).strip()
    if not run_id:
        raise ValueError("run_id vazio")
    run_prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"
    retrieval_prefix = f"{run_prefix}/retrieval"
    manifest_key = f"{run_prefix}/manifest.json"
    config_key = f"{run_prefix}/config.json"
    query_key = f"{retrieval_prefix}/query_metrics.parquet"
    summary_key = f"{retrieval_prefix}/summary.parquet"
    for key in (manifest_key, config_key, query_key, summary_key):
        if not storage.object_exists(bucket, key):
            raise FileNotFoundError(f"Artefato ausente: s3://{bucket}/{key}")

    run_manifest = load_json(storage, bucket, manifest_key)
    run_config = load_json(storage, bucket, config_key)
    if run_manifest.get("status") != "success":
        raise RuntimeError(
            f"Run {run_id} não está concluído: status={run_manifest.get('status')}"
        )
    query_df = read_parquet(storage, bucket, query_key)
    summary_df = read_parquet(storage, bucket, summary_key)
    if query_df.empty or summary_df.empty:
        raise RuntimeError(f"{run_id}: query_metrics/summary vazios")

    split = str(run_config.get("split") or run_manifest.get("split") or "").strip()
    if split == "dev":
        print(
            f"[insights] {run_id}: seleção balanceada, fatores e bootstrap ({bootstrap_runs} runs)..."
        )
        insights = build_automatic_insights(query_df, bootstrap_runs=bootstrap_runs)
    else:
        insights = _empty_insights()

    html = build_html(
        run_id=run_id,
        run_config=run_config,
        run_manifest=run_manifest,
        query_df=query_df,
        summary_df=summary_df,
        insights=insights,
    )
    if split == "holdout":
        html = html.replace(
            '<button class="tab-btn" data-tab="tab-insights">Insights automáticos</button>',
            '<button class="tab-btn" data-tab="tab-insights" disabled title="Seleção e tuning pertencem ao DEV">Insights · somente DEV</button>',
        )

    return {
        "run_id": run_id,
        "run_prefix": run_prefix,
        "retrieval_prefix": retrieval_prefix,
        "manifest": run_manifest,
        "config": run_config,
        "split": split,
        "query_df": query_df,
        "summary_df": summary_df,
        "insights": insights,
        "html": html,
        "benchmark_sha256": run_config.get("benchmark_sha256"),
        "config_ids": sorted(query_df["config_id"].astype(str).unique().tolist()),
        "selection": _load_selection(storage, bucket, run_id),
    }


def _validate_retrieval_pair(dev: dict[str, Any], holdout: dict[str, Any]) -> list[str]:
    if dev["split"] != "dev":
        raise RuntimeError(
            f"--run-id deve ser DEV quando --holdout-run-id é usado; split={dev['split']}"
        )
    if holdout["split"] != "holdout":
        raise RuntimeError(
            f"--holdout-run-id deve ser HOLDOUT; split={holdout['split']}"
        )
    if dev.get("benchmark_sha256") != holdout.get("benchmark_sha256"):
        raise RuntimeError("DEV e HOLDOUT usam benchmark_sha256 diferentes.")
    holdout_ids = set(holdout["config_ids"])
    dev_ids = set(dev["config_ids"])
    if not holdout_ids.issubset(dev_ids):
        raise RuntimeError("HOLDOUT contém config_id que não existe no DEV.")
    selection = dev.get("selection") or {}
    selected_ids = {
        str(item.get("config_id"))
        for item in (selection.get("selected") or [])
        if str(item.get("config_id") or "").strip()
    }
    if selected_ids and holdout_ids != selected_ids:
        missing = sorted(selected_ids - holdout_ids)
        extra = sorted(holdout_ids - selected_ids)
        raise RuntimeError(
            "HOLDOUT não corresponde exatamente às configs congeladas no DEV. "
            f"missing={missing}; extra={extra}"
        )
    return sorted(holdout_ids)


def _retrieval_comparison_payload(
    dev: dict[str, Any],
    holdout: dict[str, Any] | None,
) -> dict[str, Any]:
    if holdout is None:
        return {"dev": None, "holdout": None, "configs": []}
    compare_ids = _validate_retrieval_pair(dev, holdout)
    dev_frame = (
        dev["query_df"]
        .loc[dev["query_df"]["config_id"].astype(str).isin(compare_ids)]
        .copy()
    )
    hold_frame = (
        holdout["query_df"]
        .loc[holdout["query_df"]["config_id"].astype(str).isin(compare_ids)]
        .copy()
    )

    def row_label(group: pd.DataFrame) -> str:
        first = group.iloc[0]
        return _config_label_from_row(first)

    dev_cfg = {
        str(config_id): {
            "config_id": str(config_id),
            "label": row_label(group),
            **_aggregate_retrieval(group),
        }
        for config_id, group in dev_frame.groupby("config_id", sort=False)
    }
    hold_cfg = {
        str(config_id): {
            "config_id": str(config_id),
            "label": row_label(group),
            **_aggregate_retrieval(group),
        }
        for config_id, group in hold_frame.groupby("config_id", sort=False)
    }
    return {
        "dev": {
            "run_id": dev["run_id"],
            "overall": _aggregate_retrieval(dev_frame),
            "queries": int(dev_frame["query_id"].astype(str).nunique()),
            "executions": int(len(dev_frame)),
        },
        "holdout": {
            "run_id": holdout["run_id"],
            "overall": _aggregate_retrieval(hold_frame),
            "queries": int(hold_frame["query_id"].astype(str).nunique()),
            "executions": int(len(hold_frame)),
        },
        "configs": [
            {
                "config_id": config_id,
                "label": dev_cfg[config_id]["label"],
                "dev": dev_cfg[config_id],
                "holdout": hold_cfg[config_id],
            }
            for config_id in compare_ids
        ],
    }


def render_dual_retrieval_html(
    *, dev: dict[str, Any], holdout: dict[str, Any] | None
) -> str:
    import base64

    comparison = _retrieval_comparison_payload(dev, holdout)
    dev_b64 = base64.b64encode(dev["html"].encode("utf-8")).decode("ascii")
    holdout_b64 = (
        base64.b64encode(holdout["html"].encode("utf-8")).decode("ascii")
        if holdout is not None
        else ""
    )
    template = r"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/><title>ACartomante · Retrieval DEV × HOLDOUT</title><style>
:root{--bg:#f6f7fb;--card:#fff;--ink:#182033;--muted:#697386;--line:#e5e8f0;--blue:#3157d5;--green:#18864b;--red:#c43d4b;--shadow:0 8px 24px rgba(30,40,70,.07)}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}.top{position:sticky;top:0;z-index:20;background:#fff;border-bottom:1px solid var(--line);padding:14px 20px;box-shadow:0 3px 14px rgba(17,24,39,.04)}.topline{max-width:1600px;margin:auto;display:flex;align-items:center;justify-content:space-between;gap:16px}.title h1{font-size:20px;margin:0}.title p{margin:3px 0 0;color:var(--muted);font-size:12px}.split-nav{display:flex;gap:6px}.split-btn{border:1px solid var(--line);background:#fff;border-radius:9px;padding:8px 14px;font-weight:700;cursor:pointer}.split-btn.active{background:var(--blue);border-color:var(--blue);color:#fff}.split-btn:disabled{opacity:.38;cursor:not-allowed}.view{display:none}.view.active{display:block}.compare{max-width:1500px;margin:0 auto;padding:20px}.grid{display:grid;grid-template-columns:repeat(12,minmax(0,1fr));gap:14px}.span-12{grid-column:span 12}.span-6{grid-column:span 6}.card{background:#fff;border:1px solid var(--line);border-radius:14px;box-shadow:var(--shadow);padding:16px}.card h2{font-size:16px;margin:0 0 5px}.sub{color:var(--muted);font-size:12px;margin-bottom:12px}.kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:9px}.kpi{border:1px solid var(--line);border-radius:11px;padding:11px}.kpi .v{font-size:21px;font-weight:760}.kpi .l{font-size:11px;color:var(--muted)}.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:10px}table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}th{background:#f9fafc;color:#4f596e}th:first-child,td:first-child{text-align:left}.delta-pos{color:var(--green);font-weight:700}.delta-neg{color:var(--red);font-weight:700}.chart{height:310px;position:relative}.chart svg{width:100%;height:100%;overflow:visible}.chart .gridline{stroke:#edf0f6;stroke-width:1}.chart .axis{stroke:#cbd3df;stroke-width:1}.chart .tick{fill:#697386;font-size:11px}.chart .bar-dev{fill:#3157d5}.chart .bar-holdout{fill:#7652cc}.chart-legend{display:flex;gap:14px;flex-wrap:wrap;color:var(--muted);font-size:11px;margin-top:7px}.chart-legend span:before{content:"";display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px;background:#3157d5}.chart-legend span.holdout:before{background:#7652cc}.frame{width:100%;height:calc(100vh - 82px);border:0;background:#fff}.empty{padding:50px;text-align:center;color:var(--muted)}@media(max-width:900px){.span-6{grid-column:span 12}.kpis{grid-template-columns:repeat(2,1fr)}.topline{align-items:flex-start;flex-direction:column}}
</style></head><body><header class="top"><div class="topline"><div class="title"><h1>ACartomante · Retrieval Benchmark</h1><p id="headerMeta"></p></div><nav class="split-nav"><button class="split-btn" id="btn-general">GERAL</button><button class="split-btn" id="btn-dev">DEV</button><button class="split-btn" id="btn-holdout">HOLDOUT</button></nav></div></header><section id="view-general" class="view"><div class="compare" id="generalContent"></div></section><section id="view-dev" class="view"><iframe class="frame" id="frame-dev" title="Retrieval DEV"></iframe></section><section id="view-holdout" class="view"><iframe class="frame" id="frame-holdout" title="Retrieval HOLDOUT"></iframe></section><script>
const DATA=__COMPARISON__,DEV_B64="__DEV_B64__",HOLDOUT_B64="__HOLDOUT_B64__",hasHoldout=Boolean(DATA.holdout);const $=id=>document.getElementById(id);const pct=v=>`${(100*Number(v||0)).toFixed(1)}%`;const num=v=>Number(v||0).toFixed(3);const ms=v=>`${Number(v||0).toFixed(1)} ms`;const esc=v=>String(v??"").replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;","\"":"&quot;","'":"&#39;"}[c]));const delta=(a,b)=>Number(b||0)-Number(a||0);const dc=v=>v>1e-12?'delta-pos':v<-1e-12?'delta-neg':'';const dnum=v=>`${v>=0?'+':''}${v.toFixed(3)}`;const dpct=v=>`${v>=0?'+':''}${(100*v).toFixed(1)} pp`;function decode64(x){if(!x)return '';const bin=atob(x),bytes=Uint8Array.from(bin,c=>c.charCodeAt(0));return new TextDecoder().decode(bytes)}
function setView(name){['general','dev','holdout'].forEach(x=>{$(`view-${x}`).classList.toggle('active',x===name);$(`btn-${x}`).classList.toggle('active',x===name)});if(name==='dev'&&!$('frame-dev').dataset.loaded){$('frame-dev').srcdoc=decode64(DEV_B64);$('frame-dev').dataset.loaded='1'}if(name==='holdout'&&hasHoldout&&!$('frame-holdout').dataset.loaded){$('frame-holdout').srcdoc=decode64(HOLDOUT_B64);$('frame-holdout').dataset.loaded='1'}}
function svgEl(name,attrs={}){const e=document.createElementNS('http://www.w3.org/2000/svg',name);Object.entries(attrs).forEach(([k,v])=>e.setAttribute(k,v));return e}function svgText(svg,x,y,text,anchor='middle'){const t=svgEl('text',{x,y,class:'tick','text-anchor':anchor});t.textContent=text;svg.appendChild(t);return t}function chartSvg(id,w=900,h=290){const el=$(id);el.innerHTML='';const svg=svgEl('svg',{viewBox:`0 0 ${w} ${h}`});el.appendChild(svg);return svg}
function metricOverviewChart(){const svg=chartSvg('retrievalMetricsChart'),w=900,h=290,l=65,r=20,t=18,b=55,iw=w-l-r,ih=h-t-b,metrics=[['nDCG@10','ndcg_at_10'],['Hit@10','hit_at_10'],['Coverage@10','evidence_coverage_at_10'],['MRR@10','mrr_at_10']],d=DATA.dev.overall,o=DATA.holdout.overall;for(let i=0;i<=4;i++){const y=t+ih*(1-i/4),v=i/4;svg.appendChild(svgEl('line',{x1:l,y1:y,x2:w-r,y2:y,class:'gridline'}));svgText(svg,l-8,y+4,v.toFixed(2),'end')}const gap=iw/metrics.length,bw=Math.min(38,gap*.24);metrics.forEach(([label,key],i)=>{const cx=l+gap*(i+.5),dv=Number(d[key]||0),hv=Number(o[key]||0);[[dv,-bw*.65,'bar-dev','DEV'],[hv,bw*.65,'bar-holdout','HOLDOUT']].forEach(([v,off,cls,side])=>{const bh=ih*v,rect=svgEl('rect',{x:cx+off-bw/2,y:t+ih-bh,width:bw,height:Math.max(1,bh),rx:4,class:cls});const title=svgEl('title');title.textContent=`${label} · ${side}: ${v.toFixed(3)}`;rect.appendChild(title);svg.appendChild(rect)});svgText(svg,cx,t+ih+19,label);});}
function horizontalConfigChart(id,valueKey,formatter){const rows=DATA.configs,h=Math.max(340,rows.length*34+45),svg=chartSvg(id,900,h),w=900,l=320,r=25,t=18,b=25,iw=w-l-r,gap=(h-t-b)/rows.length,bh=Math.min(10,gap*.32),max=Math.max(1e-9,...rows.flatMap(x=>[Number(x.dev[valueKey]||0),Number(x.holdout[valueKey]||0)]));for(let i=0;i<=4;i++){const x=l+iw*i/4;svg.appendChild(svgEl('line',{x1:x,y1:t,x2:x,y2:h-b,class:'gridline'}));svgText(svg,x,h-5,formatter(max*i/4));}rows.forEach((row,i)=>{const cy=t+gap*(i+.5),raw=String(row.label),label=raw.length>44?raw.slice(0,41)+'…':raw;svgText(svg,l-10,cy+4,label,'end');[['dev','bar-dev',-bh*.7,'DEV'],['holdout','bar-holdout',bh*.7,'HOLDOUT']].forEach(([side,cls,off,sideLabel])=>{const v=Number(row[side][valueKey]||0),rect=svgEl('rect',{x:l,y:cy+off-bh/2,width:iw*(v/max),height:bh,rx:3,class:cls});const title=svgEl('title');title.textContent=`${raw} · ${sideLabel}: ${formatter(v)}`;rect.appendChild(title);svg.appendChild(rect);});});}
function renderGeneral(){if(!hasHoldout){$('generalContent').innerHTML='<div class="card empty">O comparativo GERAL é habilitado quando um run HOLDOUT é informado.</div>';return}const d=DATA.dev,h=DATA.holdout,da=d.overall,ha=h.overall;$('generalContent').innerHTML=`<div class="grid"><div class="card span-6"><h2>DEV · mesmas 10 configs</h2><div class="sub">${d.queries} queries · ${d.executions} execuções usadas no comparativo</div><div class="kpis"><div class="kpi"><div class="v">${num(da.ndcg_at_10)}</div><div class="l">nDCG@10</div></div><div class="kpi"><div class="v">${pct(da.hit_at_10)}</div><div class="l">Hit@10</div></div><div class="kpi"><div class="v">${pct(da.evidence_coverage_at_10)}</div><div class="l">Coverage@10</div></div><div class="kpi"><div class="v">${num(da.mrr_at_10)}</div><div class="l">MRR@10</div></div></div></div><div class="card span-6"><h2>HOLDOUT</h2><div class="sub">${h.queries} queries · ${h.executions} execuções · mesmas configs congeladas no DEV</div><div class="kpis"><div class="kpi"><div class="v">${num(ha.ndcg_at_10)}</div><div class="l">nDCG@10</div></div><div class="kpi"><div class="v">${pct(ha.hit_at_10)}</div><div class="l">Hit@10</div></div><div class="kpi"><div class="v">${pct(ha.evidence_coverage_at_10)}</div><div class="l">Coverage@10</div></div><div class="kpi"><div class="v">${num(ha.mrr_at_10)}</div><div class="l">MRR@10</div></div></div></div><div class="card span-6"><h2>Métricas gerais · DEV × HOLDOUT</h2><div class="sub">Visão rápida da estabilidade das quatro métricas principais.</div><div class="chart" id="retrievalMetricsChart"></div><div class="chart-legend"><span>DEV</span><span class="holdout">HOLDOUT</span></div></div><div class="card span-6"><h2>nDCG@10 por configuração</h2><div class="sub">As mesmas 10 configurações congeladas, comparadas sem nova seleção.</div><div class="chart" id="retrievalNdcgChart" style="height:390px"></div><div class="chart-legend"><span>DEV</span><span class="holdout">HOLDOUT</span></div></div><div class="card span-12"><h2>Latência p95 por configuração</h2><div class="sub">Ajuda a verificar se o custo operacional observado no DEV também se manteve no HOLDOUT.</div><div class="chart" id="retrievalLatencyChart" style="height:390px"></div><div class="chart-legend"><span>DEV</span><span class="holdout">HOLDOUT</span></div></div><div class="card span-12"><h2>10 configurações congeladas · DEV × HOLDOUT</h2><div class="sub">A lista não é recalculada no HOLDOUT. A tabela mostra somente estabilidade/deslocamento de desempenho.</div><div class="table-wrap"><table><thead><tr><th>Configuração</th><th>DEV nDCG</th><th>HOLDOUT nDCG</th><th>Δ nDCG</th><th>DEV Hit@10</th><th>HOLDOUT Hit@10</th><th>Δ Hit</th><th>DEV Cov@10</th><th>HOLDOUT Cov@10</th><th>DEV p95</th><th>HOLDOUT p95</th></tr></thead><tbody>${DATA.configs.map(x=>{const a=x.dev,b=x.holdout,dn=delta(a.ndcg_at_10,b.ndcg_at_10),dh=delta(a.hit_at_10,b.hit_at_10);return `<tr><td>${esc(x.label)}</td><td>${num(a.ndcg_at_10)}</td><td>${num(b.ndcg_at_10)}</td><td class="${dc(dn)}">${dnum(dn)}</td><td>${pct(a.hit_at_10)}</td><td>${pct(b.hit_at_10)}</td><td class="${dc(dh)}">${dpct(dh)}</td><td>${pct(a.evidence_coverage_at_10)}</td><td>${pct(b.evidence_coverage_at_10)}</td><td>${ms(a.latency_p95_ms)}</td><td>${ms(b.latency_p95_ms)}</td></tr>`}).join('')}</tbody></table></div></div></div>`;metricOverviewChart();horizontalConfigChart('retrievalNdcgChart','ndcg_at_10',v=>Number(v).toFixed(3));horizontalConfigChart('retrievalLatencyChart','latency_p95_ms',v=>`${Number(v).toFixed(0)} ms`);}
$('btn-dev').onclick=()=>setView('dev');$('btn-holdout').onclick=()=>hasHoldout&&setView('holdout');$('btn-general').onclick=()=>hasHoldout&&setView('general');$('btn-holdout').disabled=!hasHoldout;$('btn-general').disabled=!hasHoldout;$('headerMeta').textContent=hasHoldout?`${DATA.dev.run_id} × ${DATA.holdout.run_id}`:`${__DEV_RUN__} · HOLDOUT ainda não informado`;renderGeneral();setView(hasHoldout?'general':'dev');
</script></body></html>"""
    return (
        template.replace("__COMPARISON__", safe_script_json(comparison))
        .replace("__DEV_B64__", dev_b64)
        .replace("__HOLDOUT_B64__", holdout_b64)
        .replace("__DEV_RUN__", safe_script_json(dev["run_id"]))
    )


def main() -> None:
    args = parse_args()
    if args.bootstrap_runs < 0:
        raise ValueError("--bootstrap-runs deve ser >= 0.")

    config = ConfigManager.load()
    storage = RustFSManager.from_config(config.environment.rustfs)
    bucket = config.storage.benchmarks_bucket

    dev = load_retrieval_run(
        storage, bucket, args.run_id, bootstrap_runs=args.bootstrap_runs
    )
    holdout: dict[str, Any] | None = None
    if args.holdout_run_id:
        holdout = load_retrieval_run(
            storage, bucket, args.holdout_run_id, bootstrap_runs=0
        )
        _validate_retrieval_pair(dev, holdout)

    html = render_dual_retrieval_html(dev=dev, holdout=holdout)
    if args.output is not None:
        output = args.output
    elif holdout is not None:
        output = (
            Path("reports") / f"retrieval_{dev['run_id']}__{holdout['run_id']}.html"
        )
    else:
        output = Path("reports") / f"retrieval_{dev['run_id']}.html"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html, encoding="utf-8")

    # Somente DEV pode criar/atualizar a seleção congelada.
    if dev["split"] == "dev" and dev["insights"].get("final_selection"):
        selection_payload = {
            "run_id": dev["run_id"],
            "split": dev["split"],
            "benchmark_sha256": dev["config"].get("benchmark_sha256"),
            "method": dev["insights"]["selection_method"],
            "selected": dev["insights"]["final_selection"],
        }
        selection_path = output.with_name(f"retrieval_{dev['run_id']}_final10.json")
        selection_path.write_text(
            json.dumps(selection_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"final10: {selection_path.resolve()}")
        if not args.no_upload:
            selection_key = f"{dev['retrieval_prefix']}/final_selection_10.json"
            storage.upload_bytes(
                bucket,
                selection_key,
                json.dumps(selection_payload, ensure_ascii=False, indent=2).encode(
                    "utf-8"
                ),
                content_type="application/json; charset=utf-8",
            )
            print(f"selection: s3://{bucket}/{selection_key}")

    if not args.no_upload:
        report_key = f"{dev['run_prefix']}/reports/{REPORT_KEY_NAME}"
        storage.upload_bytes(
            bucket,
            report_key,
            html.encode("utf-8"),
            content_type="text/html; charset=utf-8",
        )
        print(f"rustfs: s3://{bucket}/{report_key}")

    print("=== RETRIEVAL DASHBOARD OK ===")
    print(f"dev: {dev['run_id']} | query rows={len(dev['query_df'])}")
    if holdout is not None:
        print(f"holdout: {holdout['run_id']} | query rows={len(holdout['query_df'])}")
        print("geral: enabled | holdout selection: frozen from DEV")
    else:
        print("holdout: ausente | seletor desabilitado")
        print("geral: disabled")
    print(f"local: {output.resolve()}")

    if args.open_browser:
        webbrowser.open(output.resolve().as_uri())


if __name__ == "__main__":
    main()
