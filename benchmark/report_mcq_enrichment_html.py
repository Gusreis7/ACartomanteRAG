#!/usr/bin/env python3
from __future__ import annotations

import argparse
import io
import json
import math
import webbrowser
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from benchmark.matrix import BENCHMARK_PREFIX
from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager


PAIR_KEYS = ["query_id", "config_id", "model"]
REPORT_NAME = "mcq_enrichment_vs_baseline.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dashboard MCQ pareado: baseline vs enrichment."
    )
    parser.add_argument("--baseline-run-id", required=True)
    parser.add_argument("--enriched-run-id", required=True)
    parser.add_argument("--bootstrap-runs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--no-upload", action="store_true")
    parser.add_argument("--open", action="store_true")
    return parser.parse_args()


def read_json(
    storage: RustFSManager,
    bucket: str,
    key: str,
) -> dict[str, Any]:
    return json.loads(
        storage.download_bytes(
            bucket,
            key,
        ).decode("utf-8")
    )


def read_parquet(
    storage: RustFSManager,
    bucket: str,
    key: str,
) -> pd.DataFrame:
    return pd.read_parquet(
        io.BytesIO(
            storage.download_bytes(
                bucket,
                key,
            )
        )
    )


def load_run(
    storage: RustFSManager,
    bucket: str,
    run_id: str,
):
    prefix = f"{BENCHMARK_PREFIX}/runs/{run_id}"

    config = read_json(
        storage,
        bucket,
        f"{prefix}/config.json",
    )
    manifest = read_json(
        storage,
        bucket,
        f"{prefix}/manifest.json",
    )

    if str(manifest.get("status")) != "success":
        raise RuntimeError(
            f"Run {run_id} não está com status=success."
        )

    if str(config.get("phase")) != "mcq":
        raise RuntimeError(
            f"Run {run_id} não pertence à fase MCQ."
        )

    results = read_parquet(
        storage,
        bucket,
        f"{prefix}/mcq/results.parquet",
    )

    contexts = None
    contexts_key = f"{prefix}/mcq/contexts.parquet"

    if storage.object_exists(
        bucket,
        contexts_key,
    ):
        contexts = read_parquet(
            storage,
            bucket,
            contexts_key,
        )

    return config, manifest, results, contexts


def scalar(value: Any) -> Any:
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
            value = value.item()
        except Exception:
            pass

    if (
        isinstance(value, float)
        and not math.isfinite(value)
    ):
        return None

    return value


def safe_json(value: Any) -> str:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            separators=(",", ":"),
            default=str,
        )
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
    )


def numeric_mean(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame.columns:
        return 0.0

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    return float(values.mean()) if len(values) else 0.0


def numeric_p95(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame.columns:
        return 0.0

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).dropna()

    return (
        float(values.quantile(0.95))
        if len(values)
        else 0.0
    )


def rate(
    frame: pd.DataFrame,
    column: str,
) -> float:
    if column not in frame.columns or frame.empty:
        return 0.0

    return float(
        frame[column]
        .astype(float)
        .mean()
    )


def bootstrap_delta(
    frame: pd.DataFrame,
    *,
    runs: int,
    seed: int,
) -> dict[str, float]:
    values = (
        frame.groupby(
            "query_id",
            sort=False,
        )["accuracy_delta"]
        .mean()
        .astype(float)
    )

    observed = (
        float(values.mean())
        if len(values)
        else 0.0
    )

    if runs <= 0 or len(values) <= 1:
        return {
            "delta": observed,
            "low": observed,
            "high": observed,
        }

    rng = np.random.default_rng(seed)
    arr = values.to_numpy(dtype=float)

    samples = np.empty(
        runs,
        dtype=float,
    )

    for index in range(runs):
        samples[index] = float(
            rng.choice(
                arr,
                size=len(arr),
                replace=True,
            ).mean()
        )

    return {
        "delta": observed,
        "low": float(
            np.quantile(
                samples,
                0.025,
            )
        ),
        "high": float(
            np.quantile(
                samples,
                0.975,
            )
        ),
    }


def aggregate(
    frame: pd.DataFrame,
    *,
    bootstrap_runs: int,
    seed: int,
) -> dict[str, Any]:
    boot = bootstrap_delta(
        frame,
        runs=bootstrap_runs,
        seed=seed,
    )

    base_latency = numeric_mean(
        frame,
        "generation_latency_ms_baseline",
    )
    enr_latency = numeric_mean(
        frame,
        "generation_latency_ms_enriched",
    )

    base_tokens = numeric_mean(
        frame,
        "input_tokens_baseline",
    )
    enr_tokens = numeric_mean(
        frame,
        "input_tokens_enriched",
    )

    return {
        "rows": int(len(frame)),
        "queries": int(
            frame["query_id"].nunique()
        ),
        "baseline_accuracy": rate(
            frame,
            "is_correct_baseline",
        ),
        "enriched_accuracy": rate(
            frame,
            "is_correct_enriched",
        ),
        "accuracy_delta": boot["delta"],
        "ci95_low": boot["low"],
        "ci95_high": boot["high"],
        "improved_rows": int(
            (
                frame["accuracy_delta"]
                > 0
            ).sum()
        ),
        "regressed_rows": int(
            (
                frame["accuracy_delta"]
                < 0
            ).sum()
        ),
        "unchanged_rows": int(
            (
                frame["accuracy_delta"]
                == 0
            ).sum()
        ),
        "baseline_ambiguous": rate(
            frame,
            "is_ambiguous_baseline",
        ),
        "enriched_ambiguous": rate(
            frame,
            "is_ambiguous_enriched",
        ),
        "baseline_wrong": rate(
            frame,
            "is_wrong_baseline",
        ),
        "enriched_wrong": rate(
            frame,
            "is_wrong_enriched",
        ),
        "baseline_invalid": rate(
            frame,
            "is_invalid_baseline",
        ),
        "enriched_invalid": rate(
            frame,
            "is_invalid_enriched",
        ),
        "baseline_latency_mean_ms": base_latency,
        "enriched_latency_mean_ms": enr_latency,
        "latency_delta_ms": (
            enr_latency
            - base_latency
        ),
        "baseline_latency_p95_ms": numeric_p95(
            frame,
            "generation_latency_ms_baseline",
        ),
        "enriched_latency_p95_ms": numeric_p95(
            frame,
            "generation_latency_ms_enriched",
        ),
        "baseline_input_tokens_mean": base_tokens,
        "enriched_input_tokens_mean": enr_tokens,
        "input_tokens_delta": (
            enr_tokens
            - base_tokens
        ),
        "enrichment_context_chars_mean": numeric_mean(
            frame,
            "enrichment_context_chars_enriched",
        ),
        "enrichment_hit_count_mean": numeric_mean(
            frame,
            "enrichment_hit_count_enriched",
        ),
    }


def build_breakdown(
    paired: pd.DataFrame,
    *,
    column: str,
    output_name: str,
    bootstrap_runs: int,
    seed: int,
) -> list[dict[str, Any]]:
    if column not in paired.columns:
        return []

    rows = []

    for value, group in paired.groupby(
        column,
        sort=False,
        dropna=False,
    ):
        item = aggregate(
            group,
            bootstrap_runs=bootstrap_runs,
            seed=seed,
        )
        item[output_name] = str(value)
        rows.append(item)

    return rows


def load_config_labels(
    storage: RustFSManager,
    bucket: str,
    retrieval_run_id: str,
) -> dict[str, dict[str, Any]]:
    key = (
        f"{BENCHMARK_PREFIX}/runs/"
        f"{retrieval_run_id}/retrieval/"
        "final_selection_10.json"
    )

    if not storage.object_exists(
        bucket,
        key,
    ):
        return {}

    payload = read_json(
        storage,
        bucket,
        key,
    )

    output = {}

    for index, item in enumerate(
        payload.get("selected") or [],
        start=1,
    ):
        row = dict(item)
        config_id = str(
            row.get("config_id")
            or ""
        ).strip()

        if not config_id:
            continue

        strategy = str(
            row.get("chunk_strategy")
            or ""
        )
        size = str(
            row.get("chunk_size")
            or ""
        )
        retrieval = str(
            row.get("retrieval_strategy")
            or ""
        )
        embedder = str(
            row.get("embedder")
            or ""
        )

        head = "/".join(
            value
            for value in (
                strategy,
                size,
            )
            if value
        )

        label = " · ".join(
            value
            for value in (
                head,
                retrieval,
                embedder,
            )
            if value
        )

        output[config_id] = {
            **row,
            "selection_rank": int(
                row.get("selection_rank")
                or index
            ),
            "label": (
                label
                or config_id
            ),
        }

    return output


def build_question_rows(
    paired: pd.DataFrame,
) -> list[dict[str, Any]]:
    rows = []

    for query_id, group in paired.groupby(
        "query_id",
        sort=False,
    ):
        first = group.iloc[0]

        baseline = rate(
            group,
            "is_correct_baseline",
        )
        enriched = rate(
            group,
            "is_correct_enriched",
        )

        rows.append(
            {
                "query_id": str(query_id),
                "query": str(
                    first.get(
                        "query_baseline",
                        first.get(
                            "query_enriched",
                            "",
                        ),
                    )
                    or ""
                ),
                "canonical_work": str(
                    first.get(
                        "canonical_work_baseline",
                        first.get(
                            "canonical_work_enriched",
                            "",
                        ),
                    )
                    or ""
                ),
                "difficulty": str(
                    first.get(
                        "difficulty_baseline",
                        first.get(
                            "difficulty_enriched",
                            "",
                        ),
                    )
                    or ""
                ),
                "query_type": str(
                    first.get(
                        "query_type_baseline",
                        first.get(
                            "query_type_enriched",
                            "",
                        ),
                    )
                    or ""
                ),
                "baseline_accuracy": baseline,
                "enriched_accuracy": enriched,
                "accuracy_delta": (
                    enriched
                    - baseline
                ),
                "improved_rows": int(
                    (
                        group["accuracy_delta"]
                        > 0
                    ).sum()
                ),
                "regressed_rows": int(
                    (
                        group["accuracy_delta"]
                        < 0
                    ).sum()
                ),
                "unchanged_rows": int(
                    (
                        group["accuracy_delta"]
                        == 0
                    ).sum()
                ),
                "enrichment_context_chars_mean": numeric_mean(
                    group,
                    "enrichment_context_chars_enriched",
                ),
            }
        )

    rows.sort(
        key=lambda item: (
            -float(
                item["accuracy_delta"]
            ),
            item["query_id"],
        )
    )

    return rows


def parse_hits(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, list):
        return [
            dict(item)
            for item in value
            if isinstance(item, dict)
        ]

    text = str(value).strip()

    if not text:
        return []

    try:
        payload = json.loads(text)
    except Exception:
        return []

    if not isinstance(payload, list):
        return []

    return [
        dict(item)
        for item in payload
        if isinstance(item, dict)
    ]


def build_hit_samples(
    contexts: pd.DataFrame | None,
    question_lookup: dict[str, dict[str, Any]],
    config_lookup: dict[str, dict[str, Any]],
    *,
    limit: int = 100,
) -> list[dict[str, Any]]:
    if (
        contexts is None
        or contexts.empty
        or (
            "enrichment_hits_json"
            not in contexts.columns
        )
    ):
        return []

    unique = (
        contexts
        .drop_duplicates(
            [
                "query_id",
                "config_id",
            ]
        )
        .head(limit)
    )

    rows = []

    for _, row in unique.iterrows():
        query_id = str(
            row.get("query_id")
            or ""
        )
        config_id = str(
            row.get("config_id")
            or ""
        )

        question = question_lookup.get(
            query_id,
            {},
        )

        config_meta = config_lookup.get(
            config_id,
            {},
        )

        rows.append(
            {
                "query_id": query_id,
                "query": question.get(
                    "query",
                    "",
                ),
                "canonical_work": question.get(
                    "canonical_work",
                    "",
                ),
                "config_id": config_id,
                "config_label": config_meta.get(
                    "label",
                    config_id,
                ),
                "embedder": scalar(
                    row.get(
                        "enrichment_embedder"
                    )
                ),
                "index_id": scalar(
                    row.get(
                        "enrichment_index_id"
                    )
                ),
                "collection": scalar(
                    row.get(
                        "enrichment_collection"
                    )
                ),
                "context_chars": int(
                    row.get(
                        "enrichment_context_chars"
                    )
                    or 0
                ),
                "hit_count": int(
                    row.get(
                        "enrichment_hit_count"
                    )
                    or 0
                ),
                "hits": parse_hits(
                    row.get(
                        "enrichment_hits_json"
                    )
                ),
            }
        )

    return rows


def build_transitions(
    paired: pd.DataFrame,
) -> list[dict[str, Any]]:
    grouped = (
        paired
        .groupby(
            [
                "selected_class_baseline",
                "selected_class_enriched",
            ],
            dropna=False,
        )
        .size()
        .reset_index(
            name="n"
        )
    )

    rows = []

    for _, row in grouped.iterrows():
        rows.append(
            {
                "baseline": str(
                    row[
                        "selected_class_baseline"
                    ]
                ),
                "enriched": str(
                    row[
                        "selected_class_enriched"
                    ]
                ),
                "n": int(row["n"]),
            }
        )

    rows.sort(
        key=lambda item: (
            -item["n"],
            item["baseline"],
            item["enriched"],
        )
    )

    return rows


def render_html(
    payload: dict[str, Any],
) -> str:
    template = r'''<!doctype html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MCQ · Baseline vs Enrichment</title>
<style>
:root{
  --bg:#f7f8fc;--card:#fff;--ink:#182033;--muted:#667085;--line:#e6e8f0;
  --blue:#315efb;--purple:#7457ff;--green:#0e9f6e;--red:#d94d62;
  --green-bg:#eaf8f2;--red-bg:#fff0f2;--shadow:0 8px 24px rgba(35,45,80,.07)
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:Inter,system-ui,sans-serif}
main{max-width:1500px;margin:auto;padding:28px}
.header{display:flex;justify-content:space-between;gap:20px;margin-bottom:18px}
h1{font-size:30px;margin:0 0 7px}h2{font-size:17px;margin:0 0 12px}
.sub{color:var(--muted);font-size:13px;line-height:1.5}
.pills{display:flex;gap:8px;flex-wrap:wrap;justify-content:flex-end}
.pill{background:#fff;border:1px solid var(--line);padding:7px 10px;border-radius:999px;font-size:11px;color:var(--muted)}
.tabs{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:16px}
.tab{border:1px solid var(--line);background:#fff;padding:9px 13px;border-radius:10px;cursor:pointer;color:var(--muted);font-weight:700}
.tab.active{color:#fff;border-color:var(--blue);background:linear-gradient(135deg,var(--blue),var(--purple))}
.panel{display:none}.panel.active{display:block}
.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:14px}
.s12{grid-column:span 12}.s8{grid-column:span 8}.s6{grid-column:span 6}.s4{grid-column:span 4}.s3{grid-column:span 3}
.card{background:var(--card);border:1px solid var(--line);border-radius:16px;box-shadow:var(--shadow);padding:18px}
.kpi{min-height:126px;display:flex;flex-direction:column;justify-content:space-between}
.label{font-size:12px;color:var(--muted)}.value{font-size:31px;font-weight:800;letter-spacing:-.03em}
.detail{font-size:12px;color:var(--muted);line-height:1.45}
.good{color:var(--green)!important}.bad{color:var(--red)!important}
.tag{display:inline-block;border-radius:999px;padding:4px 8px;font-size:11px;font-weight:750}
.tag.good{background:var(--green-bg)}.tag.bad{background:var(--red-bg)}
.note{font-size:13px;color:var(--muted);line-height:1.55}
.bar-row{display:grid;grid-template-columns:minmax(180px,1.4fr) 1fr 72px;gap:10px;align-items:center;margin:9px 0}
.track{height:10px;border-radius:999px;background:#eef0f6;overflow:hidden}
.fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--purple))}
.fill.bad{background:linear-gradient(90deg,#ef8292,var(--red))}
.num{text-align:right;font-variant-numeric:tabular-nums}
.table-wrap{overflow:auto;border:1px solid var(--line);border-radius:12px}
table{width:100%;border-collapse:collapse;min-width:860px;font-size:12px}
th,td{padding:10px 11px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{position:sticky;top:0;background:#fafbff;color:#475467;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
tr:hover td{background:#fafbff}
.toolbar{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:12px}
.toolbar input,.toolbar select{border:1px solid var(--line);background:#fff;border-radius:10px;padding:9px 10px;color:var(--ink)}
.toolbar input{min-width:280px;flex:1}
.hit-card{border:1px solid var(--line);border-radius:13px;padding:14px;margin-bottom:10px}
.hit-title{font-weight:750}.hit-meta{font-size:11px;color:var(--muted)}
.hit{background:#fafbff;border:1px solid #edf0f6;border-radius:10px;padding:10px;margin-top:8px}
.hit .score{color:var(--blue);font-weight:800}.summary{font-size:12px;line-height:1.48;margin-top:5px}
.compare{display:grid;grid-template-columns:1fr 1fr;gap:12px}
.compare>div{border:1px solid var(--line);background:#fcfcff;border-radius:12px;padding:13px}
.compare .big{font-size:25px;font-weight:800;margin-top:3px}
.footer{text-align:center;color:var(--muted);font-size:11px;padding:24px 0 8px}
@media(max-width:900px){.s8,.s6,.s4,.s3{grid-column:span 12}.header{flex-direction:column}.pills{justify-content:flex-start}}
</style>
</head>
<body>
<main>
<div class="header">
  <div>
    <h1>MCQ · Baseline vs Enrichment</h1>
    <div class="sub" id="subtitle"></div>
  </div>
  <div class="pills">
    <div class="pill" id="basePill"></div>
    <div class="pill" id="enrPill"></div>
    <div class="pill" id="retPill"></div>
  </div>
</div>

<div class="tabs">
  <button class="tab active" data-panel="overview">Visão geral</button>
  <button class="tab" data-panel="models">Modelos</button>
  <button class="tab" data-panel="configs">Retrievals</button>
  <button class="tab" data-panel="questions">Questões</button>
  <button class="tab" data-panel="hits">Hits enrichment</button>
</div>

<section class="panel active" id="panel-overview">
<div class="grid">
  <div class="card s3 kpi"><div class="label">Accuracy baseline</div><div class="value" id="baseAcc"></div><div class="detail">Run original congelado.</div></div>
  <div class="card s3 kpi"><div class="label">Accuracy enrichment</div><div class="value" id="enrAcc"></div><div class="detail" id="accDelta"></div></div>
  <div class="card s3 kpi"><div class="label">Melhoras / regressões</div><div class="value" id="changes"></div><div class="detail" id="changesDetail"></div></div>
  <div class="card s3 kpi"><div class="label">Contexto adicional</div><div class="value" id="contextCost"></div><div class="detail" id="contextDetail"></div></div>

  <div class="card s8">
    <h2>Accuracy pareada</h2>
    <div class="compare">
      <div><div class="label">Baseline</div><div class="big" id="boxBase"></div></div>
      <div><div class="label">Enrichment</div><div class="big" id="boxEnr"></div></div>
    </div>
    <div class="note" style="margin-top:14px" id="ciNote"></div>
  </div>

  <div class="card s4">
    <h2>Transições</h2>
    <div id="transitions"></div>
  </div>

  <div class="card s6">
    <h2>Latência média de geração</h2>
    <div id="latencyBars"></div>
  </div>

  <div class="card s6">
    <h2>Tokens de entrada</h2>
    <div id="tokenBars"></div>
  </div>

  <div class="card s12">
    <h2>Leitura metodológica</h2>
    <div class="note">
      Comparação pareada por <b>query_id × config_id × model</b>. O retrieval textual
      original permanece congelado; o segundo run apenas adiciona os resumos recuperados.
      O CI95 do delta de accuracy é calculado por bootstrap em <b>query_id</b>.
    </div>
  </div>
</div>
</section>

<section class="panel" id="panel-models">
<div class="grid">
  <div class="card s12"><h2>Δ accuracy por modelo</h2><div id="modelBars"></div></div>
  <div class="card s12"><div class="table-wrap"><table id="modelTable"></table></div></div>
</div>
</section>

<section class="panel" id="panel-configs">
<div class="grid">
  <div class="card s12"><h2>Δ accuracy por retrieval config</h2><div id="configBars"></div></div>
  <div class="card s12"><div class="table-wrap"><table id="configTable"></table></div></div>
</div>
</section>

<section class="panel" id="panel-questions">
<div class="grid">
  <div class="card s12">
    <div class="toolbar">
      <input id="qSearch" placeholder="Filtrar pergunta, obra, tipo ou dificuldade">
      <select id="qMode">
        <option value="all">Todas</option>
        <option value="improved">Melhoraram</option>
        <option value="regressed">Pioraram</option>
        <option value="unchanged">Sem mudança</option>
      </select>
    </div>
    <div class="table-wrap"><table id="questionTable"></table></div>
  </div>
</div>
</section>

<section class="panel" id="panel-hits">
<div class="grid">
  <div class="card s12">
    <div class="toolbar"><input id="hitSearch" placeholder="Filtrar query, obra, config ou embedder"></div>
    <div class="note" style="margin-bottom:12px">
      Amostras reais do <code>contexts.parquet</code> enriquecido: quais summaries
      entraram no contexto e com qual score.
    </div>
    <div id="hitList"></div>
  </div>
</div>
</section>

<div class="footer">Dashboard self-contained · RustFS benchmark artifacts</div>
</main>

<script>
const D=__DATA__;
const $=id=>document.getElementById(id);
const pct=v=>`${(Number(v||0)*100).toFixed(1)}%`;
const pp=v=>`${Number(v||0)>=0?'+':''}${(Number(v||0)*100).toFixed(1)} pp`;
const n=(v,d=1)=>Number(v||0).toLocaleString('pt-BR',{minimumFractionDigits:d,maximumFractionDigits:d});
const cls=v=>Number(v||0)>0?'good':Number(v||0)<0?'bad':'';
const esc=x=>String(x??'').replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;').replaceAll('"','&quot;');

function pairBars(target,a,b,fmt){
  const m=Math.max(Number(a||0),Number(b||0),1);
  $(target).innerHTML=`
    <div class="bar-row"><div>Baseline</div><div class="track"><div class="fill" style="width:${100*Number(a||0)/m}%"></div></div><div class="num">${fmt(a)}</div></div>
    <div class="bar-row"><div>Enrichment</div><div class="track"><div class="fill" style="width:${100*Number(b||0)/m}%"></div></div><div class="num">${fmt(b)}</div></div>`;
}

function deltaBars(target,rows,key){
  const max=Math.max(...rows.map(r=>Math.abs(Number(r.accuracy_delta||0))),.001);
  $(target).innerHTML=rows.map(r=>{
    const v=Number(r.accuracy_delta||0);
    return `<div class="bar-row">
      <div>${esc(r[key])}</div>
      <div class="track"><div class="fill ${v<0?'bad':''}" style="width:${Math.max(2,100*Math.abs(v)/max)}%"></div></div>
      <div class="num ${cls(v)}">${pp(v)}</div>
    </div>`;
  }).join('');
}

function statTable(target,rows,key){
  $(target).innerHTML=`
  <thead><tr>
    <th>${esc(key)}</th><th class="num">Base</th><th class="num">Enr.</th>
    <th class="num">Δ</th><th class="num">CI low</th><th class="num">CI high</th>
    <th class="num">+</th><th class="num">-</th><th class="num">Δ tokens</th><th class="num">Δ latency</th>
  </tr></thead>
  <tbody>${rows.map(r=>`<tr>
    <td>${esc(r[key])}</td>
    <td class="num">${pct(r.baseline_accuracy)}</td>
    <td class="num">${pct(r.enriched_accuracy)}</td>
    <td class="num ${cls(r.accuracy_delta)}">${pp(r.accuracy_delta)}</td>
    <td class="num">${pp(r.ci95_low)}</td>
    <td class="num">${pp(r.ci95_high)}</td>
    <td class="num good">${r.improved_rows}</td>
    <td class="num bad">${r.regressed_rows}</td>
    <td class="num">${Number(r.input_tokens_delta||0)>=0?'+':''}${n(r.input_tokens_delta,0)}</td>
    <td class="num">${Number(r.latency_delta_ms||0)>=0?'+':''}${n(r.latency_delta_ms,0)} ms</td>
  </tr>`).join('')}</tbody>`;
}

function renderQuestions(){
  const term=$('qSearch').value.toLowerCase().trim();
  const mode=$('qMode').value;
  const rows=D.questions.filter(r=>{
    const blob=[r.query,r.canonical_work,r.query_type,r.difficulty,r.query_id].join(' ').toLowerCase();
    const d=Number(r.accuracy_delta||0);
    if(term&&!blob.includes(term)) return false;
    if(mode==='improved'&&d<=0) return false;
    if(mode==='regressed'&&d>=0) return false;
    if(mode==='unchanged'&&d!==0) return false;
    return true;
  });
  $('questionTable').innerHTML=`
  <thead><tr>
    <th>Questão</th><th>Obra</th><th>Tipo</th><th>Dificuldade</th>
    <th class="num">Base</th><th class="num">Enr.</th><th class="num">Δ</th>
    <th class="num">+</th><th class="num">-</th><th class="num">Chars enr.</th>
  </tr></thead>
  <tbody>${rows.map(r=>`<tr>
    <td><b>${esc(r.query)}</b><div class="hit-meta">${esc(r.query_id)}</div></td>
    <td>${esc(r.canonical_work)}</td><td>${esc(r.query_type)}</td><td>${esc(r.difficulty)}</td>
    <td class="num">${pct(r.baseline_accuracy)}</td><td class="num">${pct(r.enriched_accuracy)}</td>
    <td class="num ${cls(r.accuracy_delta)}">${pp(r.accuracy_delta)}</td>
    <td class="num good">${r.improved_rows}</td><td class="num bad">${r.regressed_rows}</td>
    <td class="num">${n(r.enrichment_context_chars_mean,0)}</td>
  </tr>`).join('')}</tbody>`;
}

function hText(h){return h.text||h.resumo||h.summary||h.document_text||''}
function hWork(h){return h.titulo||h.canonical_work||h.work||''}
function hSec(h){return h.section_label||h.section_title||h.section_header||''}

function renderHits(){
  const term=$('hitSearch').value.toLowerCase().trim();
  const rows=D.enrichment_samples.filter(r=>
    !term||[r.query,r.canonical_work,r.config_label,r.embedder,r.query_id].join(' ').toLowerCase().includes(term)
  );
  $('hitList').innerHTML=rows.length?rows.map(r=>`
    <div class="hit-card">
      <div class="hit-title">${esc(r.query||r.query_id)}</div>
      <div class="hit-meta">gold=${esc(r.canonical_work)} · ${esc(r.config_label)} · ${esc(r.embedder)} · ${r.hit_count} hits · ${n(r.context_chars,0)} chars</div>
      ${(r.hits||[]).map((h,i)=>`
        <div class="hit">
          <span class="score">#${i+1} ${Number(h.score||0).toFixed(4)}</span>
          · ${esc(h.document_type||'summary')} · ${esc(hWork(h))}${hSec(h)?' · '+esc(hSec(h)):''}
          <div class="summary">${esc(hText(h))}</div>
        </div>`).join('')}
    </div>`).join(''):'<div class="note">Nenhum resultado para o filtro.</div>';
}

function init(){
  const o=D.overall;
  $('subtitle').textContent=`${o.queries} questões · ${o.rows.toLocaleString('pt-BR')} pares · bootstrap ${D.bootstrap.runs}`;
  $('basePill').textContent=`baseline: ${D.baseline_run_id}`;
  $('enrPill').textContent=`enriched: ${D.enriched_run_id}`;
  $('retPill').textContent=`retrieval: ${D.source_retrieval_run_id}`;

  $('baseAcc').textContent=pct(o.baseline_accuracy);
  $('enrAcc').textContent=pct(o.enriched_accuracy);
  $('enrAcc').className=`value ${cls(o.accuracy_delta)}`;
  $('accDelta').innerHTML=`<span class="tag ${cls(o.accuracy_delta)}">${pp(o.accuracy_delta)}</span> · CI95 ${pp(o.ci95_low)} → ${pp(o.ci95_high)}`;

  $('changes').textContent=`${o.improved_rows} / ${o.regressed_rows}`;
  $('changesDetail').textContent=`melhoras / regressões · ${o.unchanged_rows} sem mudança`;

  $('contextCost').textContent=`${n(o.enrichment_context_chars_mean,0)} chars`;
  $('contextDetail').textContent=`${n(o.enrichment_hit_count_mean,1)} hits por contexto em média`;

  $('boxBase').textContent=pct(o.baseline_accuracy);
  $('boxEnr').textContent=pct(o.enriched_accuracy);
  $('boxEnr').className=`big ${cls(o.accuracy_delta)}`;
  $('ciNote').innerHTML=`Δ observado: <b class="${cls(o.accuracy_delta)}">${pp(o.accuracy_delta)}</b> · CI95 por query_id: <b>${pp(o.ci95_low)}</b> a <b>${pp(o.ci95_high)}</b>.`;

  const tmax=Math.max(...D.transitions.map(x=>x.n),1);
  $('transitions').innerHTML=D.transitions.slice(0,10).map(t=>`
    <div class="bar-row">
      <div>${esc(t.baseline)} → ${esc(t.enriched)}</div>
      <div class="track"><div class="fill" style="width:${100*t.n/tmax}%"></div></div>
      <div class="num">${t.n}</div>
    </div>`).join('');

  pairBars('latencyBars',o.baseline_latency_mean_ms,o.enriched_latency_mean_ms,v=>`${n(v,0)} ms`);
  pairBars('tokenBars',o.baseline_input_tokens_mean,o.enriched_input_tokens_mean,v=>n(v,0));

  deltaBars('modelBars',D.models,'model');
  statTable('modelTable',D.models,'model');

  deltaBars('configBars',D.configs,'label');
  statTable('configTable',D.configs,'label');

  renderQuestions();
  renderHits();

  document.querySelectorAll('.tab').forEach(btn=>btn.addEventListener('click',()=>{
    document.querySelectorAll('.tab').forEach(x=>x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x=>x.classList.remove('active'));
    btn.classList.add('active');
    $(`panel-${btn.dataset.panel}`).classList.add('active');
  }));

  $('qSearch').addEventListener('input',renderQuestions);
  $('qMode').addEventListener('change',renderQuestions);
  $('hitSearch').addEventListener('input',renderHits);
}
init();
</script>
</body>
</html>'''

    return template.replace(
        "__DATA__",
        safe_json(payload),
    )


def main() -> None:
    args = parse_args()

    if args.bootstrap_runs < 0:
        raise ValueError(
            "--bootstrap-runs deve ser >= 0."
        )

    config = ConfigManager.load()
    storage = RustFSManager.from_config(
        config.environment.rustfs
    )
    bucket = config.storage.benchmarks_bucket

    (
        base_cfg,
        base_manifest,
        baseline,
        _,
    ) = load_run(
        storage,
        bucket,
        args.baseline_run_id,
    )

    (
        enr_cfg,
        enr_manifest,
        enriched,
        enr_contexts,
    ) = load_run(
        storage,
        bucket,
        args.enriched_run_id,
    )

    if bool(
        (
            base_cfg.get("enrichment")
            or {}
        ).get("enabled", False)
    ):
        raise RuntimeError(
            "O baseline informado já possui enrichment."
        )

    if not bool(
        (
            enr_cfg.get("enrichment")
            or {}
        ).get("enabled", False)
    ):
        raise RuntimeError(
            "O run enriched informado não possui enrichment."
        )

    source_retrieval = str(
        base_cfg.get("source_retrieval_run_id")
        or base_manifest.get("source_retrieval_run_id")
        or ""
    )

    enriched_source_retrieval = str(
        enr_cfg.get("source_retrieval_run_id")
        or enr_manifest.get("source_retrieval_run_id")
        or ""
    )

    if (
        source_retrieval
        != enriched_source_retrieval
    ):
        raise RuntimeError(
            "Os runs usam source_retrieval_run_id diferentes."
        )

    if (
        base_cfg.get("benchmark_sha256")
        != enr_cfg.get("benchmark_sha256")
    ):
        raise RuntimeError(
            "Os runs usam benchmark_sha256 diferentes."
        )

    required = set(
        PAIR_KEYS
        + [
            "is_correct",
            "is_ambiguous",
            "is_wrong",
            "is_invalid",
            "selected_class",
        ]
    )

    for label, frame in (
        ("baseline", baseline),
        ("enriched", enriched),
    ):
        missing = sorted(
            required
            - set(frame.columns)
        )

        if missing:
            raise RuntimeError(
                f"{label} sem colunas obrigatórias: {missing}"
            )

        if frame.duplicated(
            PAIR_KEYS
        ).any():
            raise RuntimeError(
                f"{label} possui pares duplicados."
            )

    paired = baseline.merge(
        enriched,
        on=PAIR_KEYS,
        how="inner",
        suffixes=(
            "_baseline",
            "_enriched",
        ),
        validate="one_to_one",
    )

    if (
        len(paired) != len(baseline)
        or len(paired) != len(enriched)
    ):
        raise RuntimeError(
            "Baseline e enriched não possuem o mesmo escopo. "
            f"baseline={len(baseline)} "
            f"enriched={len(enriched)} "
            f"paired={len(paired)}"
        )

    paired["accuracy_delta"] = (
        paired["is_correct_enriched"].astype(int)
        - paired["is_correct_baseline"].astype(int)
    )

    overall = aggregate(
        paired,
        bootstrap_runs=args.bootstrap_runs,
        seed=args.seed,
    )

    config_lookup = load_config_labels(
        storage,
        bucket,
        source_retrieval,
    )

    models = build_breakdown(
        paired,
        column="model",
        output_name="model",
        bootstrap_runs=args.bootstrap_runs,
        seed=args.seed,
    )

    configs = build_breakdown(
        paired,
        column="config_id",
        output_name="config_id",
        bootstrap_runs=args.bootstrap_runs,
        seed=args.seed,
    )

    for row in configs:
        meta = config_lookup.get(
            row["config_id"],
            {},
        )
        row["label"] = (
            meta.get("label")
            or row["config_id"]
        )
        row["selection_rank"] = (
            meta.get("selection_rank")
        )
        row["embedder"] = (
            meta.get("embedder")
        )
        row["retrieval_strategy"] = (
            meta.get("retrieval_strategy")
        )
        row["chunk_strategy"] = (
            meta.get("chunk_strategy")
        )
        row["chunk_size"] = (
            meta.get("chunk_size")
        )

    configs.sort(
        key=lambda item: (
            int(
                item.get("selection_rank")
                or 999
            ),
            item["config_id"],
        )
    )

    questions = build_question_rows(
        paired
    )

    question_lookup = {
        item["query_id"]: item
        for item in questions
    }

    payload = {
        "baseline_run_id": (
            args.baseline_run_id
        ),
        "enriched_run_id": (
            args.enriched_run_id
        ),
        "source_retrieval_run_id": (
            source_retrieval
        ),
        "bootstrap": {
            "unit": "query_id",
            "runs": args.bootstrap_runs,
            "seed": args.seed,
        },
        "enrichment": (
            enr_cfg.get("enrichment")
            or {}
        ),
        "overall": overall,
        "models": [
            {
                key: scalar(value)
                for key, value
                in row.items()
            }
            for row in models
        ],
        "configs": [
            {
                key: scalar(value)
                for key, value
                in row.items()
            }
            for row in configs
        ],
        "questions": questions,
        "transitions": build_transitions(
            paired
        ),
        "enrichment_samples": build_hit_samples(
            enr_contexts,
            question_lookup,
            config_lookup,
        ),
    }

    html = render_html(
        payload
    )

    output = (
        args.output
        or (
            Path("reports")
            / (
                "mcq_enrichment_"
                f"{args.enriched_run_id}.html"
            )
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        html,
        encoding="utf-8",
    )

    print(
        "=== MCQ ENRICHMENT REPORT OK ==="
    )
    print(
        f"baseline: {args.baseline_run_id}"
    )
    print(
        f"enriched: {args.enriched_run_id}"
    )
    print(
        f"paired rows: {overall['rows']}"
    )
    print(
        f"queries: {overall['queries']}"
    )
    print(
        "baseline accuracy: "
        f"{overall['baseline_accuracy']:.3f}"
    )
    print(
        "enriched accuracy: "
        f"{overall['enriched_accuracy']:.3f}"
    )
    print(
        "delta: "
        f"{overall['accuracy_delta']:+.3f} "
        "CI95 ["
        f"{overall['ci95_low']:+.3f}, "
        f"{overall['ci95_high']:+.3f}]"
    )
    print(
        f"local: {output.resolve()}"
    )

    if not args.no_upload:
        key = (
            f"{BENCHMARK_PREFIX}/runs/"
            f"{args.enriched_run_id}/"
            f"reports/{REPORT_NAME}"
        )

        storage.upload_bytes(
            bucket,
            key,
            html.encode("utf-8"),
            content_type=(
                "text/html; charset=utf-8"
            ),
        )

        print(
            f"rustfs: s3://{bucket}/{key}"
        )

    if args.open:
        webbrowser.open(
            output.resolve().as_uri()
        )


if __name__ == "__main__":
    main()
