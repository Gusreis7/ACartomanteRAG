from __future__ import annotations

import argparse
import html
import io
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager


RUNS_PREFIX = "retrieval/v1/runs"

METHOD_LABELS = {
    "bm25": "BM25",
    "dense": "Dense",
    "dense_ce": "Dense + CrossEncoder",
    "dense_cross_encoder": "Dense + CrossEncoder",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gera relatório HTML para benchmark de retrieval."
    )

    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)

    parser.add_argument(
        "--run-prefix",
        default=None,
        help=(
            "Prefixo exato do run. Se omitido, usa o run "
            "mais recente em retrieval/v1/runs/."
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Arquivo HTML local. Padrão: "
            "reports/retrieval/<run_id>.html"
        ),
    )

    parser.add_argument(
        "--upload",
        action="store_true",
        help="Também envia report.html para o mesmo run no RustFS.",
    )

    return parser.parse_args()


def load_config(args: argparse.Namespace):
    kwargs = {}

    if args.env_file is not None:
        kwargs["env_file"] = args.env_file

    if args.config_file is not None:
        kwargs["yaml_file"] = args.config_file

    return ConfigManager.load(**kwargs)


def resolve_latest_run(
    rustfs: RustFSManager,
    bucket: str,
) -> str:
    keys = rustfs.list_objects(
        bucket=bucket,
        prefix=f"{RUNS_PREFIX}/",
    )

    manifests = sorted(
        key
        for key in keys
        if key.endswith("/manifest.json")
    )

    if not manifests:
        raise FileNotFoundError(
            "Nenhum benchmark encontrado em "
            f"s3://{bucket}/{RUNS_PREFIX}/"
        )

    return manifests[-1].rsplit(
        "/manifest.json",
        1,
    )[0]


def load_run(
    rustfs: RustFSManager,
    bucket: str,
    prefix: str,
) -> dict[str, Any]:
    prefix = prefix.rstrip("/")

    keys = {
        "manifest": f"{prefix}/manifest.json",
        "summary": f"{prefix}/summary.csv",
        "results": f"{prefix}/results.csv",
        "evidence": f"{prefix}/evidence_matches.csv",
    }

    missing = [
        key
        for key in keys.values()
        if not rustfs.object_exists(bucket, key)
    ]

    if missing:
        raise FileNotFoundError(
            "Run incompleto. Objetos ausentes: "
            + ", ".join(
                f"s3://{bucket}/{key}"
                for key in missing
            )
        )

    manifest = json.loads(
        rustfs.download_bytes(
            bucket,
            keys["manifest"],
        ).decode("utf-8")
    )

    def load_csv(key: str) -> pd.DataFrame:
        return pd.read_csv(
            io.BytesIO(
                rustfs.download_bytes(
                    bucket,
                    key,
                )
            )
        )

    return {
        "manifest": manifest,
        "summary": load_csv(keys["summary"]),
        "results": load_csv(keys["results"]),
        "evidence": load_csv(keys["evidence"]),
    }


def esc(value: Any) -> str:
    if value is None:
        return "—"

    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass

    return html.escape(str(value))


def fmt_ms(value: Any) -> str:
    try:
        if pd.isna(value):
            return "—"
        value = float(value)
    except (TypeError, ValueError):
        return "—"

    if value >= 1000:
        return f"{value / 1000:.2f} s"

    return f"{value:.1f} ms"


def fmt_float(
    value: Any,
    digits: int = 3,
) -> str:
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return "—"


def method_label(value: Any) -> str:
    value = str(value)
    return METHOD_LABELS.get(
        value,
        value,
    )


def metric_value(
    row: pd.Series,
    name: str,
) -> float | None:
    if name not in row.index:
        return None

    value = row.get(name)

    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def metric_cell(
    name: str,
    value: float | None,
) -> str:
    if value is None:
        display = "—"
        width = 0.0
    else:
        display = f"{value * 100:.1f}%"
        width = max(
            0.0,
            min(
                100.0,
                value * 100.0,
            ),
        )

    return f"""
    <div class="metric-cell">
      <div class="metric-head">
        <span>{esc(name)}</span>
        <strong>{display}</strong>
      </div>
      <div class="bar">
        <div class="bar-fill" style="width:{width:.2f}%"></div>
      </div>
    </div>
    """


def quality_cards(
    summary: pd.DataFrame,
) -> str:
    cards = []

    for _, row in summary.iterrows():
        method = method_label(
            row.get("method")
        )

        cards.append(
            f"""
            <article class="method-card">
              <div class="method-title">{esc(method)}</div>
              <div class="metric-grid">
                {metric_cell("Recall@1", metric_value(row, "recall_at_1"))}
                {metric_cell("Recall@5", metric_value(row, "recall_at_5"))}
                {metric_cell("Recall@10", metric_value(row, "recall_at_10"))}
                {metric_cell("MRR@10", metric_value(row, "mrr_at_10"))}
              </div>
            </article>
            """
        )

    return "".join(cards)


def latency_rows(
    summary: pd.DataFrame,
) -> str:
    rows = []

    for _, row in summary.iterrows():
        rows.append(
            "<tr>"
            f"<td><strong>{esc(method_label(row.get('method')))}</strong></td>"
            f"<td>{fmt_ms(row.get('total_ms_mean'))}</td>"
            f"<td>{fmt_ms(row.get('total_ms_p50'))}</td>"
            f"<td>{fmt_ms(row.get('total_ms_p95'))}</td>"
            f"<td>{fmt_ms(row.get('embedding_ms_p50'))}</td>"
            f"<td>{fmt_ms(row.get('qdrant_ms_p50'))}</td>"
            f"<td>{fmt_ms(row.get('rerank_ms_p50'))}</td>"
            "</tr>"
        )

    return "".join(rows)


def evidence_stats(
    evidence: pd.DataFrame,
) -> dict[str, Any]:
    if evidence.empty:
        return {
            "matched_queries": 0,
            "unmatched_queries": 0,
            "avg_coverage": 0.0,
            "modes": {},
        }

    working = evidence.copy()
    working["query_id"] = (
        working["query_id"]
        .astype(str)
    )

    matched = working[
        working["chunk_id"].notna()
    ]

    all_queries = (
        working["query_id"]
        .nunique()
    )

    matched_queries = (
        matched["query_id"]
        .nunique()
    )

    coverage = pd.to_numeric(
        matched.get(
            "coverage",
            pd.Series(dtype=float),
        ),
        errors="coerce",
    )

    modes = (
        matched.get(
            "match_mode",
            pd.Series(dtype=str),
        )
        .fillna("unknown")
        .astype(str)
        .value_counts()
        .to_dict()
    )

    return {
        "matched_queries": int(
            matched_queries
        ),
        "unmatched_queries": int(
            all_queries - matched_queries
        ),
        "avg_coverage": float(
            coverage.mean()
            if not coverage.empty
            else 0.0
        ),
        "modes": modes,
    }


def failure_rows(
    results: pd.DataFrame,
    limit: int = 40,
) -> str:
    if results.empty:
        return (
            '<tr><td colspan="7">'
            "Sem resultados.</td></tr>"
        )

    working = results.copy()

    mrr = pd.to_numeric(
        working.get(
            "mrr_at_10",
            pd.Series(
                0,
                index=working.index,
            ),
        ),
        errors="coerce",
    ).fillna(0)

    working = (
        working[
            mrr.eq(0)
        ]
        .sort_values(
            by=[
                "query_id",
                "method",
            ]
        )
        .head(limit)
    )

    if working.empty:
        return (
            '<tr><td colspan="7">'
            "Nenhuma falha em MRR@10.</td></tr>"
        )

    rows = []

    for _, row in working.iterrows():
        rows.append(
            "<tr>"
            f"<td>{esc(row.get('query_id'))}</td>"
            f"<td>{esc(row.get('query'))}</td>"
            f"<td>{esc(row.get('canonical_work'))}</td>"
            f"<td>{esc(method_label(row.get('method')))}</td>"
            f"<td>{esc(row.get('relevant_chunks'))}</td>"
            f"<td>{esc(row.get('first_relevant_rank'))}</td>"
            f"<td>{fmt_ms(row.get('total_ms'))}</td>"
            "</tr>"
        )

    return "".join(rows)


def compact_run_name(
    prefix: str,
) -> str:
    return prefix.rstrip(
        "/"
    ).split("/")[-1]


def build_html(
    run_prefix: str,
    data: dict[str, Any],
) -> str:
    manifest = data["manifest"]
    summary = data["summary"]
    results = data["results"]
    evidence = data["evidence"]

    ev = evidence_stats(
        evidence
    )

    benchmark_rows = int(
        manifest.get(
            "benchmark_rows",
            0,
        )
        or 0
    )

    scored_queries = int(
        manifest.get(
            "scored_queries",
            ev["matched_queries"],
        )
        or 0
    )

    evidence_pct = (
        scored_queries
        / benchmark_rows
        * 100.0
        if benchmark_rows
        else 0.0
    )

    run_id = manifest.get(
        "run_id",
        compact_run_name(
            run_prefix
        ),
    )

    warmup = (
        manifest.get("warmup")
        or {}
    )

    reranker = (
        manifest.get("reranker")
        or {}
    )

    match_modes = ", ".join(
        f"{key}: {value}"
        for key, value
        in ev["modes"].items()
    ) or "—"

    css = """
    :root {
      --bg: #f8faff;
      --surface: #ffffff;
      --text: #182033;
      --muted: #6d7890;
      --line: #e5e9f5;
      --blue: #316ff6;
      --purple: #7658f6;
      --shadow: 0 14px 42px rgba(45, 61, 102, .08);
      --radius: 18px;
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      color: var(--text);
      background:
        radial-gradient(circle at 10% 0%, rgba(49,111,246,.08), transparent 30%),
        radial-gradient(circle at 90% 5%, rgba(118,88,246,.08), transparent 28%),
        var(--bg);
      font-family:
        Inter,
        ui-sans-serif,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif;
      line-height: 1.5;
    }

    .page {
      width: min(1180px, calc(100% - 40px));
      margin: 0 auto;
      padding: 52px 0 76px;
    }

    .hero {
      display: flex;
      justify-content: space-between;
      align-items: flex-end;
      gap: 28px;
      margin-bottom: 26px;
    }

    .eyebrow {
      color: var(--blue);
      font-size: 12px;
      font-weight: 800;
      letter-spacing: .14em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }

    h1 {
      margin: 0;
      font-size: clamp(34px, 5vw, 54px);
      line-height: 1;
      letter-spacing: -.045em;
    }

    h2 {
      margin: 0 0 16px;
      font-size: 20px;
      letter-spacing: -.02em;
    }

    .subtitle {
      margin-top: 14px;
      color: var(--muted);
      max-width: 720px;
    }

    .badge {
      background: linear-gradient(135deg, var(--blue), var(--purple));
      color: white;
      padding: 9px 14px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }

    .top-grid {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 14px;
    }

    .panel {
      margin-top: 18px;
      padding: 22px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: rgba(255,255,255,.94);
      box-shadow: var(--shadow);
    }

    .stat {
      background: white;
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 17px 18px;
    }

    .stat-label {
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 6px;
    }

    .stat-value {
      font-size: 22px;
      font-weight: 780;
      letter-spacing: -.03em;
      overflow-wrap: anywhere;
    }

    .stat-value.small {
      font-size: 13px;
      line-height: 1.4;
      letter-spacing: 0;
    }

    .method-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 14px;
    }

    .method-card {
      padding: 18px;
      background: linear-gradient(180deg, #fff, #fbfcff);
      border: 1px solid var(--line);
      border-radius: 16px;
    }

    .method-title {
      font-weight: 780;
      margin-bottom: 15px;
    }

    .metric-grid {
      display: grid;
      gap: 13px;
    }

    .metric-head {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      color: var(--muted);
      font-size: 12px;
    }

    .metric-head strong {
      color: var(--text);
    }

    .bar {
      height: 6px;
      margin-top: 6px;
      overflow: hidden;
      border-radius: 999px;
      background: #edf0f8;
    }

    .bar-fill {
      height: 100%;
      border-radius: inherit;
      background: linear-gradient(90deg, var(--blue), var(--purple));
    }

    .table-wrap {
      overflow-x: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
    }

    table {
      width: 100%;
      min-width: 760px;
      border-collapse: collapse;
      background: white;
    }

    th, td {
      padding: 12px 14px;
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid var(--line);
      font-size: 12px;
    }

    th {
      color: var(--muted);
      background: #fafbff;
      font-weight: 700;
    }

    tr:last-child td {
      border-bottom: 0;
    }

    .meta-grid {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 0 24px;
    }

    .meta-row {
      padding: 11px 0;
      border-bottom: 1px solid var(--line);
    }

    .meta-key {
      display: block;
      color: var(--muted);
      font-size: 11px;
      margin-bottom: 4px;
    }

    .meta-value {
      display: block;
      font-size: 13px;
      overflow-wrap: anywhere;
    }

    .footer {
      margin-top: 30px;
      color: var(--muted);
      font-size: 11px;
      text-align: center;
    }

    @media (max-width: 900px) {
      .top-grid,
      .method-grid,
      .meta-grid {
        grid-template-columns: 1fr 1fr;
      }
    }

    @media (max-width: 620px) {
      .page {
        width: calc(100% - 24px);
        padding-top: 28px;
      }

      .hero {
        display: block;
      }

      .badge {
        display: inline-block;
        margin-top: 16px;
      }

      .top-grid,
      .method-grid,
      .meta-grid {
        grid-template-columns: 1fr;
      }

      .panel {
        padding: 16px;
      }
    }
    """

    return f"""<!doctype html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Retrieval Benchmark · {esc(run_id)}</title>
  <style>{css}</style>
</head>
<body>
  <main class="page">
    <header class="hero">
      <div>
        <div class="eyebrow">ACartomanteRAG · Retrieval</div>
        <h1>Benchmark Report</h1>
        <div class="subtitle">
          BM25, Dense e Dense + CrossEncoder avaliados sobre
          a mesma versão da camada Silver.
        </div>
      </div>
      <div class="badge">{esc(run_id)}</div>
    </header>

    <section class="top-grid">
      <div class="stat">
        <div class="stat-label">Queries avaliadas</div>
        <div class="stat-value">{scored_queries}/{benchmark_rows}</div>
      </div>

      <div class="stat">
        <div class="stat-label">Evidence match</div>
        <div class="stat-value">{evidence_pct:.1f}%</div>
      </div>

      <div class="stat">
        <div class="stat-label">Silver dataset</div>
        <div class="stat-value small">
          {esc(manifest.get("silver_dataset_id"))}
        </div>
      </div>

      <div class="stat">
        <div class="stat-label">Index</div>
        <div class="stat-value small">
          {esc(manifest.get("index_id"))}
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Qualidade de retrieval</h2>
      <div class="method-grid">
        {quality_cards(summary)}
      </div>
    </section>

    <section class="panel">
      <h2>Latência</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Método</th>
              <th>Total mean</th>
              <th>Total p50</th>
              <th>Total p95</th>
              <th>Embedding p50</th>
              <th>Qdrant p50</th>
              <th>Rerank p50</th>
            </tr>
          </thead>
          <tbody>
            {latency_rows(summary)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Saúde do benchmark</h2>

      <div class="top-grid">
        <div class="stat">
          <div class="stat-label">Evidence matched</div>
          <div class="stat-value">{ev["matched_queries"]}</div>
        </div>

        <div class="stat">
          <div class="stat-label">Evidence unmatched</div>
          <div class="stat-value">{ev["unmatched_queries"]}</div>
        </div>

        <div class="stat">
          <div class="stat-label">Coverage média</div>
          <div class="stat-value">{ev["avg_coverage"] * 100:.1f}%</div>
        </div>

        <div class="stat">
          <div class="stat-label">Warm-up</div>
          <div class="stat-value">{esc(warmup.get("runs", 0))}×</div>
        </div>
      </div>

      <div class="meta-grid" style="margin-top:14px">
        <div class="meta-row">
          <span class="meta-key">Evidence matching</span>
          <span class="meta-value">{esc(match_modes)}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Evidence min coverage</span>
          <span class="meta-value">
            {fmt_float(manifest.get("evidence_min_coverage"), 2)}
          </span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Dense candidate K</span>
          <span class="meta-value">{esc(manifest.get("dense_candidate_k"))}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Top K final</span>
          <span class="meta-value">{esc(manifest.get("top_k"))}</span>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>Falhas em MRR@10</h2>

      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Query ID</th>
              <th>Query</th>
              <th>Obra</th>
              <th>Método</th>
              <th>Chunks relevantes</th>
              <th>Primeiro rank</th>
              <th>Latência</th>
            </tr>
          </thead>
          <tbody>
            {failure_rows(results)}
          </tbody>
        </table>
      </div>
    </section>

    <section class="panel">
      <h2>Execução</h2>

      <div class="meta-grid">
        <div class="meta-row">
          <span class="meta-key">Run prefix</span>
          <span class="meta-value">{esc(run_prefix)}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Criado em</span>
          <span class="meta-value">{esc(manifest.get("created_at_utc"))}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Collection</span>
          <span class="meta-value">{esc(manifest.get("collection"))}</span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Seleção</span>
          <span class="meta-value">
            {esc(manifest.get("selection"))}
            · release={esc(manifest.get("release"))}
          </span>
        </div>

        <div class="meta-row">
          <span class="meta-key">CrossEncoder</span>
          <span class="meta-value">
            {esc(reranker.get("model"))}
            @ {esc(reranker.get("revision"))}
          </span>
        </div>

        <div class="meta-row">
          <span class="meta-key">Warm-up total</span>
          <span class="meta-value">
            {fmt_ms(warmup.get("total_ms"))}
            · descartado das métricas
          </span>
        </div>
      </div>
    </section>

    <div class="footer">
      ACartomanteRAG · relatório estático gerado a partir dos artifacts do benchmark
    </div>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    config = load_config(args)

    rustfs = RustFSManager.from_config(
        config.environment.rustfs
    )

    bucket = (
        config.storage.benchmarks_bucket
    )

    run_prefix = (
        args.run_prefix
        or resolve_latest_run(
            rustfs,
            bucket,
        )
    )

    print(
        "[+] Run: "
        f"s3://{bucket}/{run_prefix}/"
    )

    data = load_run(
        rustfs,
        bucket,
        run_prefix,
    )

    report = build_html(
        run_prefix,
        data,
    )

    run_id = (
        data["manifest"].get(
            "run_id"
        )
        or compact_run_name(
            run_prefix
        )
    )

    output = (
        args.output
        or (
            Path("reports")
            / "retrieval"
            / f"{run_id}.html"
        )
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    output.write_text(
        report,
        encoding="utf-8",
    )

    print(
        "[OK] HTML: "
        f"{output.resolve()}"
    )

    if args.upload:
        key = (
            f"{run_prefix.rstrip('/')}/"
            "report.html"
        )

        rustfs.upload_bytes(
            bucket=bucket,
            key=key,
            data=report.encode(
                "utf-8"
            ),
            content_type=(
                "text/html; charset=utf-8"
            ),
        )

        print(
            "[OK] Upload: "
            f"s3://{bucket}/{key}"
        )


if __name__ == "__main__":
    main()
