from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from src.retrieval.types import RetrievalResponse

DEFAULT_KS = (1, 3, 5, 10)


def evaluate_response(
    response: RetrievalResponse,
    relevant_chunk_ids: Iterable[str],
    *,
    ks: tuple[int, ...] = DEFAULT_KS,
) -> dict:
    """
    Recall@k é query-level e binário: qualquer chunk que represente a
    evidência conta como sucesso. Duplicatas causadas por overlap não inflam
    a métrica.
    """
    relevant = {
        str(value) for value in relevant_chunk_ids if value is not None and str(value)
    }
    ranked = [hit.chunk_id for hit in response.hits]
    metrics = {}

    for k in ks:
        success = any(chunk_id in relevant for chunk_id in ranked[:k])
        value = 1.0 if success else 0.0
        metrics[f"hit_at_{k}"] = value
        metrics[f"recall_at_{k}"] = value

    first_rank = next(
        (
            rank
            for rank, chunk_id in enumerate(ranked[:10], start=1)
            if chunk_id in relevant
        ),
        None,
    )
    metrics["first_relevant_rank"] = first_rank
    metrics["mrr_at_10"] = 0.0 if first_rank is None else 1.0 / first_rank
    return metrics


def aggregate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """
    Qualidade: média por método.

    Latência: mean, p50 e p95 por estágio e total.
    """
    quality_columns = [
        column
        for column in results.columns
        if (
            column.startswith("hit_at_")
            or column.startswith("recall_at_")
            or column == "mrr_at_10"
        )
    ]

    timing_columns = [column for column in results.columns if column.endswith("_ms")]

    rows = []

    for method, group in results.groupby(
        "method",
        sort=False,
    ):
        row = {
            "method": method,
        }

        for column in quality_columns:
            row[column] = pd.to_numeric(
                group[column],
                errors="coerce",
            ).mean()

        for column in timing_columns:
            values = pd.to_numeric(
                group[column],
                errors="coerce",
            ).dropna()

            if values.empty:
                continue

            row[f"{column}_mean"] = float(values.mean())
            row[f"{column}_p50"] = float(values.quantile(0.50))
            row[f"{column}_p95"] = float(values.quantile(0.95))

        rows.append(row)

    return pd.DataFrame(rows)
