from __future__ import annotations

import math
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any, Iterable

import numpy as np

EVAL_KS = (1, 3, 5, 10)


def normalize_for_matching(value: Any) -> str:
    """Normalização conservadora para comparar gold evidence e chunks.

    Não reescreve o conteúdo: apenas casefold, remove acentos/pontuação e
    colapsa espaços para tornar o matching robusto a diferenças de formatação.
    """
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


def _matched_gold_positions(
    gold_normalized: str,
    candidate_normalized: str,
    *,
    min_match_chars: int = 8,
) -> set[int]:
    if not gold_normalized or not candidate_normalized:
        return set()

    matcher = SequenceMatcher(
        None,
        gold_normalized,
        candidate_normalized,
        autojunk=False,
    )

    covered: set[int] = set()
    for block in matcher.get_matching_blocks():
        if block.size < min_match_chars:
            continue
        covered.update(range(block.a, block.a + block.size))
    return covered


def _denominator_positions(gold_normalized: str) -> set[int]:
    return {
        index
        for index, char in enumerate(gold_normalized)
        if not char.isspace()
    }


def coverage_curve(
    gold_text: str,
    retrieved_texts: Iterable[str],
    *,
    min_match_chars: int = 8,
) -> list[dict[str, float]]:
    """Calcula cobertura individual, marginal e cumulativa da evidência gold.

    A cobertura cumulativa usa a UNIÃO das posições do gold cobertas pelos
    chunks recuperados até cada rank. Isso evita penalizar chunk sizes menores
    quando a evidência necessária está naturalmente dividida em vários chunks.
    """
    gold = normalize_for_matching(gold_text)
    denominator = _denominator_positions(gold)

    if not denominator:
        return [
            {
                "individual": 0.0,
                "marginal": 0.0,
                "cumulative": 0.0,
            }
            for _ in retrieved_texts
        ]

    seen: set[int] = set()
    output: list[dict[str, float]] = []

    for text in retrieved_texts:
        candidate = normalize_for_matching(text)
        matched = _matched_gold_positions(
            gold,
            candidate,
            min_match_chars=min_match_chars,
        )
        matched &= denominator

        newly_covered = matched - seen
        seen |= matched

        output.append(
            {
                "individual": len(matched) / len(denominator),
                "marginal": len(newly_covered) / len(denominator),
                "cumulative": len(seen) / len(denominator),
            }
        )

    return output


def _value_at_rank(values: list[float], k: int) -> float:
    if not values:
        return 0.0
    index = min(k, len(values)) - 1
    return float(values[index])


def ndcg_at_k(relevances: list[float], k: int) -> float:
    if not relevances or k <= 0:
        return 0.0

    values = [max(0.0, float(value)) for value in relevances[:k]]

    def dcg(items: list[float]) -> float:
        return sum(
            ((2.0**rel) - 1.0) / math.log2(rank + 1.0)
            for rank, rel in enumerate(items, start=1)
        )

    actual = dcg(values)
    ideal = dcg(sorted(values, reverse=True))
    return actual / ideal if ideal > 0 else 0.0


def query_metrics(
    *,
    gold_evidence: str,
    canonical_work: str,
    retrieved: list[dict[str, Any]],
    evidence_min_coverage: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    texts = [str(item.get("text") or "") for item in retrieved]
    curve = coverage_curve(gold_evidence, texts)

    canonical_work_norm = normalize_for_matching(canonical_work)
    cumulative_values: list[float] = []
    individual_values: list[float] = []
    correct_work_values: list[bool] = []
    rank_rows: list[dict[str, Any]] = []

    first_hit_rank: int | None = None

    for rank, (item, coverage) in enumerate(zip(retrieved, curve), start=1):
        retrieved_work = str(
            item.get("titulo")
            or item.get("work")
            or item.get("canonical_work")
            or ""
        )
        correct_work = (
            bool(canonical_work_norm)
            and normalize_for_matching(retrieved_work) == canonical_work_norm
        )

        cumulative = float(coverage["cumulative"])
        individual = float(coverage["individual"])
        cumulative_values.append(cumulative)
        individual_values.append(individual)
        correct_work_values.append(correct_work)

        hit = cumulative >= evidence_min_coverage
        if hit and first_hit_rank is None:
            first_hit_rank = rank

        rank_rows.append(
            {
                **item,
                "rank": rank,
                "individual_evidence_coverage": individual,
                "marginal_evidence_coverage": float(coverage["marginal"]),
                "cumulative_evidence_coverage": cumulative,
                "evidence_hit_at_rank": hit,
                "correct_work": correct_work,
            }
        )

    metrics: dict[str, Any] = {
        "first_hit_rank": first_hit_rank,
        "mrr_at_10": (
            1.0 / first_hit_rank
            if first_hit_rank is not None and first_hit_rank <= 10
            else 0.0
        ),
        "ndcg_at_10": ndcg_at_k(individual_values, 10),
        "coverage_auc_at_10": (
            float(np.mean(cumulative_values[:10]))
            if cumulative_values
            else 0.0
        ),
    }

    for k in EVAL_KS:
        coverage_at_k = _value_at_rank(cumulative_values, k)
        metrics[f"evidence_coverage_at_{k}"] = coverage_at_k
        metrics[f"hit_at_{k}"] = coverage_at_k >= evidence_min_coverage
        metrics[f"correct_work_at_{k}"] = any(correct_work_values[:k])

    return rank_rows, metrics
