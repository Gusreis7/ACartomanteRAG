from __future__ import annotations

from difflib import SequenceMatcher

import pandas as pd

from src.retrieval.text import normalize_text, tokenize


def build_evidence_matches(
    benchmark: pd.DataFrame,
    chunks: pd.DataFrame,
    *,
    min_coverage: float = 0.60,
) -> pd.DataFrame:
    """
    Mapeia evidence_text -> chunk_ids para a Silver avaliada.

    Primeiro tenta containment textual normalizado. Se a evidência cruzar
    fronteira de chunk, usa a cobertura do maior trecho contíguo em tokens.
    Vários chunks podem ser relevantes por overlap; a métrica continua
    binária por query, então isso não infla Recall/Hit@k.
    """
    required_benchmark = {
        "query_id",
        "query",
        "canonical_work",
        "evidence_text",
    }
    missing = required_benchmark - set(benchmark.columns)
    if missing:
        raise ValueError(
            "Benchmark sem colunas obrigatórias: " + ", ".join(sorted(missing))
        )

    required_chunks = {"chunk_id", "titulo", "text"}
    missing_chunks = required_chunks - set(chunks.columns)
    if missing_chunks:
        raise ValueError(
            "Silver sem colunas para evidence matching: "
            + ", ".join(sorted(missing_chunks))
        )

    chunk_titles = chunks["titulo"].fillna("").astype(str).map(normalize_text)
    rows = []

    for _, item in benchmark.iterrows():
        query_id = str(item["query_id"])
        work = normalize_text(item["canonical_work"])
        evidence = str(item["evidence_text"])
        evidence_norm = normalize_text(evidence)

        candidate_indices = chunks.index[chunk_titles.eq(work)].tolist()
        exact_matches = []
        for idx in candidate_indices:
            chunk_norm = normalize_text(chunks.at[idx, "text"])
            if evidence_norm and evidence_norm in chunk_norm:
                exact_matches.append(idx)

        if exact_matches:
            selected = [(idx, 1.0, "exact") for idx in exact_matches]
        else:
            evidence_tokens = tokenize(evidence)
            scored = []
            for idx in candidate_indices:
                chunk_tokens = tokenize(chunks.at[idx, "text"])
                coverage = _contiguous_coverage(
                    evidence_tokens,
                    chunk_tokens,
                )
                scored.append((idx, coverage))

            best = max((coverage for _, coverage in scored), default=0.0)
            if best < min_coverage:
                selected = []
            else:
                tolerance = 0.02
                selected = [
                    (idx, coverage, "contiguous_overlap")
                    for idx, coverage in scored
                    if coverage >= max(min_coverage, best - tolerance)
                ]

        if not selected:
            rows.append(
                {
                    "query_id": query_id,
                    "chunk_id": None,
                    "coverage": 0.0,
                    "match_mode": "unmatched",
                }
            )
            continue

        for idx, coverage, mode in selected:
            rows.append(
                {
                    "query_id": query_id,
                    "chunk_id": str(chunks.at[idx, "chunk_id"]),
                    "coverage": float(coverage),
                    "match_mode": mode,
                }
            )

    return pd.DataFrame(rows)


def _contiguous_coverage(
    evidence_tokens: list[str],
    chunk_tokens: list[str],
) -> float:
    if not evidence_tokens or not chunk_tokens:
        return 0.0

    matcher = SequenceMatcher(
        a=evidence_tokens,
        b=chunk_tokens,
        autojunk=False,
    )
    best = matcher.find_longest_match(
        0,
        len(evidence_tokens),
        0,
        len(chunk_tokens),
    )
    return best.size / len(evidence_tokens)
