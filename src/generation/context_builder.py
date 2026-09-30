from __future__ import annotations

from typing import Any

from src.retrieval.types import RetrievalHit


def build_context(
    hits: list[RetrievalHit],
    *,
    max_chars: int,
) -> tuple[str, list[dict[str, Any]]]:
    max_chars = max(int(max_chars), 1)
    blocks: list[str] = []
    evidences: list[dict[str, Any]] = []
    used_chars = 0

    for index, hit in enumerate(hits, start=1):
        evidence_id = f"E{index}"
        metadata = hit.metadata or {}

        work = _first(
            metadata,
            ("titulo", "documento", "obra", "canonical_work"),
        )
        section = _first(
            metadata,
            (
                "section_title",
                "section_heading",
                "secao",
                "seção",
                "capitulo",
                "capítulo",
            ),
        )
        category = _first(metadata, ("categoria", "category"))

        header_lines = [
            f"[{evidence_id}]",
            f"Obra: {work}" if work else "Obra: não informada",
        ]
        if section:
            header_lines.append(f"Seção: {section}")
        if category:
            header_lines.append(f"Categoria: {category}")
        header_lines.append(f"Chunk ID: {hit.chunk_id}")

        prefix = "\n".join(header_lines) + "\nTrecho: "
        remaining = max_chars - used_chars - len(prefix) - 2
        if remaining <= 0:
            break

        text = str(hit.text or "").strip()
        truncated = False
        if len(text) > remaining:
            text = text[:remaining].rstrip()
            if text:
                text += " …"
            truncated = True

        block = prefix + text + "\n\n"
        blocks.append(block)
        used_chars += len(block)

        evidences.append(
            {
                "evidence_id": evidence_id,
                "chunk_id": hit.chunk_id,
                "score": hit.score,
                "obra": work,
                "secao": section,
                "categoria": category,
                "excerpt": text,
                "truncated": truncated,
            }
        )

        if truncated:
            break

    if not blocks:
        return "Nenhuma evidência textual foi disponibilizada.", []

    return "".join(blocks).strip(), evidences


def _first(metadata: dict[str, Any], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        value = metadata.get(key)
        if value is None:
            continue
        value = str(value).strip()
        if value:
            return value
    return None
