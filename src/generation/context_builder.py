from __future__ import annotations

from typing import Any

from src.retrieval.types import RetrievalHit


ENRICHMENT_HEADING = (
    "CONTEXTO ENRIQUECIDO "
    "(DERIVADO; NÃO É EVIDÊNCIA PRIMÁRIA):"
)

TEXTUAL_HEADING = (
    "EVIDÊNCIAS TEXTUAIS ORIGINAIS:"
)


def build_context(
    hits: list[RetrievalHit],
    *,
    max_chars: int,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    API original mantida para compatibilidade.
    """
    return _build_textual_context(
        hits,
        max_chars=max_chars,
    )


def build_context_with_enrichment(
    hits: list[RetrievalHit],
    *,
    enrichment_hits: list[
        dict[str, Any]
    ] | None,
    max_chars: int,
    enrichment_max_chars: int,
    enrichment_max_fraction: float,
) -> tuple[
    str,
    list[dict[str, Any]],
    dict[str, int],
]:
    """
    Monta um orçamento conjunto.

    O enrichment pode consumir no máximo:
        min(
            enrichment_max_chars,
            max_chars * enrichment_max_fraction
        )

    O espaço não usado pelo enrichment volta automaticamente
    para as evidências textuais.

    Somente evidências [E] entram no contrato público.
    Os resumos [R] são contexto interno derivado.
    """
    total_budget = max(
        int(
            max_chars
        ),
        1,
    )

    fraction = min(
        max(
            float(
                enrichment_max_fraction
            ),
            0.0,
        ),
        1.0,
    )

    enrichment_budget = min(
        max(
            int(
                enrichment_max_chars
            ),
            0,
        ),
        int(
            total_budget
            * fraction
        ),
    )

    enrichment_context = ""
    enrichment_used: list[
        dict[str, Any]
    ] = []

    if (
        enrichment_budget > 0
        and enrichment_hits
    ):
        heading_cost = (
            len(
                ENRICHMENT_HEADING
            )
            + 1
        )

        body_budget = max(
            enrichment_budget
            - heading_cost,
            0,
        )

        (
            enrichment_body,
            enrichment_used,
        ) = format_enrichment_context(
            enrichment_hits,
            max_chars=(
                body_budget
            ),
        )

        if enrichment_body:
            enrichment_context = (
                ENRICHMENT_HEADING
                + "\n"
                + enrichment_body
            )

    prefix_cost = 0

    if enrichment_context:
        prefix_cost += (
            len(
                enrichment_context
            )
            + 2
        )

    prefix_cost += (
        len(
            TEXTUAL_HEADING
        )
        + 1
    )

    textual_budget = max(
        total_budget
        - prefix_cost,
        1,
    )

    textual_context, evidences = (
        _build_textual_context(
            hits,
            max_chars=(
                textual_budget
            ),
        )
    )

    parts: list[str] = []

    if enrichment_context:
        parts.append(
            enrichment_context
        )

    parts.append(
        TEXTUAL_HEADING
        + "\n"
        + textual_context
    )

    context = (
        "\n\n".join(
            parts
        )
        .strip()
    )

    if len(context) > total_budget:
        context = (
            context[
                :total_budget
            ]
            .rstrip()
        )

    usage = {
        "total_chars": len(
            context
        ),
        "enrichment_chars": len(
            enrichment_context
        ),
        "enrichment_hits": len(
            enrichment_used
        ),
        "textual_chars": len(
            textual_context
        ),
        "textual_evidences": len(
            evidences
        ),
    }

    return (
        context,
        evidences,
        usage,
    )


def format_enrichment_context(
    hits: list[
        dict[str, Any]
    ],
    *,
    max_chars: int,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    """
    Formato intencionalmente igual ao benchmark MCQ.
    """
    max_chars = max(
        int(
            max_chars
        ),
        0,
    )

    blocks: list[str] = []
    used: list[
        dict[str, Any]
    ] = []

    chars = 0

    for rank, hit in enumerate(
        hits,
        start=1,
    ):
        title = str(
            hit.get(
                "titulo"
            )
            or "não informada"
        )

        document_type = str(
            hit.get(
                "document_type"
            )
            or "summary"
        )

        section = str(
            hit.get(
                "section_label"
            )
            or ""
        ).strip()

        part_index = hit.get(
            "part_index"
        )

        part_count = hit.get(
            "part_count"
        )

        text = str(
            hit.get(
                "text"
            )
            or ""
        ).strip()

        events = [
            str(
                item
            ).strip()
            for item
            in (
                hit.get(
                    "events"
                )
                or []
            )
            if str(
                item
            ).strip()
        ]

        header = [
            f"[R{rank}]",
            f"Obra: {title}",
        ]

        if (
            document_type
            == "work_summary"
        ):
            header.append(
                "Tipo: resumo da obra"
            )

        else:
            header.append(
                "Tipo: resumo de seção"
            )

        if section:
            header.append(
                f"Seção: {section}"
            )

        if (
            part_index is not None
            and part_count is not None
        ):
            try:
                if int(
                    part_count
                ) > 1:
                    header.append(
                        "Parte: "
                        f"{part_index}/"
                        f"{part_count}"
                    )
            except Exception:
                pass

        body = (
            "\n".join(
                header
            )
            + "\nResumo: "
            + text
        )

        if events:
            body += (
                "\nEventos:\n"
                + "\n".join(
                    f"- {event}"
                    for event in events
                )
            )

        body += "\n\n"

        remaining = (
            max_chars
            - chars
        )

        if remaining <= 0:
            break

        truncated = False

        if len(body) > remaining:
            body = (
                body[
                    :remaining
                ]
                .rstrip()
                + " …\n"
            )

            truncated = True

        blocks.append(
            body
        )

        chars += len(
            body
        )

        used.append(
            {
                "rank": rank,
                "enrichment_document_id": (
                    hit.get(
                        "enrichment_document_id"
                    )
                ),
                "score": (
                    hit.get(
                        "score"
                    )
                ),
                "document_type": (
                    document_type
                ),
                "titulo": title,
                "section_label": (
                    section
                    or None
                ),
                "part_index": (
                    part_index
                ),
                "part_count": (
                    part_count
                ),
                "truncated": (
                    truncated
                ),
            }
        )

        if truncated:
            break

    return (
        "".join(
            blocks
        ).strip(),
        used,
    )


def _build_textual_context(
    hits: list[RetrievalHit],
    *,
    max_chars: int,
) -> tuple[
    str,
    list[dict[str, Any]],
]:
    max_chars = max(
        int(
            max_chars
        ),
        1,
    )

    blocks: list[str] = []

    evidences: list[
        dict[str, Any]
    ] = []

    used_chars = 0

    for index, hit in enumerate(
        hits,
        start=1,
    ):
        evidence_id = (
            f"E{index}"
        )

        metadata = (
            hit.metadata
            or {}
        )

        work = _first(
            metadata,
            (
                "titulo",
                "documento",
                "obra",
                "canonical_work",
            ),
        )

        section = _first(
            metadata,
            (
                "section_title",
                "section_heading",
                "section_label",
                "secao",
                "seção",
                "capitulo",
                "capítulo",
            ),
        )

        category = _first(
            metadata,
            (
                "categoria",
                "category",
            ),
        )

        header_lines = [
            f"[{evidence_id}]",
            (
                f"Obra: {work}"
                if work
                else (
                    "Obra: "
                    "não informada"
                )
            ),
        ]

        if section:
            header_lines.append(
                f"Seção: {section}"
            )

        if category:
            header_lines.append(
                f"Categoria: {category}"
            )

        header_lines.append(
            "Chunk ID: "
            f"{hit.chunk_id}"
        )

        prefix = (
            "\n".join(
                header_lines
            )
            + "\nTrecho: "
        )

        remaining = (
            max_chars
            - used_chars
            - len(
                prefix
            )
            - 2
        )

        if remaining <= 0:
            break

        text = str(
            hit.text
            or ""
        ).strip()

        truncated = False

        if len(text) > remaining:
            text = (
                text[
                    :remaining
                ]
                .rstrip()
            )

            if text:
                text += " …"

            truncated = True

        block = (
            prefix
            + text
            + "\n\n"
        )

        blocks.append(
            block
        )

        used_chars += len(
            block
        )

        evidences.append(
            {
                "evidence_id": (
                    evidence_id
                ),
                "chunk_id": (
                    hit.chunk_id
                ),
                "score": (
                    hit.score
                ),
                "obra": work,
                "secao": section,
                "categoria": (
                    category
                ),
                "excerpt": text,
                "truncated": (
                    truncated
                ),
            }
        )

        if truncated:
            break

    if not blocks:
        return (
            "Nenhuma evidência textual "
            "foi disponibilizada.",
            [],
        )

    return (
        "".join(
            blocks
        ).strip(),
        evidences,
    )


def _first(
    metadata: dict[str, Any],
    keys: tuple[str, ...],
) -> str | None:
    for key in keys:
        value = metadata.get(
            key
        )

        if value is None:
            continue

        value = str(
            value
        ).strip()

        if value:
            return value

    return None
