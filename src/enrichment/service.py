from __future__ import annotations

import hashlib
import json
import operator
import re
import unicodedata
from dataclasses import dataclass
from time import perf_counter
from typing import Annotated, Any, TypedDict
from urllib import request as urllib_request

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field

from src.utils.chunking import build_sections

MAX_EVENTS_PER_BLOCK = 5
MAX_EVENTS_FINAL = 15

UNIT_SYSTEM_PROMPT = (
    "Resuma fatos narrativos com concisão e fidelidade. "
    "Use somente o trecho fornecido e siga o schema."
)

UNIT_PROMPT_TEMPLATE = """\
Analise SOMENTE este trecho de "{title}", de {author}.

Extraia apenas:

1. RESUMO
Um resumo factual curto desta unidade.

2. EVENTOS
No máximo {max_events} acontecimentos importantes da unidade, em frases curtas.
Quando uma consequência estiver explícita, inclua-a na mesma frase.

Não extraia personagens.
Não extraia temas.
Não faça interpretação.
Não crie uma lista separada de pontos importantes.
Não use conhecimento externo.
Não repita o mesmo fato.

UNIDADE: {unit_id}
SEÇÃO: {section_label}
PARTE: {part_index}/{part_count}

TRECHO:
<<<
{text}
>>>
"""

FINAL_SYSTEM_PROMPT = (
    "Faça apenas uma consolidação narrativa factual e compacta. "
    "Use somente as anotações recebidas e siga o schema."
)

FINAL_PROMPT_TEMPLATE = """\
Consolide as anotações ordenadas de "{title}", de {author}.

Produza somente:

1. RESUMO GERAL
Um resumo factual e cronológico do material recebido.

2. EVENTOS PRINCIPAIS
No máximo {max_events} eventos importantes em ordem narrativa aproximada.

Não extraia personagens.
Não produza temas.
Não faça interpretação.
Não faça análise crítica.
Não crie pontos importantes separados.
Não use conhecimento externo.
Não complete lacunas de memória.
Evite repetir o mesmo evento.

ANOTAÇÕES:
{material}
"""


class UnitSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    resumo: str = Field(default="", description="Resumo factual curto da unidade.")
    eventos: list[str] = Field(
        default_factory=list,
        description="Eventos factuais mais relevantes da unidade.",
    )


class FinalNarrativeSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")
    resumo_geral: str = Field(
        default="",
        description="Resumo factual e cronológico do material.",
    )
    eventos_principais: list[str] = Field(
        default_factory=list,
        description="Eventos principais em ordem narrativa aproximada.",
    )


@dataclass(frozen=True)
class WorkSource:
    title: str
    author: str
    category: str
    source_document_id: str
    bronze_artifact_id: str
    bronze_object_key: str
    text: str


@dataclass(frozen=True)
class PlannedUnit:
    unit_id: str
    section_index: int
    section_label: str
    section_type: str
    part_index: int
    part_count: int
    start_char: int
    end_char: int
    text: str


@dataclass(frozen=True)
class UnitResult:
    unit_id: str
    section_index: int
    section_label: str
    section_type: str
    part_index: int
    part_count: int
    start_char: int
    end_char: int
    elapsed_seconds: float
    summary: UnitSummary


@dataclass(frozen=True)
class EnrichmentResult:
    plan: dict[str, Any]
    blocks: list[dict[str, Any]]
    summary: dict[str, Any]
    timings_seconds: dict[str, Any]
    reduce_calls: int


class EnrichmentState(TypedDict, total=False):
    source: WorkSource
    units: list[PlannedUnit]
    plan: dict[str, Any]
    unit_results: Annotated[list[UnitResult], operator.add]
    ordered_results: list[UnitResult]
    final_summary: FinalNarrativeSummary
    reduce_calls: int


class UnitTaskState(TypedDict):
    source: WorkSource
    unit: PlannedUnit


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def split_char_windows(
    text: str,
    *,
    max_chars: int,
    overlap: int,
) -> list[tuple[int, int, str]]:
    text = str(text)
    if max_chars <= 0:
        raise ValueError("max_chars deve ser > 0.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap deve ser >= 0 e < max_chars.")
    if len(text) <= max_chars:
        return [(0, len(text), text)]

    windows: list[tuple[int, int, str]] = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + max_chars)
        end = target_end
        if target_end < len(text):
            search_start = max(start + int(max_chars * 0.70), start)
            paragraph_break = text.rfind("\n\n", search_start, target_end)
            sentence_break = text.rfind(". ", search_start, target_end)
            if paragraph_break > start:
                end = paragraph_break + 2
            elif sentence_break > start:
                end = sentence_break + 2

        piece = text[start:end].strip()
        if piece:
            windows.append((start, end, piece))
        if end >= len(text):
            break

        next_start = max(0, end - overlap)
        if next_start <= start:
            next_start = end
        start = next_start

    return windows


def section_label(section: dict[str, Any]) -> str:
    header = str(section.get("section_header") or "").strip()
    title = str(section.get("section_title") or "").strip()
    number = section.get("section_number")

    parts: list[str] = []
    if header:
        parts.append(header)
    elif number not in (None, ""):
        parts.append(f"Capítulo {number}")
    if title and title not in parts:
        parts.append(title)
    if not parts:
        parts.append(f"Seção {int(section.get('section_index') or 0) + 1}")
    return " — ".join(parts)


def looks_like_chapter(section: dict[str, Any]) -> bool:
    section_type = str(section.get("section_type") or "").casefold()
    chapter_rule = str(section.get("chapter_rule") or "").strip()
    header = str(section.get("section_header") or "").casefold()
    return (
        "chapter" in section_type
        or "capitulo" in section_type
        or "capítulo" in section_type
        or bool(chapter_rule)
        or header.startswith("capitulo")
        or header.startswith("capítulo")
    )


class WorkEnrichmentService:
    """Processa uma obra. Storage/versionamento pertencem ao dataset builder."""

    def __init__(self, config) -> None:
        self.config = config
        self.enrichment = config.enrichment

        llm = ChatOllama(
            model=self.enrichment.model,
            base_url=config.generation.base_url,
            temperature=self.enrichment.temperature,
            top_p=self.enrichment.top_p,
            num_predict=self.enrichment.max_tokens,
            num_ctx=self.enrichment.num_ctx,
            num_batch=self.enrichment.num_batch,
            keep_alive=self.enrichment.keep_alive,
            reasoning=self.enrichment.reasoning,
            validate_model_on_init=True,
            client_kwargs={"timeout": self.enrichment.timeout_seconds},
        )

        self.unit_llm = llm.with_structured_output(
            UnitSummary,
            method="json_schema",
        ).with_retry(stop_after_attempt=2)

        self.final_llm = llm.with_structured_output(
            FinalNarrativeSummary,
            method="json_schema",
        ).with_retry(stop_after_attempt=2)

        self._map_started = 0.0
        self._map_elapsed = 0.0
        self._reduce_calls = 0
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(EnrichmentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("summarize_unit", self._summarize_unit_node)
        graph.add_node("assemble", self._assemble_node)
        graph.add_node("final_summary", self._final_summary_node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._route_units, ["summarize_unit"])
        graph.add_edge("summarize_unit", "assemble")
        graph.add_edge("assemble", "final_summary")
        graph.add_edge("final_summary", END)
        return graph.compile()

    def _plan_units(self, source: WorkSource) -> tuple[list[PlannedUnit], dict[str, Any]]:
        semantic = self.enrichment.semantic
        sections, effective_strategy = build_sections(
            text=source.text,
            title=source.title,
            strategy=self.enrichment.strategy,
            min_capitulos=semantic.min_capitulos,
            min_mediana_chars=semantic.min_mediana_chars,
        )

        chapter_sections = [section for section in sections if looks_like_chapter(section)]
        use_chapters = len(chapter_sections) >= int(semantic.min_capitulos)

        if use_chapters:
            source_sections = chapter_sections
            plan_mode = "chapter_semantic"
        else:
            source_sections = [{
                "section_index": 0,
                "section_type": "document",
                "section_header": source.title,
                "section_title": "",
                "start_char": 0,
                "text": source.text,
            }]
            plan_mode = "character_windows"

        units: list[PlannedUnit] = []
        for section_position, section in enumerate(source_sections):
            section_text = str(section.get("text") or "")
            section_start = int(section.get("start_char") or 0)
            windows = split_char_windows(
                section_text,
                max_chars=self.enrichment.unit_chars,
                overlap=self.enrichment.char_overlap,
            )
            label = section_label(section)

            for part_index, (local_start, local_end, piece) in enumerate(windows, start=1):
                units.append(PlannedUnit(
                    unit_id=f"s{section_position + 1:03d}_p{part_index:02d}",
                    section_index=int(section.get("section_index", section_position) or 0),
                    section_label=label,
                    section_type=str(section.get("section_type") or ("chapter" if use_chapters else "document")),
                    part_index=part_index,
                    part_count=len(windows),
                    start_char=section_start + local_start,
                    end_char=section_start + local_end,
                    text=piece,
                ))

        if not units:
            raise RuntimeError("Planejamento do enrichment não gerou blocos.")

        return units, {
            "requested_strategy": self.enrichment.strategy,
            "effective_strategy_from_chunker": effective_strategy,
            "plan_mode": plan_mode,
            "detected_sections": len(sections),
            "detected_chapter_sections": len(chapter_sections),
            "planned_units": len(units),
            "unit_chars": self.enrichment.unit_chars,
            "char_overlap": self.enrichment.char_overlap,
        }

    def _plan_node(self, state: EnrichmentState) -> dict[str, Any]:
        units, plan = self._plan_units(state["source"])
        self._map_started = perf_counter()
        return {"units": units, "plan": plan}

    def _route_units(self, state: EnrichmentState):
        source = state["source"]
        return [
            Send("summarize_unit", {"source": source, "unit": unit})
            for unit in state["units"]
        ]

    def _summarize_unit_node(self, state: UnitTaskState) -> dict[str, Any]:
        source = state["source"]
        unit = state["unit"]
        started = perf_counter()

        summary = self.unit_llm.invoke([
            ("system", UNIT_SYSTEM_PROMPT),
            ("human", UNIT_PROMPT_TEMPLATE.format(
                title=source.title,
                author=source.author,
                max_events=MAX_EVENTS_PER_BLOCK,
                unit_id=unit.unit_id,
                section_label=unit.section_label,
                part_index=unit.part_index,
                part_count=unit.part_count,
                text=unit.text,
            )),
        ])

        elapsed = perf_counter() - started
        if not isinstance(summary, UnitSummary):
            summary = UnitSummary.model_validate(summary)

        summary = summary.model_copy(update={
            "resumo": summary.resumo.strip(),
            "eventos": [event.strip() for event in summary.eventos if event.strip()][:MAX_EVENTS_PER_BLOCK],
        })

        return {"unit_results": [UnitResult(
            unit_id=unit.unit_id,
            section_index=unit.section_index,
            section_label=unit.section_label,
            section_type=unit.section_type,
            part_index=unit.part_index,
            part_count=unit.part_count,
            start_char=unit.start_char,
            end_char=unit.end_char,
            elapsed_seconds=round(elapsed, 4),
            summary=summary,
        )]}

    def _assemble_node(self, state: EnrichmentState) -> dict[str, Any]:
        self._map_elapsed = perf_counter() - self._map_started
        ordered = sorted(
            state["unit_results"],
            key=lambda item: (item.section_index, item.part_index),
        )
        return {"ordered_results": ordered}

    @staticmethod
    def _serialize_items(items: list[dict[str, Any]]) -> str:
        return "\n".join(
            json.dumps(item, ensure_ascii=False, separators=(",", ":"))
            for item in items
        )

    @staticmethod
    def _pack_items(items: list[dict[str, Any]], max_chars: int) -> list[list[dict[str, Any]]]:
        packs: list[list[dict[str, Any]]] = []
        current: list[dict[str, Any]] = []
        current_chars = 0
        for item in items:
            item_chars = len(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            if current and current_chars + item_chars > max_chars:
                packs.append(current)
                current = []
                current_chars = 0
            current.append(item)
            current_chars += item_chars
        if current:
            packs.append(current)
        return packs

    def _invoke_final(
        self,
        *,
        title: str,
        author: str,
        items: list[dict[str, Any]],
    ) -> FinalNarrativeSummary:
        response = self.final_llm.invoke([
            ("system", FINAL_SYSTEM_PROMPT),
            ("human", FINAL_PROMPT_TEMPLATE.format(
                title=title,
                author=author,
                max_events=MAX_EVENTS_FINAL,
                material=self._serialize_items(items),
            )),
        ])
        self._reduce_calls += 1
        if not isinstance(response, FinalNarrativeSummary):
            response = FinalNarrativeSummary.model_validate(response)
        return response.model_copy(update={
            "resumo_geral": response.resumo_geral.strip(),
            "eventos_principais": [
                event.strip()
                for event in response.eventos_principais
                if event.strip()
            ][:MAX_EVENTS_FINAL],
        })

    def _hierarchical_final(
        self,
        source: WorkSource,
        ordered: list[UnitResult],
    ) -> FinalNarrativeSummary:
        items: list[dict[str, Any]] = [{
            "unidade": item.unit_id,
            "secao": item.section_label,
            "parte": f"{item.part_index}/{item.part_count}",
            "resumo": item.summary.resumo,
            "eventos": item.summary.eventos,
        } for item in ordered]

        budget = int(self.enrichment.final_context_chars)
        while len(self._serialize_items(items)) > budget and len(items) > 1:
            packs = self._pack_items(items, budget)
            if len(packs) == 1:
                break
            reduced_items: list[dict[str, Any]] = []
            for index, pack in enumerate(packs, start=1):
                reduced = self._invoke_final(
                    title=source.title,
                    author=source.author,
                    items=pack,
                )
                reduced_items.append({
                    "unidade": f"reduce_{index:03d}",
                    "secao": f"Consolidação {index}",
                    "resumo": reduced.resumo_geral,
                    "eventos": reduced.eventos_principais,
                })
            if len(reduced_items) >= len(items):
                raise RuntimeError(
                    "Reduce hierárquico não conseguiu reduzir o contexto final."
                )
            items = reduced_items

        return self._invoke_final(
            title=source.title,
            author=source.author,
            items=items,
        )

    def _final_summary_node(self, state: EnrichmentState) -> dict[str, Any]:
        final = self._hierarchical_final(state["source"], state["ordered_results"])
        return {"final_summary": final, "reduce_calls": self._reduce_calls}

    def enrich(self, source: WorkSource) -> EnrichmentResult:
        self._reduce_calls = 0
        started = perf_counter()
        result = self.graph.invoke(
            {"source": source, "unit_results": []},
            config={"max_concurrency": self.enrichment.max_parallel},
        )
        total_seconds = perf_counter() - started
        ordered = result["ordered_results"]
        final = result["final_summary"]

        blocks = [{
            "unit_id": item.unit_id,
            "section_index": item.section_index,
            "section_label": item.section_label,
            "section_type": item.section_type,
            "part_index": item.part_index,
            "part_count": item.part_count,
            "start_char": item.start_char,
            "end_char": item.end_char,
            "resumo": item.summary.resumo,
            "eventos": item.summary.eventos,
        } for item in ordered]

        return EnrichmentResult(
            plan=result["plan"],
            blocks=blocks,
            summary={
                "resumo_geral": final.resumo_geral,
                "eventos_principais": final.eventos_principais,
            },
            timings_seconds={
                "total": round(total_seconds, 4),
                "map_wall": round(self._map_elapsed, 4),
                "units": [{
                    "unit_id": item.unit_id,
                    "section_label": item.section_label,
                    "seconds": item.elapsed_seconds,
                } for item in ordered],
            },
            reduce_calls=result.get("reduce_calls", self._reduce_calls),
        )

    def unload_model(self) -> None:
        endpoint = str(self.config.generation.base_url).rstrip("/") + "/api/generate"
        payload = json.dumps({
            "model": self.enrichment.model,
            "keep_alive": 0,
        }).encode("utf-8")
        req = urllib_request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=20) as response:
                response.read()
        except Exception:
            pass


def structured_output_schema_sha256() -> str:
    payload = {
        "unit": UnitSummary.model_json_schema(),
        "final": FinalNarrativeSummary.model_json_schema(),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def prompt_sha256() -> str:
    payload = "\n---\n".join([
        UNIT_SYSTEM_PROMPT,
        UNIT_PROMPT_TEMPLATE,
        FINAL_SYSTEM_PROMPT,
        FINAL_PROMPT_TEMPLATE,
    ])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
