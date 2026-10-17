from __future__ import annotations

import argparse
import hashlib
import io
import json
import logging
import operator
import re
import threading
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Annotated, Any, TypedDict
from urllib import request as urllib_request

import pandas as pd
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from pydantic import BaseModel, ConfigDict, Field
from tqdm.auto import tqdm

from src.config.config_manager import ConfigManager
from src.storage.rustfs_manager import RustFSManager
from src.utils.chunking import build_sections

logger = logging.getLogger("acartomante.enrichment")

BRONZE_CURRENT_KEY = "current.json"
ENRICHMENT_PREFIX = "enrichments/work_summaries"

DEFAULT_MODEL = "qwen3.5:4b"
DEFAULT_TITLE = "O Alienista"
DEFAULT_MAX_TOKENS = 1024
DEFAULT_CHAR_OVERLAP = 800
DEFAULT_NUM_BATCH = 256


# ---------------------------------------------------------------------------
# Structured output: mínimo para este smoke
# ---------------------------------------------------------------------------


class UnitSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resumo: str = Field(
        default="",
        description="Resumo factual curto da unidade/capítulo.",
    )
    eventos: list[str] = Field(
        default_factory=list,
        description=(
            "Eventos mais relevantes da unidade; cada item deve ser "
            "uma frase factual curta."
        ),
    )


class FinalNarrativeSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    resumo_geral: str = Field(
        default="",
        description="Resumo factual e cronológico da obra inteira.",
    )
    eventos_principais: list[str] = Field(
        default_factory=list,
        description=("Principais eventos da obra em ordem narrativa aproximada."),
    )


# ---------------------------------------------------------------------------
# Runtime data
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class WorkSource:
    title: str
    author: str
    category: str
    source_document_id: str
    bronze_artifact_id: str
    bronze_manifest_key: str
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
    part_index: int
    part_count: int
    elapsed_seconds: float
    summary: UnitSummary


class EnrichmentState(TypedDict, total=False):
    title: str
    output_root: Path
    upload: bool
    source: WorkSource
    units: list[PlannedUnit]
    plan: dict[str, Any]
    unit_results: Annotated[list[UnitResult], operator.add]
    ordered_results: list[UnitResult]
    final_summary: FinalNarrativeSummary
    run_id: str
    metrics: dict[str, Any]
    local_paths: dict[str, str]
    uploaded_paths: dict[str, str]


class UnitTaskState(TypedDict):
    title: str
    author: str
    unit: PlannedUnit


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_text(value: str) -> str:
    value = unicodedata.normalize("NFKD", str(value))
    value = "".join(char for char in value if not unicodedata.combining(char))
    return re.sub(r"\s+", " ", value.casefold()).strip()


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", normalize_text(value))
    return value.strip("-") or "obra"


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


# ---------------------------------------------------------------------------
# LangGraph agent
# ---------------------------------------------------------------------------


class WorkEnrichmentAgent:
    """
    START -> load_source -> plan_hybrid -> summarize_unit x N
          -> assemble -> final_summary -> persist -> END

    MAP: resumo + eventos.
    FINAL: resumo_geral + eventos_principais.
    """

    def __init__(
        self,
        *,
        config,
        model: str = DEFAULT_MODEL,
        workers: int | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        unit_chars: int | None = None,
        char_overlap: int = DEFAULT_CHAR_OVERLAP,
        num_batch: int = DEFAULT_NUM_BATCH,
    ) -> None:
        self.config = config
        self.model = model
        self.storage = RustFSManager.from_config(config.environment.rustfs)
        self.bronze_bucket = config.storage.bronze_bucket
        self.silver_bucket = config.storage.silver_bucket

        self.workers = max(
            1,
            int(workers if workers is not None else config.generation.max_parallel),
        )
        self.max_tokens = int(max_tokens)
        self.unit_chars = int(
            unit_chars
            if unit_chars is not None
            else config.generation.max_context_chars
        )
        self.char_overlap = int(char_overlap)
        self.num_batch = int(num_batch)

        llm = ChatOllama(
            model=self.model,
            base_url=config.generation.base_url,
            temperature=0.0,
            top_p=config.generation.top_p,
            num_predict=self.max_tokens,
            num_ctx=config.generation.num_ctx,
            num_batch=self.num_batch,
            keep_alive=config.generation.keep_alive,
            reasoning=False,
            validate_model_on_init=True,
            client_kwargs={"timeout": config.generation.timeout_seconds},
        )

        self.unit_llm = llm.with_structured_output(
            UnitSummary, method="json_schema"
        ).with_retry(stop_after_attempt=2)
        self.final_llm = llm.with_structured_output(
            FinalNarrativeSummary,
            method="json_schema",
        ).with_retry(stop_after_attempt=2)

        self._progress_lock = threading.Lock()
        self._map_progress = None
        self._map_done = 0
        self._timings: dict[str, float] = {}
        self._run_started = 0.0
        self._map_started = 0.0
        self._final_started = 0.0

        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(EnrichmentState)

        graph.add_node("load_source", self._load_source_node)
        graph.add_node("plan_hybrid", self._plan_hybrid_node)
        graph.add_node("summarize_unit", self._summarize_unit_node)
        graph.add_node("assemble", self._assemble_node)
        graph.add_node("final_summary", self._final_summary_node)
        graph.add_node("persist", self._persist_node)

        graph.add_edge(START, "load_source")
        graph.add_edge("load_source", "plan_hybrid")
        graph.add_conditional_edges(
            "plan_hybrid",
            self._route_units,
            ["summarize_unit"],
        )
        graph.add_edge("summarize_unit", "assemble")
        graph.add_edge("assemble", "final_summary")
        graph.add_edge("final_summary", "persist")
        graph.add_edge("persist", END)

        return graph.compile()

    # ------------------------------------------------------------------
    # Source
    # ------------------------------------------------------------------

    def _resolve_work(self, title: str) -> WorkSource:
        current = json.loads(
            self.storage.download_bytes(
                bucket=self.bronze_bucket,
                key=BRONZE_CURRENT_KEY,
            ).decode("utf-8")
        )

        bronze_artifact_id = str(current.get("bronze_artifact_id") or "").strip()
        bronze_manifest_key = str(current.get("manifest_key") or "").strip()

        if not bronze_artifact_id or not bronze_manifest_key:
            raise RuntimeError(
                "current.json da Bronze inválido: "
                "bronze_artifact_id/manifest_key ausentes."
            )

        manifest = pd.read_parquet(
            io.BytesIO(
                self.storage.download_bytes(
                    bucket=self.bronze_bucket,
                    key=bronze_manifest_key,
                )
            )
        )

        if "status_bronze" in manifest.columns:
            manifest = manifest[
                manifest["status_bronze"].astype(str).eq("success")
            ].copy()

        wanted = normalize_text(title)
        manifest["_title_norm"] = (
            manifest["titulo"].fillna("").astype(str).map(normalize_text)
        )

        matches = manifest[manifest["_title_norm"].eq(wanted)].copy()

        if matches.empty:
            contains = manifest[
                manifest["_title_norm"].str.contains(
                    re.escape(wanted),
                    regex=True,
                )
            ].copy()
            if len(contains) == 1:
                matches = contains

        if len(matches) != 1:
            nearby = (
                manifest.loc[
                    manifest["_title_norm"].str.contains(
                        "alienista",
                        regex=False,
                    ),
                    "titulo",
                ]
                .dropna()
                .astype(str)
                .tolist()
            )
            raise RuntimeError(
                f"Obra '{title}' não resolvida de forma única. "
                f"Candidatas próximas: {nearby}"
            )

        row = matches.iloc[0]
        bronze_object_key = str(row.get("bronze_object_key") or "").strip()

        if not bronze_object_key:
            raise RuntimeError("Registro Bronze sem bronze_object_key.")

        text = self.storage.download_bytes(
            bucket=self.bronze_bucket,
            key=bronze_object_key,
        ).decode("utf-8", errors="replace")

        if not text.strip():
            raise RuntimeError("Texto Bronze vazio.")

        return WorkSource(
            title=str(row.get("titulo") or title),
            author=str(row.get("autor") or "Machado de Assis"),
            category=str(row.get("categoria") or ""),
            source_document_id=str(row.get("source_document_id") or ""),
            bronze_artifact_id=bronze_artifact_id,
            bronze_manifest_key=bronze_manifest_key,
            bronze_object_key=bronze_object_key,
            text=text,
        )

    def _load_source_node(
        self,
        state: EnrichmentState,
    ) -> dict[str, Any]:
        started = perf_counter()
        source = self._resolve_work(state["title"])
        self._timings["load_source"] = perf_counter() - started

        print(
            f"[source] {source.title} | "
            f"{len(source.text):,} chars | "
            f"{len(source.text.split()):,} words"
        )

        return {"source": source}

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def _plan_units(
        self,
        source: WorkSource,
    ) -> tuple[list[PlannedUnit], dict[str, Any]]:
        semantic = self.config.silver.semantic

        sections, effective_strategy = build_sections(
            text=source.text,
            title=source.title,
            strategy="hybrid",
            min_capitulos=semantic.min_capitulos,
            min_mediana_chars=semantic.min_mediana_chars,
        )

        chapter_sections = [
            section for section in sections if looks_like_chapter(section)
        ]
        use_chapters = len(chapter_sections) >= int(semantic.min_capitulos)

        if use_chapters:
            source_sections = chapter_sections
            plan_mode = "chapter_semantic"
        else:
            source_sections = [
                {
                    "section_index": 0,
                    "section_type": "document",
                    "section_header": source.title,
                    "section_title": "",
                    "start_char": 0,
                    "text": source.text,
                }
            ]
            plan_mode = "character_windows"

        units: list[PlannedUnit] = []

        for section_position, section in enumerate(source_sections):
            section_text = str(section.get("text") or "")
            section_start = int(section.get("start_char") or 0)
            windows = split_char_windows(
                section_text,
                max_chars=self.unit_chars,
                overlap=self.char_overlap,
            )
            label = section_label(section)

            for part_index, (
                local_start,
                local_end,
                piece,
            ) in enumerate(windows, start=1):
                units.append(
                    PlannedUnit(
                        unit_id=(f"s{section_position + 1:03d}_p{part_index:02d}"),
                        section_index=int(
                            section.get("section_index", section_position) or 0
                        ),
                        section_label=label,
                        section_type=str(
                            section.get("section_type")
                            or ("chapter" if use_chapters else "document")
                        ),
                        part_index=part_index,
                        part_count=len(windows),
                        start_char=section_start + local_start,
                        end_char=section_start + local_end,
                        text=piece,
                    )
                )

        if not units:
            raise RuntimeError("Planejamento não gerou blocos.")

        return units, {
            "requested_strategy": "hybrid",
            "effective_strategy_from_chunker": effective_strategy,
            "plan_mode": plan_mode,
            "detected_sections": len(sections),
            "detected_chapter_sections": len(chapter_sections),
            "planned_units": len(units),
            "unit_chars": self.unit_chars,
            "char_overlap": self.char_overlap,
        }

    def _plan_hybrid_node(
        self,
        state: EnrichmentState,
    ) -> dict[str, Any]:
        started = perf_counter()
        units, plan = self._plan_units(state["source"])
        self._timings["plan_hybrid"] = perf_counter() - started

        print(
            "[plan] "
            f"{plan['plan_mode']} | "
            f"chunker={plan['effective_strategy_from_chunker']} | "
            f"sections={plan['detected_sections']} | "
            f"chapters={plan['detected_chapter_sections']} | "
            f"blocks={plan['planned_units']} | "
            f"workers={self.workers}"
        )

        self._map_done = 0
        self._map_started = perf_counter()
        self._map_progress = tqdm(
            total=len(units),
            desc="Resumo · blocos",
            unit="bloco",
            dynamic_ncols=True,
        )
        self._map_progress.set_postfix(faltam=len(units))

        return {
            "units": units,
            "plan": plan,
        }

    def _route_units(self, state: EnrichmentState):
        source = state["source"]

        return [
            Send(
                "summarize_unit",
                {
                    "title": source.title,
                    "author": source.author,
                    "unit": unit,
                },
            )
            for unit in state["units"]
        ]

    # ------------------------------------------------------------------
    # MAP
    # ------------------------------------------------------------------

    def _unit_prompt(self, state: UnitTaskState) -> str:
        unit = state["unit"]

        return f"""\
Analise SOMENTE este trecho de "{state['title']}", de {state['author']}.

Extraia apenas:

1. RESUMO
Um resumo factual curto desta unidade.

2. EVENTOS
No máximo 5 acontecimentos importantes da unidade, em frases curtas.
Quando uma consequência estiver explícita, inclua-a na mesma frase.

Não extraia personagens.
Não extraia temas.
Não faça interpretação.
Não faça pontos importantes separados.
Não use conhecimento externo.
Não repita o mesmo fato.

UNIDADE: {unit.unit_id}
SEÇÃO: {unit.section_label}
PARTE: {unit.part_index}/{unit.part_count}

TRECHO:
<<<
{unit.text}
>>>
"""

    def _summarize_unit_node(
        self,
        state: UnitTaskState,
    ) -> dict[str, Any]:
        unit = state["unit"]
        started = perf_counter()

        try:
            summary = self.unit_llm.invoke(
                [
                    (
                        "system",
                        "Resuma fatos narrativos com concisão e fidelidade. "
                        "Use somente o trecho e siga o schema.",
                    ),
                    ("human", self._unit_prompt(state)),
                ]
            )
        finally:
            elapsed = perf_counter() - started

        if not isinstance(summary, UnitSummary):
            summary = UnitSummary.model_validate(summary)

        summary = summary.model_copy(
            update={
                "resumo": summary.resumo.strip(),
                "eventos": [
                    event.strip() for event in summary.eventos if event.strip()
                ][:5],
            }
        )

        result = UnitResult(
            unit_id=unit.unit_id,
            section_index=unit.section_index,
            section_label=unit.section_label,
            part_index=unit.part_index,
            part_count=unit.part_count,
            elapsed_seconds=round(elapsed, 4),
            summary=summary,
        )

        with self._progress_lock:
            self._map_done += 1

            if self._map_progress is not None:
                self._map_progress.update(1)
                self._map_progress.set_postfix(
                    faltam=max(0, self._map_progress.total - self._map_done),
                    sec=f"{elapsed:.1f}",
                    refresh=True,
                )

        return {"unit_results": [result]}

    # ------------------------------------------------------------------
    # Fan-in
    # ------------------------------------------------------------------

    def _assemble_node(
        self,
        state: EnrichmentState,
    ) -> dict[str, Any]:
        if self._map_progress is not None:
            self._map_progress.close()
            self._map_progress = None

        self._timings["map_wall"] = perf_counter() - self._map_started

        ordered = sorted(
            state["unit_results"],
            key=lambda item: (
                item.section_index,
                item.part_index,
            ),
        )

        print(
            f"[map] {len(ordered)} blocos concluídos em "
            f"{self._timings['map_wall']:.2f}s"
        )

        return {"ordered_results": ordered}

    # ------------------------------------------------------------------
    # Final synthesis
    # ------------------------------------------------------------------

    def _final_prompt(self, state: EnrichmentState) -> str:
        blocks = []

        for item in state["ordered_results"]:
            payload = {
                "unidade": item.unit_id,
                "secao": item.section_label,
                "parte": f"{item.part_index}/{item.part_count}",
                "resumo": item.summary.resumo,
                "eventos": item.summary.eventos,
            }
            blocks.append(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            )

        material = "\n".join(blocks)

        return f"""\
Consolide as anotações ordenadas de "{state['source'].title}",
de {state['source'].author}.

Produza somente:

1. RESUMO GERAL
Um resumo factual e cronológico da obra.

2. EVENTOS PRINCIPAIS
No máximo 15 eventos importantes da obra, em ordem narrativa aproximada.

Não extraia personagens.
Não produza temas.
Não faça interpretação.
Não faça análise crítica.
Não crie pontos importantes separados.
Não use conhecimento externo.
Não complete lacunas de memória.
Evite repetir o mesmo evento.

ANOTAÇÕES POR CAPÍTULO/BLOCO:
{material}
"""

    def _final_summary_node(
        self,
        state: EnrichmentState,
    ) -> dict[str, Any]:
        self._final_started = perf_counter()

        with tqdm(
            total=1,
            desc="Síntese final",
            unit="obra",
            dynamic_ncols=True,
        ) as progress:
            final_summary = self.final_llm.invoke(
                [
                    (
                        "system",
                        "Faça apenas uma consolidação narrativa factual e "
                        "compacta. Use somente as anotações recebidas.",
                    ),
                    ("human", self._final_prompt(state)),
                ]
            )
            progress.update(1)

        if not isinstance(final_summary, FinalNarrativeSummary):
            final_summary = FinalNarrativeSummary.model_validate(final_summary)

        final_summary = final_summary.model_copy(
            update={
                "resumo_geral": final_summary.resumo_geral.strip(),
                "eventos_principais": [
                    event.strip()
                    for event in final_summary.eventos_principais
                    if event.strip()
                ][:15],
            }
        )

        self._timings["final_summary"] = perf_counter() - self._final_started

        return {"final_summary": final_summary}

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _render_markdown(
        self,
        source: WorkSource,
        state: EnrichmentState,
    ) -> str:
        final = state["final_summary"]

        lines = [
            f"# {source.title}",
            "",
            f"**Autor:** {source.author}",
            f"**Modelo:** `{self.model}`",
            "",
            "## Resumo por capítulo/bloco",
            "",
        ]

        for item in state["ordered_results"]:
            heading = f"### {item.section_label}"

            if item.part_count > 1:
                heading += f" · parte {item.part_index}/{item.part_count}"

            lines.extend(
                [
                    heading,
                    "",
                    item.summary.resumo,
                    "",
                ]
            )

            for event in item.summary.eventos:
                lines.append(f"- {event}")

            lines.append("")

        lines.extend(
            [
                "## Resumo geral",
                "",
                final.resumo_geral,
                "",
                "## Eventos principais",
                "",
            ]
        )

        for event in final.eventos_principais:
            lines.append(f"- {event}")

        lines.append("")
        return "\n".join(lines)

    def _persist_node(
        self,
        state: EnrichmentState,
    ) -> dict[str, Any]:
        started = perf_counter()
        source = state["source"]

        run_id = f"summary_{slugify(source.title)}_" + datetime.now(
            timezone.utc
        ).strftime("%Y%m%dT%H%M%SZ")

        output_dir = state["output_root"] / run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        markdown = self._render_markdown(source, state)
        source_sha256 = hashlib.sha256(source.text.encode("utf-8")).hexdigest()

        summary_payload = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": self.model,
            "source": {
                "title": source.title,
                "author": source.author,
                "category": source.category,
                "source_document_id": source.source_document_id,
                "bronze_artifact_id": source.bronze_artifact_id,
                "bronze_manifest_key": source.bronze_manifest_key,
                "bronze_object_key": source.bronze_object_key,
                "text_sha256": source_sha256,
            },
            "plan": state["plan"],
            "blocks": [
                {
                    "unit_id": item.unit_id,
                    "section_index": item.section_index,
                    "section_label": item.section_label,
                    "part_index": item.part_index,
                    "part_count": item.part_count,
                    "resumo": item.summary.resumo,
                    "eventos": item.summary.eventos,
                }
                for item in state["ordered_results"]
            ],
            "summary": {
                "resumo_geral": state["final_summary"].resumo_geral,
                "eventos_principais": (state["final_summary"].eventos_principais),
            },
        }

        summary_json = json.dumps(
            summary_payload,
            ensure_ascii=False,
            indent=2,
        )

        total_seconds = perf_counter() - self._run_started

        metrics = {
            "run_id": run_id,
            "created_at_utc": utc_now(),
            "model": self.model,
            "workers": self.workers,
            "num_batch": self.num_batch,
            "max_tokens": self.max_tokens,
            "structured_output": {
                "provider": "ChatOllama",
                "method": "json_schema",
                "schemas": [
                    "UnitSummary",
                    "FinalNarrativeSummary",
                ],
                "manual_parser": False,
                "characters_enabled": False,
            },
            "source": {
                "chars": len(source.text),
                "words": len(source.text.split()),
                "bytes_utf8": len(source.text.encode("utf-8")),
            },
            "plan": state["plan"],
            "timings_seconds": {
                **{key: round(value, 4) for key, value in self._timings.items()},
                "total": round(total_seconds, 4),
            },
            "unit_timings": [
                {
                    "unit_id": item.unit_id,
                    "section_label": item.section_label,
                    "seconds": item.elapsed_seconds,
                }
                for item in state["ordered_results"]
            ],
            "output": {
                "summary_chars": len(markdown),
                "summary_words": len(markdown.split()),
                "summary_bytes_utf8": len(markdown.encode("utf-8")),
                "compression_ratio_chars": (len(markdown) / max(1, len(source.text))),
            },
        }

        metrics_json = json.dumps(
            metrics,
            ensure_ascii=False,
            indent=2,
        )

        summary_md_path = output_dir / "summary.md"
        summary_json_path = output_dir / "summary.json"
        metrics_path = output_dir / "metrics.json"

        summary_md_path.write_text(markdown, encoding="utf-8")
        summary_json_path.write_text(summary_json, encoding="utf-8")
        metrics_path.write_text(metrics_json, encoding="utf-8")

        local_paths = {
            "summary": str(summary_md_path),
            "json": str(summary_json_path),
            "metrics": str(metrics_path),
        }
        uploaded_paths: dict[str, str] = {}

        if state["upload"]:
            prefix = (
                f"{ENRICHMENT_PREFIX}/"
                f"{source.bronze_artifact_id}/"
                f"{slugify(source.title)}/"
                f"{run_id}"
            )

            objects = {
                "summary": f"{prefix}/summary.md",
                "json": f"{prefix}/summary.json",
                "metrics": f"{prefix}/metrics.json",
            }

            self.storage.upload_bytes(
                bucket=self.silver_bucket,
                key=objects["summary"],
                data=markdown.encode("utf-8"),
                content_type="text/markdown; charset=utf-8",
            )
            self.storage.upload_bytes(
                bucket=self.silver_bucket,
                key=objects["json"],
                data=summary_json.encode("utf-8"),
                content_type="application/json",
            )
            self.storage.upload_bytes(
                bucket=self.silver_bucket,
                key=objects["metrics"],
                data=metrics_json.encode("utf-8"),
                content_type="application/json",
            )

            uploaded_paths = {
                name: f"s3://{self.silver_bucket}/{key}"
                for name, key in objects.items()
            }

        self._timings["persist"] = perf_counter() - started

        print()
        print("=== ENRICHMENT SMOKE OK ===")
        print(f"run_id: {run_id}")
        print(f"title: {source.title}")
        print(f"model: {self.model}")
        print(
            "strategy: "
            f"{state['plan']['plan_mode']} "
            "(chunker="
            f"{state['plan']['effective_strategy_from_chunker']}"
            ")"
        )
        print(f"blocks: {state['plan']['planned_units']}")
        print(f"source_chars: {metrics['source']['chars']}")
        print(f"summary_chars: {metrics['output']['summary_chars']}")
        print(f"summary_words: {metrics['output']['summary_words']}")
        print(
            "compression_ratio_chars: "
            f"{metrics['output']['compression_ratio_chars']:.4f}"
        )
        print("elapsed_total_s: " f"{metrics['timings_seconds']['total']:.2f}")
        print("elapsed_map_s: " f"{metrics['timings_seconds'].get('map_wall', 0):.2f}")
        print(
            "elapsed_final_s: "
            f"{metrics['timings_seconds'].get('final_summary', 0):.2f}"
        )
        print(f"summary: {summary_md_path}")
        print(f"json: {summary_json_path}")
        print(f"metrics: {metrics_path}")

        for name, uri in uploaded_paths.items():
            print(f"{name}_s3: {uri}")

        return {
            "run_id": run_id,
            "metrics": metrics,
            "local_paths": local_paths,
            "uploaded_paths": uploaded_paths,
        }

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        title: str,
        output_root: Path,
        upload: bool,
    ) -> EnrichmentState:
        self._timings = {}
        self._run_started = perf_counter()

        return self.graph.invoke(
            {
                "title": title,
                "output_root": output_root,
                "upload": upload,
                "unit_results": [],
            },
            config={
                "max_concurrency": self.workers,
            },
        )

    def unload_model(self) -> None:
        endpoint = str(self.config.generation.base_url).rstrip("/") + "/api/generate"

        payload = json.dumps(
            {
                "model": self.model,
                "keep_alive": 0,
            }
        ).encode("utf-8")

        req = urllib_request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib_request.urlopen(req, timeout=20) as response:
                response.read()
        except Exception as exc:
            logger.warning(
                "model_unload_failed model=%s error=%s",
                self.model,
                exc,
            )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=("Smoke de enriquecimento mínimo: resumo + eventos.")
    )

    parser.add_argument("--title", default=DEFAULT_TITLE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Padrão: generation.max_parallel.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
    )
    parser.add_argument(
        "--unit-chars",
        type=int,
        default=None,
        help=("Fallback por caracteres. " "Padrão: generation.max_context_chars."),
    )
    parser.add_argument(
        "--char-overlap",
        type=int,
        default=DEFAULT_CHAR_OVERLAP,
    )
    parser.add_argument(
        "--num-batch",
        type=int,
        default=DEFAULT_NUM_BATCH,
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("reports/enrichment"),
    )
    parser.add_argument("--upload", action="store_true")
    parser.add_argument("--keep-model", action="store_true")
    parser.add_argument("--env-file", type=Path, default=None)
    parser.add_argument("--config-file", type=Path, default=None)

    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format=("%(asctime)s %(levelname)s " "%(name)s %(message)s"),
    )

    args = parse_args()
    config_kwargs: dict[str, Any] = {}

    if args.env_file is not None:
        config_kwargs["env_file"] = args.env_file
    if args.config_file is not None:
        config_kwargs["yaml_file"] = args.config_file

    config = ConfigManager.load(**config_kwargs)

    agent = WorkEnrichmentAgent(
        config=config,
        model=args.model,
        workers=args.workers,
        max_tokens=args.max_tokens,
        unit_chars=args.unit_chars,
        char_overlap=args.char_overlap,
        num_batch=args.num_batch,
    )

    try:
        agent.run(
            title=args.title,
            output_root=args.output_root,
            upload=args.upload,
        )
    finally:
        if not args.keep_model:
            agent.unload_model()


if __name__ == "__main__":
    main()
