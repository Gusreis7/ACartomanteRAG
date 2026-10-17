from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
)
from langgraph.graph import (
    END,
    START,
    StateGraph,
)
from langgraph.graph.message import (
    add_messages,
)

from src.generation.context_builder import (
    build_context,
    build_context_with_enrichment,
)
from src.generation.input_guard import (
    LiteraryInputGuard,
    REFUSAL,
)
from src.generation.prompt import (
    SYSTEM_PROMPT,
)
from src.generation.personas import (
    normalize_persona,
    persona_instruction,
)
from src.generation.provider import (
    GenerationProvider,
    GenerationServiceError,
    create_generation_provider,
)

logger = logging.getLogger("acartomante.generation.graph")


class RAGState(
    TypedDict,
    total=False,
):
    query: str

    persona: str

    messages: Annotated[
        list[BaseMessage],
        add_messages,
    ]

    retrieval_strategy: str

    retrieval_response: Any

    enrichment_response: Any

    context: str

    context_usage: dict[
        str,
        int,
    ]

    evidences: list[dict[str, Any]]

    answer: str
    generation_usage: dict[str, Any]
    guard_decision: str
    guard_usage: dict[str, Any]

    timings_ms: dict[
        str,
        float,
    ]


class MachadoRAG:
    def __init__(
        self,
        config,
        *,
        generation_provider: GenerationProvider | None = None,
        checkpointer=None,
    ) -> None:
        # Modelos de recuperação só são necessários ao construir o runtime.
        from src.retrieval.enrichment import EnrichmentRetriever
        from src.retrieval.service import RetrievalService

        self.config = config

        self.retrieval = RetrievalService(config)

        self.enrichment_retrieval = None
        if config.enrichment_retrieval.enabled:
            try:
                self.enrichment_retrieval = EnrichmentRetriever(config)
            except Exception as exc:
                if not config.enrichment_retrieval.fail_open:
                    raise
                # A evidência original continua obrigatória. Resumos
                # auxiliares indisponíveis não devem impedir o startup.
                logger.warning(
                    "enrichment_startup_unavailable error_type=%s",
                    type(exc).__name__,
                )

        self.generation_provider = generation_provider or create_generation_provider(
            config.generation
        )
        self.input_guard = (
            LiteraryInputGuard(
                self.generation_provider,
                max_tokens=config.chat.guard_max_tokens,
            )
            if config.chat.input_guard_enabled
            else None
        )

        self.checkpointer = checkpointer

        self.graph = self._build_graph()

    def _build_graph(
        self,
    ):
        graph = StateGraph(RAGState)

        graph.add_node(
            "input_guard",
            self._input_guard,
        )

        graph.add_node(
            "refuse",
            self._refuse,
        )

        graph.add_node(
            "retrieve",
            self._retrieve,
        )

        graph.add_node(
            "build_context",
            self._build_context,
        )

        graph.add_node(
            "generate",
            self._generate,
        )

        graph.add_edge(
            START,
            "input_guard",
        )

        graph.add_conditional_edges(
            "input_guard",
            self._route_input,
            {"retrieve": "retrieve", "refuse": "refuse"},
        )

        graph.add_edge(
            "refuse",
            END,
        )

        graph.add_edge(
            "retrieve",
            "build_context",
        )

        graph.add_edge(
            "build_context",
            "generate",
        )

        graph.add_edge(
            "generate",
            END,
        )

        return graph.compile(checkpointer=(self.checkpointer))

    def ask(
        self,
        query: str,
        *,
        thread_id: str | None = None,
        persona: str = "neutral",
    ) -> dict[str, Any]:
        query = str(query).strip()

        if not query:
            raise ValueError("Query não pode ser vazia.")

        persona = normalize_persona(persona)

        invoke_config = None

        if self.checkpointer is not None:
            if not thread_id:
                raise ValueError("thread_id é obrigatório " "quando há checkpointer.")

            invoke_config = {"configurable": {"thread_id": (thread_id)}}

        result = self.graph.invoke(
            {
                "query": query,
                "persona": persona,
                "timings_ms": {},
                # O checkpoint preserva mensagens, não os resultados da pergunta anterior.
                "retrieval_response": None,
                "enrichment_response": None,
                "retrieval_strategy": "not_run",
                "context": "",
                "context_usage": {},
                "evidences": [],
                "answer": "",
                "generation_usage": {},
                "guard_decision": "pending",
                "guard_usage": {},
            },
            config=(invoke_config),
        )

        response = result.get("retrieval_response")

        return {
            "query": query,
            "generation_usage": result.get("generation_usage", {}),
            "input_guard": {
                "decision": result["guard_decision"],
                "usage": result.get("guard_usage", {}),
            },
            "answer": (result["answer"]),
            "retrieval_strategy": (result["retrieval_strategy"]),
            # Contrato público continua contendo
            # exclusivamente evidências originais.
            "evidences": (result["evidences"]),
            "timings_ms": (result["timings_ms"]),
            "silver_dataset_id": getattr(response, "silver_dataset_id", None),
            "index_id": getattr(response, "index_id", None),
            "collection": getattr(response, "collection", None),
        }

    def _input_guard(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        questions = [
            _message_text(message.content)[:1000]
            for message in state.get("messages", [])
            if isinstance(message, HumanMessage)
        ][-2:]
        decision = (
            self.input_guard.review(state["query"], questions)
            if self.input_guard
            else None
        )
        timings = dict(state.get("timings_ms", {}))
        timings["input_guard_ms"] = (perf_counter() - started) * 1000.0
        return {
            "guard_decision": decision.decision if decision else "disabled",
            "guard_usage": decision.usage if decision else {},
            "timings_ms": timings,
        }

    @staticmethod
    def _route_input(state: RAGState) -> str:
        decision = state.get("guard_decision")
        if decision in {"allow", "disabled"}:
            return "retrieve"
        if decision in {"off_topic", "unsafe"}:
            return "refuse"
        raise GenerationServiceError(
            "Não foi possível validar o assunto da pergunta. Tente novamente."
        )

    @staticmethod
    def _refuse(state: RAGState) -> dict[str, Any]:
        timings = dict(state.get("timings_ms", {}))
        timings["total_ms"] = timings.get("input_guard_ms", 0.0)
        return {
            "answer": REFUSAL,
            "messages": [
                HumanMessage(content=state["query"]),
                AIMessage(content=REFUSAL),
            ],
            "evidences": [],
            "context": "",
            "context_usage": {},
            "retrieval_response": None,
            "enrichment_response": None,
            "retrieval_strategy": "not_run",
            "generation_usage": {},
            "timings_ms": timings,
        }

    def _retrieve(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        node_started = perf_counter()

        strategy = self.config.retrieval_strategy

        enrichment_response = None
        textual_ms = 0.0
        enrichment_ms = 0.0

        if self.enrichment_retrieval is None:
            textual_started = perf_counter()

            response = self.retrieval.search(
                state["query"],
                method=(strategy),
            )

            textual_ms = (perf_counter() - textual_started) * 1000.0

        else:
            # As duas trilhas são executadas concorrente e independentemente.
            # O enrichment usa o alias conhecido/fallback como hint.
            # Após o textual terminar, validamos o alias contra o index_id real.
            alias_hint = self.enrichment_retrieval.runtime_alias_hint(method=(strategy))

            textual_started = perf_counter()

            enrichment_started = perf_counter()

            enrichment_error = None

            with ThreadPoolExecutor(
                max_workers=2,
                thread_name_prefix=("acartomante-retrieval"),
            ) as executor:
                textual_future = executor.submit(
                    self.retrieval.search,
                    state["query"],
                    method=(strategy),
                )

                enrichment_future = executor.submit(
                    self.enrichment_retrieval.search,
                    state["query"],
                    method=(strategy),
                    alias_hint=(alias_hint),
                )

                response = textual_future.result()

                textual_ms = (perf_counter() - textual_started) * 1000.0

                try:
                    enrichment_response = enrichment_future.result()

                except Exception as exc:
                    enrichment_error = exc

                enrichment_ms = (perf_counter() - enrichment_started) * 1000.0

            # Garante o contrato "mesmo embedder do índice textual".
            # Normalmente não há segunda chamada: o alias fica memorizado.
            try:
                resolved_alias = self.enrichment_retrieval.resolve_alias(
                    method=(strategy),
                    textual_index_id=(
                        getattr(
                            response,
                            "index_id",
                            None,
                        )
                    ),
                )

                self.enrichment_retrieval.remember_runtime_alias(
                    method=(strategy),
                    alias=(resolved_alias),
                )

                if enrichment_response is not None and (
                    enrichment_response.embedder_alias != resolved_alias
                ):
                    retry_started = perf_counter()

                    enrichment_response = self.enrichment_retrieval.search(
                        state["query"],
                        method=(strategy),
                        textual_index_id=(
                            getattr(
                                response,
                                "index_id",
                                None,
                            )
                        ),
                    )

                    enrichment_ms += (perf_counter() - retry_started) * 1000.0

                    enrichment_error = None

                elif enrichment_response is None and enrichment_error is not None:
                    raise enrichment_error

            except Exception as exc:
                if not (self.config.enrichment_retrieval.fail_open):
                    raise

                enrichment_response = None

                logger.warning(
                    "enrichment_retrieval_failed " "strategy=%s error=%s",
                    strategy,
                    exc,
                )

        timings = dict(
            state.get(
                "timings_ms",
                {},
            )
        )

        timings["textual_retrieval_ms"] = textual_ms

        timings["enrichment_retrieval_ms"] = enrichment_ms

        timings["retrieval_node_ms"] = (perf_counter() - node_started) * 1000.0

        return {
            "retrieval_strategy": (strategy),
            "retrieval_response": (response),
            "enrichment_response": (enrichment_response),
            "timings_ms": (timings),
        }

    def _build_context(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        started = perf_counter()

        enrichment_response = state.get("enrichment_response")

        if enrichment_response is not None and (
            self.config.enrichment_retrieval.enabled
        ):
            (
                context,
                evidences,
                usage,
            ) = build_context_with_enrichment(
                state["retrieval_response"].hits,
                enrichment_hits=(enrichment_response.hits),
                max_chars=(self.config.generation.max_context_chars),
                enrichment_max_chars=(
                    self.config.enrichment_retrieval.max_context_chars
                ),
                enrichment_max_fraction=(
                    self.config.enrichment_retrieval.max_context_fraction
                ),
            )

        else:
            (
                context,
                evidences,
            ) = build_context(
                state["retrieval_response"].hits,
                max_chars=(self.config.generation.max_context_chars),
            )

            usage = {
                "total_chars": (len(context)),
                "enrichment_chars": 0,
                "enrichment_hits": 0,
                "textual_chars": (len(context)),
                "textual_evidences": (len(evidences)),
            }

        timings = dict(
            state.get(
                "timings_ms",
                {},
            )
        )

        timings["context_builder_ms"] = (perf_counter() - started) * 1000.0

        # Métricas numéricas cabem no contrato atual sem
        # expor o conteúdo derivado como evidência.
        timings["context_total_chars"] = float(usage["total_chars"])

        timings["context_enrichment_chars"] = float(usage["enrichment_chars"])

        timings["context_enrichment_hits"] = float(usage["enrichment_hits"])

        timings["context_textual_chars"] = float(usage["textual_chars"])

        return {
            "context": context,
            "context_usage": (usage),
            "evidences": (evidences),
            "timings_ms": (timings),
        }

    def _generate(
        self,
        state: RAGState,
    ) -> dict[str, Any]:
        started = perf_counter()

        history = _recent_history(
            state.get(
                "messages",
                [],
            ),
            max_turns=(self.config.chat.history_turns),
        )

        prompt_parts = [
            ("PERGUNTA DO USUÁRIO:\n" + state["query"]),
        ]

        if history:
            prompt_parts.append("HISTÓRICO RECENTE:\n" + history)

        prompt_parts.append("CONTEXTO RECUPERADO:\n" + state["context"])

        prompt_parts.append(
            "Responda à pergunta seguindo "
            "rigorosamente as regras do "
            "prompt de sistema."
        )

        system_prompt = (
            SYSTEM_PROMPT
            + "\
\
"
            + persona_instruction(
                state.get(
                    "persona",
                    "neutral",
                )
            )
        )

        message = self.generation_provider.invoke(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    "\n\n".join(prompt_parts),
                ),
            ]
        )

        answer = _message_text(message.content)

        if not answer:
            raise GenerationServiceError("Modelo de geração retornou resposta vazia.")

        timings = dict(
            state.get(
                "timings_ms",
                {},
            )
        )

        timings["generation_ms"] = (perf_counter() - started) * 1000.0

        timings["total_ms"] = sum(
            value
            for key, value in timings.items()
            if (key != "total_ms" and key.endswith("_ms"))
        )

        return {
            "generation_usage": {
                "provider": self.config.generation.provider,
                "model": self.config.generation.model,
                **(getattr(message, "usage_metadata", None) or {}),
                **{
                    key: value
                    for key, value in (
                        getattr(message, "response_metadata", {}) or {}
                    ).items()
                    if key in {"estimated_cost_usd", "region", "retry_attempts"}
                },
            },
            "answer": (answer),
            "messages": [
                HumanMessage(content=(state["query"])),
                AIMessage(content=(answer)),
            ],
            "timings_ms": (timings),
        }


def _recent_history(
    messages: list[BaseMessage],
    *,
    max_turns: int,
) -> str:
    if not messages:
        return ""

    selected = messages[-(max_turns * 2) :]

    lines: list[str] = []

    for message in selected:
        if isinstance(
            message,
            HumanMessage,
        ):
            role = "Usuário"

        elif isinstance(
            message,
            AIMessage,
        ):
            role = "Assistente"

        else:
            continue

        lines.append(f"{role}: " f"{_message_text(message.content)}")

    return "\n".join(lines)


def _message_text(
    content: Any,
) -> str:
    if isinstance(
        content,
        str,
    ):
        return content.strip()

    if isinstance(
        content,
        list,
    ):
        parts = []

        for item in content:
            if isinstance(
                item,
                str,
            ):
                parts.append(item)

            elif isinstance(
                item,
                dict,
            ):
                text = item.get("text")

                if text:
                    parts.append(str(text))

        return "\n".join(parts).strip()

    return str(content).strip()
