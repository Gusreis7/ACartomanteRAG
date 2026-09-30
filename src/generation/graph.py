from __future__ import annotations

from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from src.generation.context_builder import build_context
from src.generation.prompt import SYSTEM_PROMPT
from src.generation.provider import GenerationProvider, OllamaGenerationProvider
from src.retrieval.service import RetrievalService


class RAGState(TypedDict, total=False):
    query: str
    messages: Annotated[list[BaseMessage], add_messages]
    retrieval_strategy: str
    retrieval_response: Any
    context: str
    evidences: list[dict[str, Any]]
    answer: str
    timings_ms: dict[str, float]


class MachadoRAG:
    def __init__(
        self,
        config,
        *,
        generation_provider: GenerationProvider | None = None,
        checkpointer=None,
    ) -> None:
        self.config = config
        self.retrieval = RetrievalService(config)
        self.generation_provider = (
            generation_provider
            or OllamaGenerationProvider(config.generation)
        )
        self.checkpointer = checkpointer
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("build_context", self._build_context)
        graph.add_node("generate", self._generate)
        graph.add_edge(START, "retrieve")
        graph.add_edge("retrieve", "build_context")
        graph.add_edge("build_context", "generate")
        graph.add_edge("generate", END)
        return graph.compile(checkpointer=self.checkpointer)

    def ask(self, query: str, *, thread_id: str | None = None) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("Query não pode ser vazia.")

        invoke_config = None
        if self.checkpointer is not None:
            if not thread_id:
                raise ValueError(
                    "thread_id é obrigatório quando há checkpointer."
                )
            invoke_config = {
                "configurable": {"thread_id": thread_id}
            }

        result = self.graph.invoke(
            {"query": query, "timings_ms": {}},
            config=invoke_config,
        )
        response = result["retrieval_response"]

        return {
            "query": query,
            "answer": result["answer"],
            "retrieval_strategy": result["retrieval_strategy"],
            "evidences": result["evidences"],
            "timings_ms": result["timings_ms"],
            "silver_dataset_id": response.silver_dataset_id,
            "index_id": response.index_id,
            "collection": response.collection,
        }

    def _retrieve(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        strategy = self.config.retrieval_strategy
        response = self.retrieval.search(
            state["query"],
            method=strategy,
        )

        timings = dict(state.get("timings_ms", {}))
        timings["retrieval_node_ms"] = (
            perf_counter() - started
        ) * 1000.0

        return {
            "retrieval_strategy": strategy,
            "retrieval_response": response,
            "timings_ms": timings,
        }

    def _build_context(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        context, evidences = build_context(
            state["retrieval_response"].hits,
            max_chars=self.config.generation.max_context_chars,
        )

        timings = dict(state.get("timings_ms", {}))
        timings["context_builder_ms"] = (
            perf_counter() - started
        ) * 1000.0

        return {
            "context": context,
            "evidences": evidences,
            "timings_ms": timings,
        }

    def _generate(self, state: RAGState) -> dict[str, Any]:
        started = perf_counter()
        history = _recent_history(
            state.get("messages", []),
            max_turns=self.config.chat.history_turns,
        )

        prompt_parts = [
            "PERGUNTA DO USUÁRIO:\n" + state["query"],
        ]
        if history:
            prompt_parts.append("HISTÓRICO RECENTE:\n" + history)

        prompt_parts.append(
            "EVIDÊNCIAS RECUPERADAS:\n" + state["context"]
        )
        prompt_parts.append(
            "Responda à pergunta seguindo rigorosamente as regras do prompt de sistema."
        )

        message = self.generation_provider.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", "\n\n".join(prompt_parts)),
            ]
        )
        answer = _message_text(message.content)

        timings = dict(state.get("timings_ms", {}))
        timings["generation_ms"] = (
            perf_counter() - started
        ) * 1000.0
        timings["total_ms"] = sum(
            value
            for key, value in timings.items()
            if key != "total_ms"
        )

        return {
            "answer": answer,
            "messages": [
                HumanMessage(content=state["query"]),
                AIMessage(content=answer),
            ],
            "timings_ms": timings,
        }


def _recent_history(messages: list[BaseMessage], *, max_turns: int) -> str:
    if not messages:
        return ""

    selected = messages[-(max_turns * 2):]
    lines: list[str] = []
    for message in selected:
        if isinstance(message, HumanMessage):
            role = "Usuário"
        elif isinstance(message, AIMessage):
            role = "Assistente"
        else:
            continue
        lines.append(f"{role}: {_message_text(message.content)}")

    return "\n".join(lines)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts).strip()

    return str(content).strip()
