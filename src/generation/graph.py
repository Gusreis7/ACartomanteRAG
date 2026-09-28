from __future__ import annotations

from time import perf_counter
from typing import Any, TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph

from src.generation.context_builder import build_context
from src.generation.prompt import SYSTEM_PROMPT
from src.retrieval.service import RetrievalService


class RAGState(TypedDict, total=False):
    query: str
    retrieval_strategy: str
    retrieval_response: Any
    context: str
    evidences: list[dict[str, Any]]
    answer: str
    timings_ms: dict[str, float]


class MachadoRAG:
    """START -> retrieve -> build_context -> generate -> END."""

    def __init__(self, config) -> None:
        self.config = config
        self.retrieval = RetrievalService(config)
        generation = config.generation

        if generation.provider != "ollama":
            raise ValueError(
                "Generation v1 suporta somente provider=ollama."
            )

        self.llm = ChatOllama(
            model=generation.model,
            base_url=generation.base_url,
            temperature=generation.temperature,
            top_p=generation.top_p,
            num_predict=generation.max_tokens,
            num_ctx=generation.num_ctx,
            keep_alive=generation.keep_alive,
            validate_model_on_init=True,
            client_kwargs={"timeout": generation.timeout_seconds},
        )
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
        return graph.compile()

    def ask(self, query: str) -> dict[str, Any]:
        query = str(query).strip()
        if not query:
            raise ValueError("Query não pode ser vazia.")

        result = self.graph.invoke(
            {
                "query": query,
                "timings_ms": {},
            }
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
        response = state["retrieval_response"]
        context, evidences = build_context(
            response.hits,
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
        user_prompt = (
            "PERGUNTA DO USUÁRIO:\n"
            f"{state['query']}\n\n"
            "EVIDÊNCIAS RECUPERADAS:\n"
            f"{state['context']}\n\n"
            "Responda à pergunta seguindo rigorosamente as regras do "
            "prompt de sistema."
        )
        message = self.llm.invoke(
            [
                ("system", SYSTEM_PROMPT),
                ("human", user_prompt),
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
            "timings_ms": timings,
        }


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
        return "\n".join(parts).strip()

    return str(content).strip()
