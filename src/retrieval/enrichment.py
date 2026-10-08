from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.embeddings.registry import EmbedderRegistry
from src.storage.qdrant_manager import QdrantManager
from src.storage.rustfs_manager import RustFSManager


logger = logging.getLogger("acartomante.retrieval.enrichment")

ENRICHMENT_INDEX_CURRENT_KEY = "enrichment/current.json"


@dataclass(frozen=True)
class EnrichmentRetrievalResponse:
    hits: list[dict[str, Any]]
    embedder_alias: str | None
    index_id: str | None
    collection: str | None


def _make_qdrant_manager(config) -> QdrantManager:
    qcfg = config.environment.qdrant

    factory = getattr(
        QdrantManager,
        "from_config",
        None,
    )

    if callable(factory):
        return factory(qcfg)

    try:
        return QdrantManager(
            url=qcfg.url,
            api_key=qcfg.api_key,
        )

    except TypeError:
        return QdrantManager(
            qcfg.url,
            qcfg.api_key,
        )


def _safe_float(
    value: Any,
) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


class EnrichmentRetriever:
    """
    Retrieval auxiliar dos resumos derivados.

    Não participa do ranking textual e não usa cross-encoder.

    A collection é escolhida pelo mesmo embedder do índice textual ativo,
    quando o manifest do índice textual permite resolver esse alias.
    BM25 e qualquer caso sem alias resolvível usam o fallback configurado.
    """

    def __init__(
        self,
        config,
    ) -> None:
        self.config = config

        self.rustfs = (
            RustFSManager.from_config(
                config.environment.rustfs
            )
        )

        self.qdrant = (
            _make_qdrant_manager(
                config
            )
        )

        self.client = getattr(
            self.qdrant,
            "client",
            self.qdrant,
        )

        self.registry = (
            EmbedderRegistry()
        )

        self.indexes_bucket = (
            config.storage
            .embedding_indexes_bucket
        )

        self.runtime_config = (
            config.enrichment_retrieval
        )

        self._index_set = (
            self._load_index_set()
        )

        self._embedders: dict[
            str,
            Any,
        ] = {}

        self._index_manifests: dict[
            str,
            dict[str, Any],
        ] = {}

        self._textual_alias_cache: dict[
            str,
            str,
        ] = {}

        self._runtime_alias_hint: dict[
            str,
            str,
        ] = {}

    def _load_index_set(
        self,
    ) -> dict[str, Any]:
        if not self.rustfs.object_exists(
            self.indexes_bucket,
            ENRICHMENT_INDEX_CURRENT_KEY,
        ):
            raise FileNotFoundError(
                "Índices de enrichment não publicados: "
                "s3://"
                f"{self.indexes_bucket}/"
                f"{ENRICHMENT_INDEX_CURRENT_KEY}"
            )

        payload = json.loads(
            self.rustfs.download_bytes(
                self.indexes_bucket,
                ENRICHMENT_INDEX_CURRENT_KEY,
            ).decode(
                "utf-8"
            )
        )

        if (
            str(
                payload.get(
                    "status"
                )
            )
            != "success"
        ):
            raise RuntimeError(
                "enrichment/current.json "
                "não possui status=success."
            )

        indexes = (
            payload.get(
                "indexes"
            )
            or {}
        )

        if not indexes:
            raise RuntimeError(
                "enrichment/current.json "
                "não possui indexes."
            )

        return payload

    def resolve_alias(
        self,
        *,
        method: str,
        textual_index_id: str | None,
    ) -> str:
        fallback = (
            self.runtime_config
            .fallback_embedder
        )

        indexes = (
            self._index_set.get(
                "indexes"
            )
            or {}
        )

        if method == "bm25":
            return self._validated_alias(
                fallback
            )

        index_id = str(
            textual_index_id
            or ""
        ).strip()

        if not index_id:
            return self._validated_alias(
                fallback
            )

        cached = (
            self._textual_alias_cache
            .get(
                index_id
            )
        )

        if cached:
            return cached

        manifest_key = (
            f"{index_id}/manifest.json"
        )

        alias = ""

        try:
            if self.rustfs.object_exists(
                self.indexes_bucket,
                manifest_key,
            ):
                manifest = json.loads(
                    self.rustfs.download_bytes(
                        self.indexes_bucket,
                        manifest_key,
                    ).decode(
                        "utf-8"
                    )
                )

                alias = str(
                    (
                        manifest.get(
                            "embedding"
                        )
                        or {}
                    ).get(
                        "model_alias"
                    )
                    or ""
                ).strip()

        except Exception as exc:
            logger.warning(
                "enrichment_textual_alias_resolution_failed "
                "index_id=%s error=%s",
                index_id,
                exc,
            )

        if not alias:
            alias = fallback

        if alias not in indexes:
            logger.warning(
                "enrichment_alias_not_published "
                "requested=%s fallback=%s",
                alias,
                fallback,
            )
            alias = fallback

        alias = self._validated_alias(
            alias
        )

        self._textual_alias_cache[
            index_id
        ] = alias

        return alias

    def runtime_alias_hint(
        self,
        *,
        method: str,
    ) -> str:
        if method == "bm25":
            return self._validated_alias(
                self.runtime_config
                .fallback_embedder
            )

        cached = (
            self._runtime_alias_hint
            .get(
                method
            )
        )

        if cached:
            return self._validated_alias(
                cached
            )

        # Primeiro request: usa o fallback. Depois que o textual
        # resolver seu index_id, o alias é verificado e memorizado.
        return self._validated_alias(
            self.runtime_config
            .fallback_embedder
        )

    def remember_runtime_alias(
        self,
        *,
        method: str,
        alias: str,
    ) -> None:
        self._runtime_alias_hint[
            method
        ] = self._validated_alias(
            alias
        )

    def _validated_alias(
        self,
        alias: str,
    ) -> str:
        indexes = (
            self._index_set.get(
                "indexes"
            )
            or {}
        )

        if alias not in indexes:
            available = ", ".join(
                sorted(
                    indexes
                )
            )

            raise RuntimeError(
                f"Embedder enrichment '{alias}' "
                "não está publicado. "
                f"Disponíveis: {available}."
            )

        if (
            alias
            not in self.config
            .embedding
            .models
        ):
            raise RuntimeError(
                f"Embedder enrichment '{alias}' "
                "não existe em embedding.models."
            )

        return alias

    def _index_manifest(
        self,
        alias: str,
    ) -> dict[str, Any]:
        if alias in self._index_manifests:
            return self._index_manifests[
                alias
            ]

        item = (
            self._index_set[
                "indexes"
            ][alias]
        )

        manifest_key = str(
            item.get(
                "manifest_key"
            )
            or ""
        ).strip()

        if not manifest_key:
            raise RuntimeError(
                "Manifest key ausente para "
                f"enrichment alias={alias}."
            )

        manifest = json.loads(
            self.rustfs.download_bytes(
                self.indexes_bucket,
                manifest_key,
            ).decode(
                "utf-8"
            )
        )

        if (
            str(
                manifest.get(
                    "status"
                )
            )
            != "success"
        ):
            raise RuntimeError(
                "Índice enrichment sem "
                f"status=success: {alias}."
            )

        self._index_manifests[
            alias
        ] = manifest

        return manifest

    def _embedder(
        self,
        alias: str,
    ):
        if alias in self._embedders:
            return self._embedders[
                alias
            ]

        model_config = (
            self.config
            .embedding
            .models[
                alias
            ]
        )

        embedder = (
            self.registry.create(
                model_config,
                normalize_embeddings=(
                    self.config
                    .embedding
                    .index
                    .normalize_embeddings
                ),
                device=(
                    self.config
                    .embedding
                    .execution
                    .device
                ),
                batch_size=(
                    self.config
                    .embedding
                    .execution
                    .batch_size
                ),
            )
        )

        manifest = (
            self._index_manifest(
                alias
            )
        )

        expected = (
            manifest.get(
                "embedding"
            )
            or {}
        )

        info = embedder.info

        identity = {
            "model": (
                info.model,
                expected.get(
                    "model"
                ),
            ),
            "resolved_revision": (
                info.resolved_revision,
                expected.get(
                    "resolved_revision"
                ),
            ),
            "dimension": (
                info.dimension,
                expected.get(
                    "dimension"
                ),
            ),
            "query_prompt_name": (
                info.query_prompt_name,
                expected.get(
                    "query_prompt_name"
                ),
            ),
        }

        mismatches = {
            key: values
            for key, values
            in identity.items()
            if values[0]
            != values[1]
        }

        if mismatches:
            raise RuntimeError(
                "Embedder runtime incompatível "
                "com índice enrichment "
                f"{alias}: {mismatches}"
            )

        self._embedders[
            alias
        ] = embedder

        return embedder

    @staticmethod
    def _embed_query(
        embedder,
        query: str,
    ) -> np.ndarray:
        batch_method = getattr(
            embedder,
            "embed_queries",
            None,
        )

        if callable(
            batch_method
        ):
            vectors = np.asarray(
                batch_method(
                    [query]
                ),
                dtype=np.float32,
            )

            if (
                vectors.ndim != 2
                or vectors.shape[0] != 1
            ):
                raise ValueError(
                    "embed_queries retornou "
                    f"shape inválido: {vectors.shape}"
                )

            vector = vectors[0]

        else:
            single_method = getattr(
                embedder,
                "embed_query",
                None,
            )

            if not callable(
                single_method
            ):
                raise AttributeError(
                    "Embedder não expõe "
                    "embed_queries() nem "
                    "embed_query()."
                )

            vector = np.asarray(
                single_method(
                    query
                ),
                dtype=np.float32,
            )

            if (
                vector.ndim == 2
                and vector.shape[0] == 1
            ):
                vector = vector[0]

            if vector.ndim != 1:
                raise ValueError(
                    "embed_query retornou "
                    f"shape inválido: {vector.shape}"
                )

        if not np.isfinite(
            vector
        ).all():
            raise ValueError(
                "Embedding da query contém "
                "NaN ou infinito."
            )

        return vector

    def _qdrant_search(
        self,
        *,
        collection: str,
        vector: np.ndarray,
        limit: int,
    ) -> list[
        dict[str, Any]
    ]:
        query_vector = np.asarray(
            vector,
            dtype=np.float32,
        ).tolist()

        query_points = getattr(
            self.client,
            "query_points",
            None,
        )

        if callable(
            query_points
        ):
            response = query_points(
                collection_name=(
                    collection
                ),
                query=query_vector,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

            points = getattr(
                response,
                "points",
                response,
            )

        else:
            search = getattr(
                self.client,
                "search",
                None,
            )

            if not callable(
                search
            ):
                raise RuntimeError(
                    "QdrantClient não possui "
                    "query_points nem search."
                )

            points = search(
                collection_name=(
                    collection
                ),
                query_vector=(
                    query_vector
                ),
                limit=limit,
                with_payload=True,
                with_vectors=False,
            )

        hits: list[
            dict[str, Any]
        ] = []

        for point in points:
            payload = dict(
                getattr(
                    point,
                    "payload",
                    None,
                )
                or {}
            )

            hits.append(
                {
                    **payload,
                    "score": (
                        _safe_float(
                            getattr(
                                point,
                                "score",
                                0.0,
                            )
                        )
                    ),
                    "qdrant_point_id": str(
                        getattr(
                            point,
                            "id",
                            "",
                        )
                    ),
                }
            )

        return hits

    def search(
        self,
        query: str,
        *,
        method: str,
        textual_index_id: str | None = None,
        alias_hint: str | None = None,
    ) -> EnrichmentRetrievalResponse:
        query = str(
            query
        ).strip()

        if not query:
            raise ValueError(
                "Query enrichment vazia."
            )

        if alias_hint:
            alias = self._validated_alias(
                alias_hint
            )
        else:
            alias = self.resolve_alias(
                method=method,
                textual_index_id=(
                    textual_index_id
                ),
            )

        index_item = (
            self._index_set[
                "indexes"
            ][alias]
        )

        index_id = str(
            index_item[
                "index_id"
            ]
        )

        collection = str(
            index_item[
                "collection"
            ]
        )

        manifest = (
            self._index_manifest(
                alias
            )
        )

        embedder = (
            self._embedder(
                alias
            )
        )

        vector = (
            self._embed_query(
                embedder,
                query,
            )
        )

        expected_dimension = int(
            (
                manifest.get(
                    "embedding"
                )
                or {}
            ).get(
                "dimension"
            )
            or 0
        )

        if (
            expected_dimension
            and vector.shape
            != (
                expected_dimension,
            )
        ):
            raise ValueError(
                "Dimensão da query incompatível "
                "com índice enrichment. "
                f"alias={alias} "
                f"esperado={expected_dimension} "
                f"recebido={vector.shape}"
            )

        hits = self._qdrant_search(
            collection=collection,
            vector=vector,
            limit=(
                self.runtime_config
                .top_k
            ),
        )

        return EnrichmentRetrievalResponse(
            hits=hits,
            embedder_alias=alias,
            index_id=index_id,
            collection=collection,
        )
