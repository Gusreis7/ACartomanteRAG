from __future__ import annotations

import ssl
from pathlib import Path
from typing import Any, Protocol, Sequence

from qdrant_client import QdrantClient, models


class QdrantConfigProtocol(Protocol):
    url: str
    api_key: str


class QdrantManager:
    DISTANCES = {
        "cosine": models.Distance.COSINE,
        "dot": models.Distance.DOT,
        "euclid": models.Distance.EUCLID,
    }

    def __init__(
        self,
        url: str,
        api_key: str,
        timeout: int = 30,
    ) -> None:
        self.url = url
        self.client = QdrantClient(
            url=url,
            api_key=api_key,
            timeout=timeout,
        )

    @classmethod
    def from_config(
        cls,
        config: QdrantConfigProtocol,
        timeout: int = 30,
    ) -> "QdrantManager":
        return cls(
            url=config.url,
            api_key=config.api_key,
            timeout=timeout,
        )

    def info(self):
        return self.client.info()

    def list_collections(self) -> list[str]:
        response = self.client.get_collections()
        return [collection.name for collection in response.collections]

    def collection_exists(
        self,
        collection_name: str,
    ) -> bool:
        return self.client.collection_exists(collection_name=collection_name)

    def get_collection(
        self,
        collection_name: str,
    ):
        return self.client.get_collection(collection_name=collection_name)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance: str = "cosine",
    ) -> None:
        if vector_size <= 0:
            raise ValueError("vector_size deve ser maior que zero.")

        distance_normalized = distance.strip().lower()

        if distance_normalized not in self.DISTANCES:
            valid = ", ".join(sorted(self.DISTANCES))
            raise ValueError(f"Distance '{distance}' inválida. Use: {valid}.")

        if self.collection_exists(collection_name):
            raise ValueError(f"Collection já existe: {collection_name}")

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=self.DISTANCES[distance_normalized],
            ),
        )

    def delete_collection(
        self,
        collection_name: str,
    ) -> None:
        self.client.delete_collection(collection_name=collection_name)

    def upsert_points(
        self,
        collection_name: str,
        ids: Sequence[int | str],
        vectors: Sequence[Sequence[float]],
        payloads: Sequence[dict[str, Any]] | None = None,
        wait: bool = True,
    ):
        if len(ids) != len(vectors):
            raise ValueError("ids e vectors devem ter a mesma quantidade.")

        if payloads is None:
            payloads = [{} for _ in ids]

        if len(ids) != len(payloads):
            raise ValueError("ids e payloads devem ter a mesma quantidade.")

        points = [
            models.PointStruct(
                id=point_id,
                vector=list(vector),
                payload=payload,
            )
            for point_id, vector, payload in zip(ids, vectors, payloads)
        ]

        return self.client.upsert(
            collection_name=collection_name,
            points=points,
            wait=wait,
        )

    def query(
        self,
        collection_name: str,
        vector: Sequence[float],
        limit: int = 10,
        query_filter: models.Filter | None = None,
        score_threshold: float | None = None,
        with_payload: bool = True,
    ) -> list[models.ScoredPoint]:
        if limit <= 0:
            raise ValueError("limit deve ser maior que zero.")

        response = self.client.query_points(
            collection_name=collection_name,
            query=list(vector),
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=with_payload,
            with_vectors=False,
        )

        return response.points

    def count(
        self,
        collection_name: str,
        exact: bool = True,
    ) -> int:
        response = self.client.count(
            collection_name=collection_name,
            exact=exact,
        )
        return response.count

    def delete_points(
        self,
        collection_name: str,
        point_ids: Sequence[int | str],
        wait: bool = True,
    ):
        return self.client.delete(
            collection_name=collection_name,
            points_selector=models.PointIdsList(points=list(point_ids)),
            wait=wait,
        )

    def close(self) -> None:
        self.client.close()


def check_qdrant_collection_ready(
    *,
    url: str,
    api_key: str | None,
    collection_name: str,
    timeout_seconds: float = 5.0,
) -> None:
    """
    Verifica Qdrant + collection sem inicializar embedder ou reranker.
    Levanta exceção quando o serviço, autenticação ou collection falhar.
    """
    from urllib.parse import quote

    import httpx

    base_url = url.rstrip("/")
    encoded_collection = quote(collection_name, safe="")

    headers: dict[str, str] = {}
    if api_key:
        headers["api-key"] = api_key

    response = httpx.get(
        f"{base_url}/collections/{encoded_collection}",
        headers=headers,
        timeout=timeout_seconds,
    )

    if response.status_code == 404:
        raise ValueError(
            "Collection Qdrant não encontrada: "
            f"{collection_name}"
        )

    response.raise_for_status()

