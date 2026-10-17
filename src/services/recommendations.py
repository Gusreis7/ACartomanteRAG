from __future__ import annotations

import io
import json
import logging
import random
from threading import Lock
from urllib.parse import urlparse

import pandas as pd

BRONZE_CURRENT_KEY = "current.json"
ALLOWED_DOMAIN_SUFFIX = "dominiopublico.mec.gov.br"

logger = logging.getLogger("acartomante.recommendations")


class RecommendationUnavailableError(RuntimeError):
    pass


class WorkRecommendationService:
    """Catálogo leve de obras para a leitura gratuita da interface.

    A Bronze do projeto é versionada. O catálogo é carregado a partir de:

        s3://<bronze_bucket>/current.json
            -> manifest_key
            -> datasets/<bronze_artifact_id>/manifest.parquet

    Depois da primeira leitura bem-sucedida, o catálogo fica em memória por
    processo. Ele não guarda estado de usuário.
    """

    def __init__(
        self, *, storage, bucket: str, current_key: str = BRONZE_CURRENT_KEY
    ) -> None:
        self.storage = storage
        self.bucket = bucket
        self.current_key = current_key
        self._catalog: list[dict[str, str | None]] | None = None
        self._catalog_source: str | None = None
        self._catalog_lock = Lock()
        self._rng = random.SystemRandom()

    @classmethod
    def from_runtime(cls, *, config, rag) -> "WorkRecommendationService":
        return cls(
            storage=rag.retrieval.rustfs,
            bucket=config.storage.bronze_bucket,
            current_key=config.storage.bronze_current_key,
        )

    @staticmethod
    def _text(value) -> str:
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        return str(value).strip()

    @classmethod
    def _official_url(cls, row: pd.Series) -> str:
        # A página de detalhe é preferida: mantém o usuário dentro do
        # Domínio Público e ainda oferece o download da obra.
        for column in ("url_detalhe", "url_download"):
            value = cls._text(row.get(column))
            if not value:
                continue

            parsed = urlparse(value)
            hostname = (parsed.hostname or "").lower()

            if parsed.scheme in {"http", "https"} and (
                hostname == ALLOWED_DOMAIN_SUFFIX
                or hostname.endswith("." + ALLOWED_DOMAIN_SUFFIX)
            ):
                return value

        return ""

    def _resolve_current_manifest(self) -> tuple[str, str]:
        """Resolve o manifest da Bronze atual pelo ponteiro current.json."""
        if not self.storage.object_exists(
            self.bucket,
            self.current_key,
        ):
            raise RecommendationUnavailableError(
                "A camada Bronze não possui current.json."
            )

        try:
            current_payload = self.storage.download_bytes(
                self.bucket,
                self.current_key,
            )
            current = json.loads(current_payload.decode("utf-8"))
        except Exception as exc:
            raise RecommendationUnavailableError(
                "Não foi possível ler o current.json da camada Bronze."
            ) from exc

        if not isinstance(current, dict):
            raise RecommendationUnavailableError(
                "current.json da camada Bronze possui formato inválido."
            )

        artifact_id = self._text(current.get("bronze_artifact_id"))
        manifest_key = self._text(current.get("manifest_key"))

        if not artifact_id:
            raise RecommendationUnavailableError(
                "current.json da camada Bronze não informa bronze_artifact_id."
            )

        if not manifest_key:
            raise RecommendationUnavailableError(
                "current.json da camada Bronze não informa manifest_key."
            )

        if not self.storage.object_exists(
            self.bucket,
            manifest_key,
        ):
            raise RecommendationUnavailableError(
                "O manifest apontado pelo current.json da Bronze não existe."
            )

        return artifact_id, manifest_key

    def _load_catalog(self) -> list[dict[str, str | None]]:
        artifact_id, manifest_key = self._resolve_current_manifest()

        logger.info(
            "recommendation_catalog_loading bucket=%s artifact=%s manifest=%s",
            self.bucket,
            artifact_id,
            manifest_key,
        )

        try:
            payload = self.storage.download_bytes(
                self.bucket,
                manifest_key,
            )
            frame = pd.read_parquet(io.BytesIO(payload))
        except Exception as exc:
            raise RecommendationUnavailableError(
                "Não foi possível ler o manifest da Bronze atual."
            ) from exc

        required = {"titulo"}
        missing = required - set(frame.columns)
        if missing:
            raise RecommendationUnavailableError(
                "Manifest Bronze sem os campos necessários para recomendações: "
                + ", ".join(sorted(missing))
            )

        if "status_bronze" in frame.columns:
            frame = frame[frame["status_bronze"].astype(str).eq("success")].copy()

        catalog: list[dict[str, str | None]] = []
        seen_titles: set[str] = set()

        for _, row in frame.iterrows():
            title = self._text(row.get("titulo"))
            source_url = self._official_url(row)
            normalized = " ".join(title.casefold().split())

            if not title or not source_url or normalized in seen_titles:
                continue

            seen_titles.add(normalized)

            catalog.append(
                {
                    "title": title,
                    "author": (self._text(row.get("autor")) or "Machado de Assis"),
                    "category": (self._text(row.get("categoria")) or None),
                    "source_url": source_url,
                }
            )

        if len(catalog) < 3:
            raise RecommendationUnavailableError(
                "A Bronze atual não possui três obras com link válido "
                "do Domínio Público."
            )

        self._catalog_source = artifact_id

        logger.info(
            "recommendation_catalog_ready artifact=%s works=%d",
            artifact_id,
            len(catalog),
        )

        return catalog

    def _get_catalog(
        self,
    ) -> list[dict[str, str | None]]:
        if self._catalog is not None:
            return self._catalog

        with self._catalog_lock:
            if self._catalog is None:
                try:
                    self._catalog = self._load_catalog()
                except RecommendationUnavailableError:
                    raise
                except Exception as exc:
                    logger.exception(
                        "recommendation_catalog_load_failed " "bucket=%s current=%s",
                        self.bucket,
                        self.current_key,
                    )
                    raise RecommendationUnavailableError(
                        "Não foi possível carregar o catálogo de obras."
                    ) from exc

        return self._catalog

    def recommend(
        self,
        *,
        count: int = 3,
        exclude_titles: list[str] | None = None,
    ) -> list[dict[str, str | None]]:
        catalog = self._get_catalog()
        count = max(
            1,
            min(int(count), 3),
        )

        excluded = {
            " ".join(str(title).casefold().split())
            for title in (exclude_titles or [])
            if str(title).strip()
        }

        available = [
            item
            for item in catalog
            if " ".join(str(item["title"]).casefold().split()) not in excluded
        ]

        # Em um acervo muito pequeno, é preferível repetir uma obra a
        # impedir uma nova leitura.
        pool = available if len(available) >= count else catalog

        return [
            dict(item)
            for item in self._rng.sample(
                pool,
                count,
            )
        ]
