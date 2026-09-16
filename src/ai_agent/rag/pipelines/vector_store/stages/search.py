import dataclasses
import logging
import typing

import qdrant_client

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import VectorStoreConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorSearchResult:
    query_chunk_id: str
    query_text: str
    results: list[dict[str, typing.Any]]

    @property
    def most_relevant_result(self) -> dict[str, typing.Any]:
        return self.results[0]

    @property
    def is_relevant(self) -> bool:
        most_relevant_result: typing.Final = self.most_relevant_result
        return most_relevant_result.get("score", 0) >= 0.99 and self.query_chunk_id == most_relevant_result.get(
            "metadata", {}
        ).get("chunk_id", "")


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorStoreSearcher:
    config: VectorStoreConfig
    client: qdrant_client.QdrantClient

    def search_batch(self, embedded_chunks: list[models.EmbeddedChunk]) -> list[VectorSearchResult]:
        results: typing.Final[list[VectorSearchResult]] = []
        test_chunks: typing.Final = embedded_chunks[: self.config.search_test_queries]

        for one_embedded_chunk in test_chunks:
            search_results = self.client.query_points(
                collection_name=self.config.collection_name,
                query=one_embedded_chunk.embedding,
                limit=self.config.search_top_k,
                with_payload=True,
            ).points

            results.append(
                VectorSearchResult(
                    query_chunk_id=one_embedded_chunk.id,
                    query_text=one_embedded_chunk.text,
                    results=[
                        {
                            "score": one_point.score,
                            "metadata": one_point.payload or {},
                        }
                        for one_point in search_results
                    ],
                )
            )

        self._log_results(results)
        return results

    def _log_results(self, results: list[VectorSearchResult]) -> None:
        LOGGER_OBJ.info(
            "Vector search completed: queries=%d, top_k=%d",
            len(results),
            self.config.search_top_k,
        )
