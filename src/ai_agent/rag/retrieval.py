import typing

import qdrant_client
import qdrant_client.models
from qdrant_client.conversions.common_types import ScoredPoint
from sentence_transformers import SentenceTransformer

from ai_agent.rag.config import RetrievalConfig


class RetrievalClient:
    def __init__(self, config: RetrievalConfig, *, local_files_only: bool = True) -> None:
        self.config = config
        self.transformer = SentenceTransformer(
            self.config.embedding_model,
            local_files_only=local_files_only,
        )

    def top_k(self, query: str) -> list[ScoredPoint]:
        normalized_query: typing.Final = query.strip()
        if not normalized_query:
            raise ValueError("Retrieval query must not be empty.")

        query_embedding: typing.Final = self.transformer.encode(
            f"query: {normalized_query}",
            normalize_embeddings=True,
        ).tolist()
        client: typing.Final = qdrant_client.QdrantClient(path=str(self.config.vector_store_path))
        query_filter: typing.Final = qdrant_client.models.Filter(
            must_not=[
                qdrant_client.models.FieldCondition(
                    key="source",
                    match=qdrant_client.models.MatchValue(value=document),
                )
                for document in self.config.excluded_documents
            ],
        )

        try:
            return client.query_points(
                collection_name=self.config.collection_name,
                query=query_embedding,
                query_filter=query_filter,
                limit=self.config.search_top_k,
                with_payload=True,
            ).points
        finally:
            client.close()
