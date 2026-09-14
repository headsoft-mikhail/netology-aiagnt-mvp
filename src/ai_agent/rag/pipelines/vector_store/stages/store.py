import dataclasses
import logging
import typing
import uuid

import qdrant_client
from qdrant_client import models as qdrant_models
from qdrant_client.conversions import common_types as qdrant_types
from qdrant_client.conversions.common_types import VectorParams

from ai_agent.rag import models as rag_models
from ai_agent.rag.config import VectorStoreConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)

_DISTANCES_MAPPING: typing.Final = {
    "cosine": qdrant_models.Distance.COSINE,
    "dot": qdrant_models.Distance.DOT,
    "euclid": qdrant_models.Distance.EUCLID,
    "manhattan": qdrant_models.Distance.MANHATTAN,
}


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class QdrantStore:
    config: VectorStoreConfig
    client: qdrant_client.QdrantClient

    def create_collection(self) -> None:
        collection_name: typing.Final = self.config.collection_name

        if self.client.collection_exists(collection_name):
            if self.config.recreate_collection:
                LOGGER_OBJ.info(f"Collection {collection_name} already exists. Recreating...")
                self.client.delete_collection(collection_name)
            else:
                collection: typing.Final = self.client.get_collection(collection_name)
                vectors_config: typing.Final = collection.config.params.vectors
                vectors_config_list: typing.Final = (
                    [vectors_config]
                    if isinstance(vectors_config, VectorParams)
                    else vectors_config.values()
                    if vectors_config
                    else []
                )
                if vectors_config_list and any(
                    (
                        self.config.vectors_dimensions == one_vectors_config.size
                        and _DISTANCES_MAPPING[self.config.distance] == one_vectors_config.distance
                    )
                    for one_vectors_config in vectors_config_list
                ):
                    LOGGER_OBJ.info(f"Collection {collection_name} already exists. Using existing collection.")
                    return
                else:
                    LOGGER_OBJ.warning("Collection exists, but it's config doesn't fit provided vectors. Recreating...")
                    self.client.delete_collection(collection_name)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=qdrant_models.VectorParams(
                size=self.config.vectors_dimensions,
                distance=_DISTANCES_MAPPING[self.config.distance],
            ),
        )

        LOGGER_OBJ.info(f"Collection {collection_name} created.")

    def upload(self, embedded_chunks: list[rag_models.EmbeddedChunk]) -> int:
        total_uploaded = 0

        for start in range(0, len(embedded_chunks), self.config.upload_batch_size):
            batch = embedded_chunks[start : start + self.config.upload_batch_size]
            points = [
                qdrant_models.PointStruct(
                    id=uuid.uuid5(uuid.NAMESPACE_URL, one_embedded_chunk.id),
                    vector=one_embedded_chunk.embedding,
                    payload={
                        "chunk_id": one_embedded_chunk.id,
                        "text": one_embedded_chunk.text,
                        **one_embedded_chunk.metadata.model_dump(),
                    },
                )
                for one_embedded_chunk in batch
                if one_embedded_chunk.is_valid_for_vector_store(self.config)
            ]

            self.client.upsert(
                collection_name=self.config.collection_name,
                points=points,
                wait=True,
            )
            total_uploaded += len(points)
            LOGGER_OBJ.info(f"Uploaded {total_uploaded}/{len(embedded_chunks)} embeddings.")

        return total_uploaded

    def _get_points_page(
        self, offset: qdrant_types.PointId | None = None
    ) -> tuple[list[qdrant_models.Record], qdrant_types.PointId | None]:
        return self.client.scroll(
            collection_name=self.config.collection_name,
            offset=offset,
            limit=self.config.upload_batch_size,
            with_payload=True,
            with_vectors=True,
        )

    def get_all_points(self) -> list[qdrant_models.Record]:
        points: typing.Final[list[qdrant_models.Record]] = []
        offset: qdrant_types.PointId | None = None

        while True:
            points_batch, offset = self._get_points_page(offset)
            points.extend(points_batch)

            if offset is None:
                break

        return points

    def close(self) -> None:
        self.client.close()
