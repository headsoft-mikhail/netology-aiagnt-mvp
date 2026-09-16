import dataclasses
import logging
import math
import typing

import pydantic
from qdrant_client import models as qdrant_models

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import VectorStoreConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorStoreValidationMetrics:
    input_embeddings_count: int
    stored_points_count: int
    invalid_dimensions_count: int
    empty_vectors_count: int
    invalid_vectors_count: int
    missing_text_count: int
    missing_chunk_id_count: int
    missing_document_id_count: int
    invalid_metadata_count: int

    @property
    def overall_validity(self) -> bool:
        return not any(
            (
                self.invalid_dimensions_count,
                self.empty_vectors_count,
                self.invalid_vectors_count,
                self.missing_text_count,
                self.missing_chunk_id_count,
                self.missing_document_id_count,
                self.invalid_metadata_count,
                self.input_embeddings_count != self.stored_points_count,
            )
        )


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorStoreValidator:
    config: VectorStoreConfig

    def validate(
        self,
        embedded_chunks: list[models.EmbeddedChunk],
        stored_points: list[qdrant_models.Record],
    ) -> VectorStoreValidationMetrics:
        invalid_dimensions_count = 0
        empty_vectors_count = 0
        invalid_vectors_count = 0
        missing_text_count = 0
        missing_document_id_count = 0
        invalid_metadata_count = 0

        input_chunk_ids: typing.Final = {one_embedded_chunk.id for one_embedded_chunk in embedded_chunks}
        stored_chunk_ids: typing.Final[set[str]] = set()

        for one_point in stored_points:
            one_vector = one_point.vector
            if one_vector and isinstance(one_vector, list):
                if len(one_vector) != self.config.vectors_dimensions:
                    invalid_dimensions_count += 1

                if not all(math.isfinite(one_value) for one_value in typing.cast("list[float]", one_vector)):
                    invalid_vectors_count += 1
            else:
                empty_vectors_count += 1

            one_point_payload = one_point.payload or {}

            if chunk_id := one_point_payload.get("chunk_id"):
                stored_chunk_ids.add(str(chunk_id))

            if not one_point_payload.get("text"):
                missing_text_count += 1

            if not one_point_payload.get("document_id"):
                missing_document_id_count += 1

            try:
                models.VectorStorePointPayload.model_validate(one_point_payload)
            except pydantic.ValidationError:
                invalid_metadata_count += 1

        metrics: typing.Final = VectorStoreValidationMetrics(
            input_embeddings_count=len(embedded_chunks),
            stored_points_count=len(stored_points),
            invalid_dimensions_count=invalid_dimensions_count,
            empty_vectors_count=empty_vectors_count,
            invalid_vectors_count=invalid_vectors_count,
            missing_text_count=missing_text_count,
            missing_chunk_id_count=len(input_chunk_ids - stored_chunk_ids),
            missing_document_id_count=missing_document_id_count,
            invalid_metadata_count=invalid_metadata_count,
        )
        self._log_metrics(metrics)
        return metrics

    def _log_metrics(self, metrics: VectorStoreValidationMetrics):
        LOGGER_OBJ.info(
            (
                "Vector store validation: "
                "input=%d, "
                "stored=%d, "
                "invalid_dimensions=%d, "
                "empty_vectors=%d, "
                "invalid_vectors=%d, "
                "missing_text=%d, "
                "missing_chunk_id=%d, "
                "missing_document_id=%d, "
                "invalid_metadata=%d, "
                "valid=%s"
            ),
            metrics.input_embeddings_count,
            metrics.stored_points_count,
            metrics.invalid_dimensions_count,
            metrics.empty_vectors_count,
            metrics.invalid_vectors_count,
            metrics.missing_text_count,
            metrics.missing_chunk_id_count,
            metrics.missing_document_id_count,
            metrics.invalid_metadata_count,
            metrics.overall_validity,
        )
