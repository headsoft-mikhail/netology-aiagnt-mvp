import dataclasses
import hashlib
import logging
import math
import typing

from ai_agent.rag import models
from ai_agent.rag.pipelines.embeddings.stages.embedding import EmbeddingModel

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingValidationMetrics:
    total_chunks: int
    total_embeddings: int
    empty_embeddings_count: int
    invalid_values_count: int
    invalid_dimensions_count: int
    invalid_metadata_count: int
    duplicate_ids_count: int
    lineage_errors_count: int
    text_mismatch_count: int

    @property
    def overall_validity(self) -> bool:
        return not any(
            (
                self.empty_embeddings_count,
                self.invalid_values_count,
                self.invalid_dimensions_count,
                self.invalid_metadata_count,
                self.duplicate_ids_count,
                self.lineage_errors_count,
                self.text_mismatch_count,
            )
        )


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingValidator:
    model: EmbeddingModel

    def validate(
        self,
        chunks: list[models.Chunk],
        embeddings: list[models.EmbeddedChunk],
    ) -> EmbeddingValidationMetrics:
        chunks_by_key: typing.Final = {self._chunk_key(one_chunk): one_chunk for one_chunk in chunks}

        seen_keys: typing.Final[set[tuple[str, int]]] = set()

        empty_embeddings_count = 0
        invalid_values_count = 0
        invalid_dimensions_count = 0
        invalid_metadata_count = 0
        duplicate_ids_count = 0
        lineage_errors_count = 0
        text_mismatch_count = 0

        for one_embedded_chunk in embeddings:
            metadata = one_embedded_chunk.metadata
            key = self._embedded_chunk_key(one_embedded_chunk)

            if key in seen_keys:
                duplicate_ids_count += 1

            seen_keys.add(key)

            chunk = chunks_by_key.get(key)

            if chunk is None:
                lineage_errors_count += 1
            else:
                if one_embedded_chunk.text != chunk.text:
                    text_mismatch_count += 1

                if not self._validate_text_hash(
                    text=one_embedded_chunk.text,
                    expected_hash=metadata.text_hash,
                ):
                    invalid_metadata_count += 1

            if not one_embedded_chunk.embedding:
                empty_embeddings_count += 1
                continue

            if not self._validate_values(one_embedded_chunk.embedding):
                invalid_values_count += 1

            if len(one_embedded_chunk.embedding) != self.model.dimensions:
                invalid_dimensions_count += 1

            if not self._validate_metadata(one_embedded_chunk):
                invalid_metadata_count += 1

        metrics: typing.Final = EmbeddingValidationMetrics(
            total_chunks=len(chunks),
            total_embeddings=len(embeddings),
            empty_embeddings_count=empty_embeddings_count,
            invalid_values_count=invalid_values_count,
            invalid_dimensions_count=invalid_dimensions_count,
            invalid_metadata_count=invalid_metadata_count,
            duplicate_ids_count=duplicate_ids_count,
            lineage_errors_count=lineage_errors_count,
            text_mismatch_count=text_mismatch_count,
        )
        self._log_metrics(metrics)
        return metrics

    def _chunk_key(
        self,
        chunk: models.Chunk,
    ) -> tuple[str, int]:
        return (
            chunk.metadata.document_id,
            chunk.metadata.position,
        )

    def _embedded_chunk_key(self, embedded_chunk: models.EmbeddedChunk) -> tuple[str, int]:
        return (
            embedded_chunk.metadata.document_id,
            embedded_chunk.metadata.position,
        )

    def _validate_values(self, embedding: list[float]) -> bool:
        return all(math.isfinite(value) for value in embedding)

    def _validate_metadata(self, embedded_chunk: models.EmbeddedChunk) -> bool:
        metadata: typing.Final = embedded_chunk.metadata

        return (
            bool(metadata.document_id)
            and metadata.position >= 0
            and metadata.embedding_model == self.model.config.model
            and metadata.embedding_dimensions == self.model.dimensions
        )

    def _validate_text_hash(
        self,
        text: str,
        expected_hash: str,
    ) -> bool:
        actual_hash: typing.Final = hashlib.sha256(text.encode("utf-8")).hexdigest()

        return actual_hash == expected_hash

    def _log_metrics(
        self,
        metrics: EmbeddingValidationMetrics,
    ) -> None:
        LOGGER_OBJ.info(
            (
                "Embedding validation: "
                "chunks=%d, "
                "embeddings=%d, "
                "empty=%d, "
                "invalid_values=%d, "
                "invalid_dimensions=%d, "
                "invalid_metadata=%d, "
                "duplicate_ids=%d, "
                "lineage_errors=%d, "
                "text_mismatch=%d, "
                "valid=%s"
            ),
            metrics.total_chunks,
            metrics.total_embeddings,
            metrics.empty_embeddings_count,
            metrics.invalid_values_count,
            metrics.invalid_dimensions_count,
            metrics.invalid_metadata_count,
            metrics.duplicate_ids_count,
            metrics.lineage_errors_count,
            metrics.text_mismatch_count,
            metrics.overall_validity,
        )
