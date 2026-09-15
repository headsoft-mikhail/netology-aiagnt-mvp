import dataclasses
import hashlib
import itertools
import logging
import re
import typing

from ai_agent.rag import models
from ai_agent.rag.config import ChunkingConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ChunkingValidationMetrics:
    total_chunks: int
    empty_chunks_count: int
    oversized_chunks_count: int
    undersized_chunks_count: int
    invalid_metadata_count: int
    invalid_ids_count: int
    lineage_errors_count: int
    overlap_errors_count: int

    @property
    def overall_validity(self) -> bool:
        return not any(
            (
                self.empty_chunks_count,
                self.oversized_chunks_count,
                self.undersized_chunks_count,
                self.invalid_metadata_count,
                self.invalid_ids_count,
                self.lineage_errors_count,
                self.overlap_errors_count,
            )
        )


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ChunkValidator:
    config: ChunkingConfig
    _SENTENCE_PATTERN: typing.Final = re.compile(r"(?<=[.!?])(?:[»”\"])?\s+")

    def validate(
        self,
        chunks: list[models.Chunk],
        documents: list[models.PreparedDocument],
    ) -> ChunkingValidationMetrics:
        document_ids: typing.Final = {document.metadata.document_id for document in documents}

        empty_chunks_count = 0
        oversized_chunks_count = 0
        undersized_chunks_count = 0
        invalid_metadata_count = 0
        invalid_ids_count = 0
        lineage_errors_count = 0
        overlap_errors_count = 0

        chunks_by_document: typing.Final[dict[str, list[models.Chunk]]] = {}

        for one_chunk in chunks:
            metadata = one_chunk.metadata

            chunks_by_document.setdefault(metadata.document_id, []).append(one_chunk)

            if not one_chunk.text.strip():
                empty_chunks_count += 1

            if metadata.chunk_token_count > metadata.chunk_size:
                oversized_chunks_count += 1

            if metadata.chunk_token_count < self._minimum_chunk_tokens():
                undersized_chunks_count += 1

            if not self._validate_metadata(one_chunk):
                invalid_metadata_count += 1

            if not self._validate_id(one_chunk):
                invalid_ids_count += 1

            if metadata.document_id not in document_ids:
                lineage_errors_count += 1

        for document_chunks in chunks_by_document.values():
            document_chunks.sort(key=lambda chunk: chunk.metadata.position)

            if not self._validate_positions(document_chunks):
                invalid_metadata_count += 1

            overlap_errors_count += self._validate_overlap(document_chunks)

        metrics: typing.Final = ChunkingValidationMetrics(
            total_chunks=len(chunks),
            empty_chunks_count=empty_chunks_count,
            oversized_chunks_count=oversized_chunks_count,
            undersized_chunks_count=undersized_chunks_count,
            invalid_metadata_count=invalid_metadata_count,
            invalid_ids_count=invalid_ids_count,
            lineage_errors_count=lineage_errors_count,
            overlap_errors_count=overlap_errors_count,
        )
        self._log_metrics(metrics)
        return metrics

    def _minimum_chunk_tokens(self) -> int:
        return max(1, int(self.config.chunk_size * 0.1))

    def _validate_metadata(
        self,
        chunk: models.Chunk,
    ) -> bool:
        metadata: typing.Final = chunk.metadata

        return (
            bool(metadata.document_id)
            and metadata.position >= 0
            and metadata.chunk_token_count >= 0
            and metadata.chunk_size == self.config.chunk_size
            and metadata.chunk_overlap == self.config.chunk_overlap
            and metadata.chunking_strategy == self.config.strategy
        )

    def _validate_id(self, chunk: models.Chunk) -> bool:
        metadata: typing.Final = chunk.metadata
        text_hash: typing.Final = hashlib.sha256(chunk.text.encode("utf-8")).hexdigest()

        expected_id: typing.Final = hashlib.sha256(
            (f"{metadata.document_id}:{metadata.position}:{text_hash}").encode()
        ).hexdigest()

        return chunk.id == expected_id

    def _validate_positions(self, chunks: list[models.Chunk]) -> bool:
        return all(chunk.metadata.position == index for index, chunk in enumerate(chunks))

    def _validate_overlap(self, chunks: list[models.Chunk]) -> int:
        if len(chunks) < 2 or self.config.chunk_overlap == 0:
            return 0

        errors = 0

        for previous, current in itertools.pairwise(chunks):
            if previous.metadata.section != current.metadata.section:
                continue

            if self.config.strategy in {"token", "text"}:
                if not self._has_token_overlap(previous, current):
                    errors += 1

            elif self.config.strategy == "sentence":
                if not self._has_sentence_overlap(previous, current):
                    errors += 1

            elif self.config.strategy == "paragraph" and not self._has_paragraph_overlap(previous, current):
                errors += 1

        return errors

    def _has_token_overlap(
        self,
        previous: models.Chunk,
        current: models.Chunk,
    ) -> bool:
        previous_tokens: typing.Final = self.config.tokenizer.encode(previous.text)
        current_tokens: typing.Final = self.config.tokenizer.encode(current.text)

        overlap_tokens: typing.Final = min(
            self.config.chunk_overlap,
            len(previous_tokens),
            len(current_tokens),
        )

        if overlap_tokens == 0:
            return True

        return previous_tokens[-overlap_tokens:] == current_tokens[:overlap_tokens]

    def _has_sentence_overlap(
        self,
        previous: models.Chunk,
        current: models.Chunk,
    ) -> bool:
        previous_sentences: typing.Final = self._split_sentences(previous.text)
        current_sentences: typing.Final = self._split_sentences(current.text)

        if not previous_sentences or not current_sentences:
            return False

        overlap_tokens = 0
        expected_overlap: typing.Final[list[str]] = []

        for sentence in reversed(previous_sentences):
            sentence_tokens = self._count_tokens(sentence)

            if overlap_tokens + sentence_tokens > self.config.chunk_overlap:
                break

            expected_overlap.insert(0, sentence)
            overlap_tokens += sentence_tokens

        if not expected_overlap:
            return True

        return current_sentences[: len(expected_overlap)] == expected_overlap

    def _has_paragraph_overlap(
        self,
        previous: models.Chunk,
        current: models.Chunk,
    ) -> bool:
        previous_paragraphs: typing.Final = self._split_paragraphs(previous.text)
        current_paragraphs: typing.Final = self._split_paragraphs(current.text)

        if not previous_paragraphs or not current_paragraphs:
            return False

        overlap_tokens = 0
        expected_overlap: typing.Final[list[str]] = []

        for paragraph in reversed(previous_paragraphs):
            paragraph_tokens = self._count_tokens(paragraph)

            if overlap_tokens + paragraph_tokens > self.config.chunk_overlap:
                break

            expected_overlap.insert(0, paragraph)
            overlap_tokens += paragraph_tokens

        if not expected_overlap:
            return True

        return current_paragraphs[: len(expected_overlap)] == expected_overlap

    def _split_sentences(
        self,
        text: str,
    ) -> list[str]:
        text = " ".join(text.split()).strip()

        if not text:
            return []

        sentences: typing.Final = self._SENTENCE_PATTERN.split(text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_paragraphs(
        self,
        text: str,
    ) -> list[str]:
        return [paragraph.strip() for paragraph in text.split("\n") if paragraph.strip()]

    def _count_tokens(
        self,
        text: str,
    ) -> int:
        return len(self.config.tokenizer.encode(text))

    def _log_metrics(
        self,
        metrics: ChunkingValidationMetrics,
    ) -> None:
        LOGGER_OBJ.info(
            (
                "Chunk validation: "
                "total=%d, "
                "empty=%d, "
                "oversized=%d, "
                "undersized=%d, "
                "invalid_metadata=%d, "
                "invalid_ids=%d, "
                "lineage_errors=%d, "
                "overlap_errors=%d, "
                "overall_validity=%s"
            ),
            metrics.total_chunks,
            metrics.empty_chunks_count,
            metrics.oversized_chunks_count,
            metrics.undersized_chunks_count,
            metrics.invalid_metadata_count,
            metrics.invalid_ids_count,
            metrics.lineage_errors_count,
            metrics.overlap_errors_count,
            metrics.overall_validity,
        )
