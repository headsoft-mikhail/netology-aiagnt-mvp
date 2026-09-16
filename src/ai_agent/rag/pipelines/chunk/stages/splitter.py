import dataclasses
import hashlib
import re
import typing

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import ChunkingConfig


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ChunkSplitter:
    config: ChunkingConfig

    _SENTENCE_PATTERN: typing.Final = re.compile(r"(?<=[.!?])(?:[»”\"])?\s+")

    def split_documents(self, documents: list[models.PreparedDocument]) -> list[models.Chunk]:
        chunks: typing.Final[list[models.Chunk]] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def split_document(self, document: models.PreparedDocument) -> list[models.Chunk]:
        sections: typing.Final = self._split_into_sections(document)
        chunks: typing.Final[list[models.Chunk]] = []

        for section_name, text in sections:
            if self.config.strategy == "sentence":
                section_chunks = self._split_by_sentences(
                    text=text,
                    document=document,
                    section=section_name,
                )
            elif self.config.strategy == "paragraph":
                section_chunks = self._split_by_paragraphs(
                    text=text,
                    document=document,
                    section=section_name,
                )
            elif self.config.strategy in {"token"}:
                section_chunks = self._split_by_tokens(
                    text=text,
                    document=document,
                    section=section_name,
                )
            else:
                raise ValueError(
                    f"Unsupported chunking strategy: {self.config.strategy!r}. "
                    "Supported strategies: sentence, paragraph, token."
                )

            chunks.extend(section_chunks)

        return self._reindex_chunks(chunks)

    def _split_into_sections(
        self,
        document: models.PreparedDocument,
    ) -> list[tuple[str | None, str]]:
        text: typing.Final = document.text.strip()

        if not text:
            return []

        configured_sections: typing.Final = document.metadata.section

        if not configured_sections:
            return [(None, text)]

        lines: typing.Final = [line.strip() for line in text.splitlines() if line.strip()]

        sections: typing.Final[list[tuple[str | None, str]]] = []
        current_section: str | None = None
        current_lines: list[str] = []

        configured_section_set: typing.Final = set(configured_sections)

        for line in lines:
            if line in configured_section_set:
                if current_lines:
                    sections.append(
                        (
                            current_section,
                            "\n".join(current_lines),
                        )
                    )

                current_section = line
                current_lines = []
                continue

            current_lines.append(line)

        if current_lines:
            sections.append(
                (
                    current_section,
                    "\n".join(current_lines),
                )
            )

        return sections

    def _split_by_sentences(
        self,
        text: str,
        document: models.PreparedDocument,
        section: str | None,
    ) -> list[models.Chunk]:
        sentences: typing.Final = self._split_sentences(text)

        return self._build_chunks_from_units(
            units=sentences,
            document=document,
            section=section,
        )

    def _split_by_paragraphs(
        self,
        text: str,
        document: models.PreparedDocument,
        section: str | None,
    ) -> list[models.Chunk]:
        paragraphs: typing.Final = self._split_paragraphs(text)

        return self._build_chunks_from_units(
            units=paragraphs,
            document=document,
            section=section,
        )

    def _split_by_tokens(
        self,
        text: str,
        document: models.PreparedDocument,
        section: str | None,
    ) -> list[models.Chunk]:
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        tokens: typing.Final = self.config.tokenizer.encode(text)

        if not tokens:
            return []

        chunks: typing.Final[list[models.Chunk]] = []

        start = 0
        tokens_count: typing.Final = len(tokens)

        while start < tokens_count:
            end = min(
                start + self.config.chunk_size,
                tokens_count,
            )

            chunk_tokens = tokens[start:end]
            chunk_text = self.config.tokenizer.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(self._create_chunk(text=chunk_text, document=document, section=section))

            if end >= tokens_count:
                break

            next_start = end - self.config.chunk_overlap

            # Защита от бесконечного цикла при некорректном overlap.
            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _build_chunks_from_units(
        self,
        units: list[str],
        document: models.PreparedDocument,
        section: str | None,
    ) -> list[models.Chunk]:
        chunks: typing.Final[list[models.Chunk]] = []

        current_units: list[str] = []
        current_tokens = 0

        for unit in units:
            unit_tokens = self._count_tokens(unit)

            # Если отдельный semantic unit превышает chunk_size,
            # сначала сохраняем уже накопленный chunk.
            if unit_tokens > self.config.chunk_size:
                if current_units:
                    chunks.append(
                        self._create_chunk(
                            text=" ".join(current_units),
                            document=document,
                            section=section,
                        )
                    )

                    current_units = []
                    current_tokens = 0

                # Длинный unit нельзя сохранить целиком,
                # поэтому используем token-aware fallback.
                chunks.extend(
                    self._split_long_unit(
                        unit=unit,
                        document=document,
                        section=section,
                    )
                )

                continue

            # Добавление unit превысит token budget.
            if current_units and current_tokens + unit_tokens > self.config.chunk_size:
                chunks.append(
                    self._create_chunk(
                        text=" ".join(current_units),
                        document=document,
                        section=section,
                    )
                )

                current_units, current_tokens = self._build_overlap(
                    units=current_units,
                )

            current_units.append(unit)
            current_tokens += unit_tokens

        if current_units:
            chunks.append(
                self._create_chunk(
                    text=" ".join(current_units),
                    document=document,
                    section=section,
                )
            )

        return chunks

    def _build_overlap(
        self,
        units: list[str],
    ) -> tuple[list[str], int]:
        overlap_units: typing.Final[list[str]] = []
        overlap_tokens = 0

        for unit in reversed(units):
            unit_tokens = self._count_tokens(unit)

            if overlap_units and overlap_tokens + unit_tokens > self.config.chunk_overlap:
                break

            if not overlap_units and unit_tokens > self.config.chunk_overlap:
                break

            overlap_units.insert(0, unit)
            overlap_tokens += unit_tokens

        return overlap_units, overlap_tokens

    def _split_long_unit(
        self,
        unit: str,
        document: models.PreparedDocument,
        section: str | None,
    ) -> list[models.Chunk]:
        words: typing.Final[list[str]] = unit.split()

        if not words:
            return []

        chunks: typing.Final[list[models.Chunk]] = []

        start = 0
        words_count: typing.Final = len(words)

        while start < words_count:
            current_words: list[str] = []

            end = start

            while end < words_count:
                word = words[end]

                # Считаем токены реального текста, а не предполагаем,
                # что количество токенов слова известно заранее.
                candidate_words = current_words + [word]
                candidate_text = " ".join(candidate_words)
                candidate_tokens = self._count_tokens(candidate_text)

                if current_words and candidate_tokens > self.config.chunk_size:
                    break

                # Даже одно очень длинное слово/токенизируемая
                # последовательность не должна приводить к бесконечному циклу.
                if not current_words and candidate_tokens > self.config.chunk_size:
                    current_words.append(word)
                    end += 1
                    break

                current_words.append(word)
                end += 1

            chunk_text = " ".join(current_words).strip()

            if chunk_text:
                chunks.append(
                    self._create_chunk(
                        text=chunk_text,
                        document=document,
                        section=section,
                    )
                )

            if end >= words_count:
                break

            # Формируем overlap только из целых слов.
            overlap_words: list[str] = []

            for word in reversed(current_words):
                candidate_words = [word] + overlap_words
                candidate_text = " ".join(candidate_words)
                candidate_tokens = self._count_tokens(candidate_text)

                if overlap_words and candidate_tokens > self.config.chunk_overlap:
                    break

                if not overlap_words and candidate_tokens > self.config.chunk_overlap:
                    break

                overlap_words.insert(0, word)

            # Защита от бесконечного цикла.
            next_start = end - len(overlap_words)

            if next_start <= start:
                next_start = end

            start = next_start

        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        text = re.sub(r"\s+", " ", text).strip()

        if not text:
            return []

        sentences: typing.Final = self._SENTENCE_PATTERN.split(text)

        return [sentence.strip() for sentence in sentences if sentence.strip()]

    def _split_paragraphs(self, text: str) -> list[str]:
        paragraphs: typing.Final = re.split(r"\n\s*\n+", text)

        return [re.sub(r"\s+", " ", paragraph).strip() for paragraph in paragraphs if paragraph.strip()]

    def _create_chunk(
        self,
        text: str,
        document: models.PreparedDocument,
        section: str | None,
    ) -> models.Chunk:
        text = text.strip()

        token_count: typing.Final = self._count_tokens(text)
        text_hash: typing.Final = hashlib.sha256(text.encode("utf-8")).hexdigest()

        # Временная позиция. Реальная позиция назначается
        # в _reindex_chunks().
        position: typing.Final = 0

        chunk_id: typing.Final = hashlib.sha256(
            f"{document.metadata.document_id}:{position}:{text_hash}".encode()
        ).hexdigest()

        metadata: typing.Final = models.ChunkMetadata(
            document_id=document.metadata.document_id,
            position=position,
            chunk_token_count=token_count,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
            chunking_strategy=self.config.strategy,
            source=document.metadata.source,
            section=section,
            text_hash=text_hash,
        )

        return models.Chunk(
            id=chunk_id,
            text=text,
            metadata=metadata,
        )

    def _reindex_chunks(
        self,
        chunks: list[models.Chunk],
    ) -> list[models.Chunk]:
        result: typing.Final[list[models.Chunk]] = []

        for position, chunk in enumerate(chunks):
            metadata = chunk.metadata.model_copy(update={"position": position})

            chunk_id = hashlib.sha256((f"{metadata.document_id}:{position}:{metadata.text_hash}").encode()).hexdigest()

            result.append(
                chunk.model_copy(
                    update={
                        "id": chunk_id,
                        "metadata": metadata,
                    }
                )
            )

        return result

    def _count_tokens(self, text: str) -> int:
        return len(self.config.tokenizer.encode(text))
