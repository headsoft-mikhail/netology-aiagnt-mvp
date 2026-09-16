import dataclasses
import hashlib
import logging
import re
import typing

from datasketch import MinHash

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import DeduplicationConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class Deduplicator:
    config: DeduplicationConfig

    def deduplicate(
        self,
        documents: list[models.ParsedDocument],
    ) -> models.DeduplicationResult:
        unique_documents: typing.Final[list[models.ParsedDocument]] = documents.copy()

        result = models.DeduplicationResult(
            documents=unique_documents,
            exact_duplicates=0,
            near_duplicates=0,
        )

        if self.config.exact:
            result += self._deduplicate_exact(unique_documents)

        if self.config.near_duplicate:
            result += self._deduplicate_near(unique_documents)

        return result

    def _deduplicate_exact(
        self,
        documents: list[models.ParsedDocument],
    ) -> models.DeduplicationResult:
        unique_documents: typing.Final[list[models.ParsedDocument]] = []
        exact_hashes: typing.Final[set[str]] = set()
        duplicates_count = 0

        for document in documents:
            text_hash = self._text_hash(document.text)

            if text_hash in exact_hashes:
                LOGGER_OBJ.info(
                    "Document %s has exact duplicate",
                    document.source,
                )
                duplicates_count += 1
                continue

            exact_hashes.add(text_hash)
            unique_documents.append(document)

        return models.DeduplicationResult(
            documents=unique_documents,
            exact_duplicates=duplicates_count,
            near_duplicates=0,
        )

    def _deduplicate_near(
        self,
        documents: list[models.ParsedDocument],
    ) -> models.DeduplicationResult:
        unique_documents: typing.Final[list[models.ParsedDocument]] = []
        duplicates_count = 0

        for document in documents:
            if self._is_near_duplicate(document, unique_documents):
                LOGGER_OBJ.info(
                    "Document %s has near duplicate",
                    document.source,
                )
                duplicates_count += 1
                continue

            unique_documents.append(document)

        return models.DeduplicationResult(
            documents=unique_documents,
            exact_duplicates=0,
            near_duplicates=duplicates_count,
        )

    @staticmethod
    def _text_hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _is_near_duplicate(
        self,
        document: models.ParsedDocument,
        existing_documents: list[models.ParsedDocument],
    ) -> bool:
        current: typing.Final = self._create_minhash(document.text)

        for existing_document in existing_documents:
            candidate = self._create_minhash(existing_document.text)
            similarity = current.jaccard(candidate)

            if similarity >= self.config.similarity_threshold:
                return True

        return False

    def _create_minhash(self, text: str) -> MinHash:
        minhash: typing.Final = MinHash(num_perm=self.config.permutations_number)
        words: typing.Final = self._tokenize(text)
        for word in words:
            minhash.update(word.encode("utf-8"))

        return minhash

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return set(re.findall(r"\w+", text.lower(), re.UNICODE))
