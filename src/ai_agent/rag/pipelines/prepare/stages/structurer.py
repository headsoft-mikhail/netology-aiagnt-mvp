import hashlib
import typing

from ai_agent.rag.pipelines import models


class DocumentStructurer:
    @staticmethod
    def structure(document: models.ParsedDocument) -> dict:
        document_id: typing.Final = hashlib.sha256(document.source.encode("utf-8")).hexdigest()

        metadata: typing.Final = {
            "source": document.source,
            "file_type": document.file_type,
            "document_id": document_id,
            "section": document.sections,
            "character_count": document.character_count,
            "word_count": document.word_count,
            "sentence_count": document.sentence_count,
        }

        return {
            "text": document.text,
            "metadata": metadata,
        }

    @classmethod
    def structure_many(cls, documents: list[models.ParsedDocument]) -> list[dict]:
        return [cls.structure(one_document) for one_document in documents]
