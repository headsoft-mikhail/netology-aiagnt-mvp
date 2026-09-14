import dataclasses
import logging
import re
import typing
import unicodedata

from ai_agent.rag import models
from ai_agent.rag.config import NormalizationConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class TextNormalizer:
    config: NormalizationConfig

    def normalize(self, document: models.ParsedDocument) -> models.ParsedDocument:
        LOGGER_OBJ.info(f"Normalizing {document.source}...")
        text = unicodedata.normalize(self.config.unicode_form, document.text)

        text = text.replace("\t", " ")
        text = re.sub(r" {2,}", " ", text)

        character_count: typing.Final = len(text)
        word_count: typing.Final = len(re.findall(r"\b\w+\b", text, re.UNICODE))
        sentence_count: typing.Final = len(re.findall(r"[.!?]+(?:\s|$)", text))

        return document.model_copy(
            update={
                "text": text,
                "character_count": character_count,
                "word_count": word_count,
                "sentence_count": sentence_count,
            }
        )
