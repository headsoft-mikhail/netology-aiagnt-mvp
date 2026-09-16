import logging
import re
import typing

from ai_agent.rag.pipelines import models

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class TextCleaner:
    @staticmethod
    def clean(document: models.ParsedDocument) -> models.ParsedDocument:
        LOGGER_OBJ.info(f"Cleaning {document.source}...")

        text = document.text

        # Normalize line endings
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # Remove control characters except newline and tab
        text = re.sub(r"[^\S\n\t]+", " ", text)

        # Remove whitespace at the beginning and end of each line
        lines = [one_line.strip() for one_line in text.split("\n")]

        # Remove empty lines
        lines = [one_line for one_line in lines if one_line]

        # Join consecutive text fragments with a single newline
        text = "\n".join(lines)

        # Remove leading/trailing whitespace from the whole document
        text = text.strip()

        return document.model_copy(update={"text": text})
