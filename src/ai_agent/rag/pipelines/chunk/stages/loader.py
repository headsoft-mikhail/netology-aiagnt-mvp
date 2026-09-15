import json
import logging
import typing
from pathlib import Path

from ai_agent.rag import models

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


def load_documents(input_path: Path) -> list[models.PreparedDocument]:
    LOGGER_OBJ.info(
        f"Start loading {input_path}...",
    )

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    documents: typing.Final[list[models.PreparedDocument]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for one_line_number, one_line in enumerate(file, start=1):
            if not one_line.strip():
                continue

            try:
                document = models.PreparedDocument.model_validate(json.loads(one_line))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"Failed to load document {input_path} at line {one_line_number}") from exc

            documents.append(document)

    LOGGER_OBJ.info(
        f"{len(documents)} documents loaded.",
    )

    return documents
