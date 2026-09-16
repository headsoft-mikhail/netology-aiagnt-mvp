import logging
import typing
from pathlib import Path

from ai_agent.rag.pipelines import models

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


def load_documents(
    input_dir: Path,
    supported_formats: list[str],
) -> list[models.SourceDocument]:
    LOGGER_OBJ.info(f"Loading documents from {input_dir}...")
    documents: typing.Final = []

    for path in sorted(input_dir.iterdir()):
        if not path.is_file():
            continue

        file_type = path.suffix.lower().lstrip(".")

        if file_type not in supported_formats:
            continue

        content = path.read_text(encoding="utf-8")

        documents.append(
            models.SourceDocument(
                content=content,
                source=str(path),
                file_type=file_type,
            )
        )

    LOGGER_OBJ.info(f"{len(documents)} documents loaded.")
    return documents
