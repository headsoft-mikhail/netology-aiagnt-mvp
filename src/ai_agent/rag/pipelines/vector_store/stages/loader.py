import json
import logging
import typing
from pathlib import Path

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.models import EmbeddedChunk

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


def load_embeddings(input_path: Path) -> list[models.EmbeddedChunk]:
    LOGGER_OBJ.info(f"Start loading embeddings from {input_path}...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    embeddings: typing.Final[list[EmbeddedChunk]] = []

    with input_path.open("r", encoding="utf-8") as file:
        for one_line_number, one_line in enumerate(file, start=1):
            one_line = one_line.strip()
            if not one_line:
                continue

            try:
                data = json.loads(one_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {one_line_number}: {exc}") from exc

            try:
                embedded_chunk = models.EmbeddedChunk.model_validate(data)
            except Exception as exc:
                raise ValueError(f"Invalid embedding at line {one_line_number}: {exc}") from exc

            embeddings.append(embedded_chunk)

    LOGGER_OBJ.info(f"{len(embeddings)} embeddings loaded.")

    return embeddings
