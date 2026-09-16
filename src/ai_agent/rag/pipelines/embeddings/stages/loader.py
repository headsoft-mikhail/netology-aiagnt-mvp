import json
import logging
import typing
from pathlib import Path

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.models import Chunk

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


def load_chunks(input_path: Path) -> list[models.Chunk]:
    LOGGER_OBJ.info(f"Start loading chunks from {input_path}...")

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    chunks: typing.Final[list[Chunk]] = []

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
                chunk = models.Chunk.model_validate(data)
            except Exception as exc:
                raise ValueError(f"Invalid chunk at line {one_line_number}: {exc}") from exc

            chunks.append(chunk)

    LOGGER_OBJ.info(
        f"{len(chunks)} documents loaded.",
    )

    return chunks
