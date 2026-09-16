import dataclasses
import json
import logging
import typing

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import PathsConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ChunkExporter:
    config: PathsConfig

    def export(self, chunks: list[models.Chunk]) -> None:
        self.config.chunks.mkdir(parents=True, exist_ok=True)

        self._export_json(chunks)
        self._export_jsonl(chunks)

        LOGGER_OBJ.info(
            "Exported %d chunks to %s",
            len(chunks),
            self.config.chunks,
        )

    def _export_json(self, chunks: list[models.Chunk]) -> None:
        data: typing.Final = [chunk.model_dump(mode="json") for chunk in chunks]

        with self.config.chunks_json.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _export_jsonl(self, chunks: list[models.Chunk]) -> None:
        with self.config.chunks_jsonl.open("w", encoding="utf-8") as file:
            for chunk in chunks:
                json.dump(
                    chunk.model_dump(mode="json"),
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")
