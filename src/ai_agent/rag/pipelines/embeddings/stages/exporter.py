import dataclasses
import json
import logging
import typing

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import PathsConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingExporter:
    config: PathsConfig

    def export(self, embeddings: list[models.EmbeddedChunk]) -> None:
        self.config.embeddings.mkdir(parents=True, exist_ok=True)

        self._export_json(embeddings)
        self._export_jsonl(embeddings)

        LOGGER_OBJ.info(
            "Exported %d embeddings to %s",
            len(embeddings),
            self.config.embeddings,
        )

    def _export_json(self, embeddings: list[models.EmbeddedChunk]) -> None:
        data: typing.Final = [one_embedding.model_dump(mode="json") for one_embedding in embeddings]

        with self.config.embeddings_json.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

    def _export_jsonl(self, embeddings: list[models.EmbeddedChunk]) -> None:
        with self.config.embeddings_jsonl.open("w", encoding="utf-8") as file:
            for one_embedding in embeddings:
                json.dump(
                    one_embedding.model_dump(mode="json"),
                    file,
                    ensure_ascii=False,
                )
                file.write("\n")
