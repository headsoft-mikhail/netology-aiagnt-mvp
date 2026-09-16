import dataclasses
import json
import logging
import typing

from ai_agent.rag.pipelines.config import PathsConfig

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class DatasetExporter:
    config: PathsConfig

    def export(self, documents: list[dict]) -> None:
        self.config.prepared.mkdir(parents=True, exist_ok=True)

        self._export_json(documents)
        self._export_jsonl(documents)

        LOGGER_OBJ.info(
            "Exported %d documents to %s",
            len(documents),
            self.config.prepared,
        )

    def _export_json(self, documents: list[dict]) -> None:
        with self.config.prepared_json.open("w", encoding="utf-8") as file:
            json.dump(documents, file, ensure_ascii=False, indent=2)

    def _export_jsonl(self, documents: list[dict]) -> None:
        with self.config.prepared_jsonl.open("w", encoding="utf-8") as file:
            for document in documents:
                file.write(json.dumps(document, ensure_ascii=False) + "\n")
