import dataclasses
import json
import typing

from ai_agent.rag.config import PathsConfig
from ai_agent.rag.pipelines.vector_store.stages.search import VectorSearchResult


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorStoreExporter:
    config: PathsConfig

    def export(self, search_results: list[VectorSearchResult]) -> None:
        self.config.vector_store.mkdir(parents=True, exist_ok=True)

        output_path: typing.Final = self.config.search_results_json

        data: typing.Final = [
            dataclasses.asdict(one_search_result) | {"is_relevant": one_search_result.is_relevant}
            for one_search_result in search_results
        ]
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )
