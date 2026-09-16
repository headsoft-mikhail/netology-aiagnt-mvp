import typing
from pathlib import Path

import pydantic
import yaml


class RetrievalConfig(pydantic.BaseModel):
    embedding_model: str
    vector_store_path: Path
    collection_name: str
    search_top_k: int = pydantic.Field(gt=0)
    min_score: float = pydantic.Field(ge=0, le=1)
    excluded_documents: list[str] = pydantic.Field(default_factory=list)


def load_retrieval_config() -> RetrievalConfig:
    config_path: typing.Final = Path(__file__).with_name("retrieval.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        data: typing.Final = yaml.safe_load(file)

    return RetrievalConfig.model_validate(data)
