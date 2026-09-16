import typing
from pathlib import Path

import pydantic
import tiktoken
import yaml


class PathsConfig(pydantic.BaseModel):
    config: Path | None = None
    input: Path
    prepared: Path
    chunks: Path
    embeddings: Path
    vector_store: Path

    @property
    def prepared_jsonl(self) -> Path:
        return Path(self.prepared, "dataset.jsonl")

    @property
    def prepared_json(self) -> Path:
        return Path(self.prepared, "dataset.json")

    @property
    def chunks_json(self) -> Path:
        return Path(self.chunks, "chunks.json")

    @property
    def chunks_jsonl(self) -> Path:
        return Path(self.chunks, "chunks.jsonl")

    @property
    def embeddings_json(self) -> Path:
        return Path(self.embeddings, "embeddings.json")

    @property
    def embeddings_jsonl(self) -> Path:
        return Path(self.embeddings, "embeddings.jsonl")

    @property
    def search_results_json(self) -> Path:
        return Path(self.vector_store, "search_results.json")


class ParsingConfig(pydantic.BaseModel):
    supported_formats: list[str]


class CleaningConfig(pydantic.BaseModel):
    remove_empty_documents: bool = True


class NormalizationConfig(pydantic.BaseModel):
    unicode_form: typing.Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"


class DeduplicationConfig(pydantic.BaseModel):
    exact: bool = True
    near_duplicate: bool = True
    similarity_threshold: float
    permutations_number: int


class ChunkingConfig(pydantic.BaseModel):
    strategy: typing.Literal["sentence", "paragraph", "token"]
    chunk_size: int = pydantic.Field(gt=0)
    chunk_overlap: int = pydantic.Field(ge=0)
    tokenizer_model: str

    @pydantic.model_validator(mode="after")
    def validate_overlap(self) -> typing.Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        return self

    @property
    def tokenizer(self):
        return tiktoken.encoding_for_model(self.tokenizer_model)


class EmbeddingConfig(pydantic.BaseModel):
    model: str = "intfloat/multilingual-e5-base"


class VectorStoreConfig(pydantic.BaseModel):
    collection_name: str
    vectors_dimensions: int
    store_type: str
    distance: typing.Literal["cosine", "dot", "euclid", "manhattan"]
    upload_batch_size: int = pydantic.Field(gt=0)
    search_top_k: int = pydantic.Field(gt=0)
    search_test_queries: int = pydantic.Field(gt=0)
    recreate_collection: bool = True


class PipelineConfig(pydantic.BaseModel):
    paths: PathsConfig
    parsing: ParsingConfig
    cleaning: CleaningConfig
    normalization: NormalizationConfig
    deduplication: DeduplicationConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig


def load_pipeline_config() -> PipelineConfig:
    config_path: typing.Final = Path(__file__).with_name("pipeline.yaml")
    with config_path.open("r", encoding="utf-8") as file:
        data: typing.Final = yaml.safe_load(file)

    pipeline_config: typing.Final = PipelineConfig.model_validate(data)
    pipeline_config.paths.config = config_path

    return pipeline_config
