import typing
from pathlib import Path

import pydantic
import pydantic_settings
import tiktoken


class PathsConfig(pydantic.BaseModel):
    input: Path = Path("src/ai_agent/rag/knowledge_base/raw")
    prepared: Path = Path("src/ai_agent/rag/knowledge_base/prepared")
    chunks: Path = Path("src/ai_agent/rag/knowledge_base/chunks")
    embeddings: Path = Path("src/ai_agent/rag/knowledge_base/embeddings")
    vector_store: Path = Path("src/ai_agent/rag/knowledge_base/vector_store")

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
    supported_formats: list[str] = pydantic.Field(default_factory=lambda: ["txt", "json", "html"])


class CleaningConfig(pydantic.BaseModel):
    remove_empty_documents: bool = True


class NormalizationConfig(pydantic.BaseModel):
    unicode_form: typing.Literal["NFC", "NFD", "NFKC", "NFKD"] = "NFC"


class DeduplicationConfig(pydantic.BaseModel):
    exact: bool = True
    near_duplicate: bool = True
    similarity_threshold: float = pydantic.Field(default=0.65, ge=0, le=1)
    permutations_number: int = pydantic.Field(default=128, gt=0)


class ChunkingConfig(pydantic.BaseModel):
    strategy: typing.Literal["sentence", "paragraph", "token"] = "sentence"
    chunk_size: int = pydantic.Field(default=220, gt=0)
    chunk_overlap: int = pydantic.Field(default=50, ge=0)
    tokenizer_model: str = "text-embedding-3-small"

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
    collection_name: str = "store_knowledge"
    vectors_dimensions: int = pydantic.Field(default=768, gt=0)
    store_type: str = "qdrant"
    distance: typing.Literal["cosine", "dot", "euclid", "manhattan"] = "cosine"
    upload_batch_size: int = pydantic.Field(default=100, gt=0)
    search_top_k: int = pydantic.Field(default=5, gt=0)
    search_test_queries: int = pydantic.Field(default=3, gt=0)
    recreate_collection: bool = True


class PipelineConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="RAG_PIPELINE_",
        env_nested_delimiter="__",
    )

    paths: PathsConfig = pydantic.Field(default_factory=PathsConfig)
    parsing: ParsingConfig = pydantic.Field(default_factory=ParsingConfig)
    cleaning: CleaningConfig = pydantic.Field(default_factory=CleaningConfig)
    normalization: NormalizationConfig = pydantic.Field(default_factory=NormalizationConfig)
    deduplication: DeduplicationConfig = pydantic.Field(default_factory=DeduplicationConfig)
    chunking: ChunkingConfig = pydantic.Field(default_factory=ChunkingConfig)
    embedding: EmbeddingConfig = pydantic.Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = pydantic.Field(default_factory=VectorStoreConfig)


rag_pipeline_config: typing.Final = PipelineConfig()
