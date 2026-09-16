import math

import pydantic

from ai_agent.rag.pipelines.config import VectorStoreConfig


class SourceDocument(pydantic.BaseModel):
    content: str
    source: str
    file_type: str


class ParsedDocument(pydantic.BaseModel):
    text: str
    source: str
    file_type: str
    sections: list[str] = pydantic.Field(default_factory=list)
    character_count: int = 0
    word_count: int = 0
    sentence_count: int = 0


class DeduplicationResult(pydantic.BaseModel):
    documents: list[ParsedDocument]
    exact_duplicates: int = 0
    near_duplicates: int = 0

    def __add__(self, other: DeduplicationResult) -> DeduplicationResult:
        return DeduplicationResult(
            documents=other.documents,
            exact_duplicates=self.exact_duplicates + other.exact_duplicates,
            near_duplicates=self.near_duplicates + other.near_duplicates,
        )


class PreparedDocumentMetadata(pydantic.BaseModel):
    source: str
    file_type: str
    document_id: str
    section: list[str] = pydantic.Field(default_factory=list)
    character_count: int = 0
    word_count: int = 0
    sentence_count: int = 0


class PreparedDocument(pydantic.BaseModel):
    text: str
    metadata: PreparedDocumentMetadata


class ChunkMetadata(pydantic.BaseModel):
    document_id: str
    position: int
    chunk_token_count: int
    chunk_size: int
    chunk_overlap: int
    chunking_strategy: str
    source: str | None = None
    section: str | None = None
    text_hash: str


class Chunk(pydantic.BaseModel):
    id: str
    text: str
    metadata: ChunkMetadata


class EmbeddedChunkMetadata(ChunkMetadata):
    embedding_model: str = pydantic.Field(min_length=1)
    embedding_dimensions: int


class EmbeddedChunk(pydantic.BaseModel):
    id: str = pydantic.Field(min_length=1)
    text: str = pydantic.Field(min_length=1)
    embedding: list[float]
    metadata: EmbeddedChunkMetadata

    def is_valid_for_vector_store(self, vector_store_config: VectorStoreConfig):
        return (
            self.embedding
            and self.metadata
            and self.metadata.embedding_dimensions == vector_store_config.vectors_dimensions
            and all(math.isfinite(value) for value in self.embedding)
        )


class VectorStorePointPayload(EmbeddedChunkMetadata):
    chunk_id: str = pydantic.Field(min_length=1)
    text: str = pydantic.Field(min_length=1)
