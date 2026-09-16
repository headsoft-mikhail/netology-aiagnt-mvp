import dataclasses
import typing

from sentence_transformers import SentenceTransformer

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import EmbeddingConfig


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingModel:
    config: EmbeddingConfig
    transformer: SentenceTransformer

    def encode(self, texts: list[str]) -> list[list[float]]:
        embeddings: typing.Final = self.transformer.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return embeddings.tolist()

    @property
    def dimensions(self) -> int:
        dimensions: typing.Final = self.transformer.get_embedding_dimension()
        if dimensions is None:
            raise ValueError(f"Cannot determine embedding dimensions for model {self.config.model!r}")
        return dimensions


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingStage:
    model: EmbeddingModel

    def run(self, chunks: list[models.Chunk]) -> list[models.EmbeddedChunk]:
        texts: typing.Final = [f"passage: {chunk.text}" for chunk in chunks]
        embeddings: typing.Final = self.model.encode(texts)

        return [
            models.EmbeddedChunk(
                id=chunk.id,
                text=chunk.text,
                embedding=embedding,
                metadata=models.EmbeddedChunkMetadata(
                    **chunk.metadata.model_dump(),
                    embedding_model=self.model.config.model,
                    embedding_dimensions=self.model.dimensions,
                ),
            )
            for chunk, embedding in zip(chunks, embeddings, strict=True)
        ]
