import datetime as dt
import logging
import typing

from sentence_transformers import SentenceTransformer

from ai_agent.rag import manifest
from ai_agent.rag.pipelines.config import PipelineConfig
from ai_agent.rag.pipelines.embeddings.stages import embedding
from ai_agent.rag.pipelines.embeddings.stages.exporter import EmbeddingExporter
from ai_agent.rag.pipelines.embeddings.stages.loader import load_chunks
from ai_agent.rag.pipelines.embeddings.stages.validator import EmbeddingValidator

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class RAGEmbeddingsPipeline:
    def __init__(self, config: PipelineConfig):
        self.config: PipelineConfig = config

        transformer: typing.Final = SentenceTransformer(self.config.embedding.model)
        embedding_model: typing.Final = embedding.EmbeddingModel(
            config=self.config.embedding,
            transformer=transformer,
        )
        self.embedding_stage = embedding.EmbeddingStage(model=embedding_model)
        self.validator = EmbeddingValidator(model=embedding_model)
        self.manifest_manager = manifest.ManifestManager(config=self.config)
        self.exporter = EmbeddingExporter(config=self.config.paths)

    def run(self) -> None:
        LOGGER_OBJ.info("Start embedding pipeline...")
        started_at: typing.Final = dt.datetime.now(tz=dt.UTC).isoformat()

        chunks: typing.Final = load_chunks(input_path=self.config.paths.chunks_jsonl)
        LOGGER_OBJ.info("Loading - DONE!\n-------------")

        embedded_chunks: typing.Final = self.embedding_stage.run(chunks)
        LOGGER_OBJ.info("Embedding - DONE!\n-------------")

        self.exporter.export(embeddings=embedded_chunks)
        LOGGER_OBJ.info("Exporting - DONE!\n-------------")

        validation_metrics: typing.Final = self.validator.validate(chunks=chunks, embeddings=embedded_chunks)
        LOGGER_OBJ.info("Validation - DONE!\n-------------")

        self.manifest_manager.create_stage_manifest(
            stage=manifest.StagesEnum.EMBEDDING,
            started_at=started_at,
            chunks=chunks,
            embedded_chunks=embedded_chunks,
            validation_metrics=validation_metrics,
        )

        LOGGER_OBJ.info("DONE!")
