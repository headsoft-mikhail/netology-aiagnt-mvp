import datetime as dt
import logging
import typing

from ai_agent.rag import manifest
from ai_agent.rag.config import PipelineConfig
from ai_agent.rag.pipelines.chunk.stages.exporter import ChunkExporter
from ai_agent.rag.pipelines.chunk.stages.loader import load_documents
from ai_agent.rag.pipelines.chunk.stages.splitter import ChunkSplitter
from ai_agent.rag.pipelines.chunk.stages.validator import ChunkValidator

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class RAGChunkPipeline:
    def __init__(self, config: PipelineConfig):
        self.config: PipelineConfig = config

        self.splitter = ChunkSplitter(config=self.config.chunking)
        self.exporter = ChunkExporter(config=self.config.paths)
        self.validator = ChunkValidator(config=self.config.chunking)
        self.manifest_manager = manifest.ManifestManager(config=self.config)

    def run(self) -> None:
        LOGGER_OBJ.info("Start chunking pipeline...")
        started_at: typing.Final = dt.datetime.now(tz=dt.UTC).isoformat()

        documents: typing.Final = load_documents(input_path=self.config.paths.prepared_jsonl)
        LOGGER_OBJ.info("Loading - DONE!\n-------------")

        chunks: typing.Final = self.splitter.split_documents(documents)
        LOGGER_OBJ.info(f"Splitting - DONE! {len(chunks)} chunks created.\n-------------")

        self.exporter.export(chunks)
        LOGGER_OBJ.info("Exporting - DONE!\n-------------")

        validation_metrics: typing.Final = self.validator.validate(chunks=chunks, documents=documents)
        if not validation_metrics.overall_validity:
            LOGGER_OBJ.error(f"Chunk validation failed: {validation_metrics}")
        LOGGER_OBJ.info("Validation - DONE!\n-------------")

        self.manifest_manager.create_stage_manifest(
            stage=manifest.StagesEnum.CHUNK,
            started_at=started_at,
            prepared_documents=documents,
            chunks=chunks,
            validation_metrics=validation_metrics,
        )

        LOGGER_OBJ.info("DONE!")
