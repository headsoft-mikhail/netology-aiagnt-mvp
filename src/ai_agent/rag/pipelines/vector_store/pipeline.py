import datetime as dt
import logging
import typing

from qdrant_client import QdrantClient

from ai_agent.rag import helpers, manifest
from ai_agent.rag.pipelines.config import PipelineConfig
from ai_agent.rag.pipelines.vector_store.stages.exporter import VectorStoreExporter
from ai_agent.rag.pipelines.vector_store.stages.loader import load_embeddings
from ai_agent.rag.pipelines.vector_store.stages.search import VectorStoreSearcher
from ai_agent.rag.pipelines.vector_store.stages.store import QdrantStore
from ai_agent.rag.pipelines.vector_store.stages.validator import VectorStoreValidator

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class RAGVectorStorePipeline:
    def __init__(self, config: PipelineConfig):
        self.config: PipelineConfig = config

        helpers.configure_local_sqlite_thread_check()
        store_client: typing.Final = QdrantClient(path=str(self.config.paths.vector_store))
        self.store = QdrantStore(config=self.config.vector_store, client=store_client)
        self.validator = VectorStoreValidator(config=self.config.vector_store)
        self.searcher: typing.Final = VectorStoreSearcher(config=self.config.vector_store, client=store_client)
        self.manifest_manager = manifest.ManifestManager(config=self.config)
        self.exporter: typing.Final = VectorStoreExporter(config=self.config.paths)

    def run(self) -> None:
        LOGGER_OBJ.info("Start vector store pipeline...")
        started_at: typing.Final = dt.datetime.now(tz=dt.UTC).isoformat()

        embedded_chunks: typing.Final = load_embeddings(input_path=self.config.paths.embeddings_jsonl)
        LOGGER_OBJ.info("Loading - DONE!\n-------------")

        try:
            self.store.create_collection()
            LOGGER_OBJ.info("Collection creation - DONE!\n-------------")

            uploaded: typing.Final = self.store.upload(embedded_chunks)
            LOGGER_OBJ.info(f"Upload - DONE! Uploaded {uploaded} embeddings.\n-------------")

            validation_metrics: typing.Final = self.validator.validate(
                embedded_chunks=embedded_chunks,
                stored_points=self.store.get_all_points(),
            )
            LOGGER_OBJ.info("Validation - DONE!\n-------------")

            search_results: typing.Final = self.searcher.search_batch(embedded_chunks=embedded_chunks)
            LOGGER_OBJ.info("Search - DONE!\n-------------")
        finally:
            self.store.close()

        self.exporter.export(search_results)
        LOGGER_OBJ.info("Exporting - DONE!\n-------------")

        self.manifest_manager.create_stage_manifest(
            manifest.StagesEnum.VECTOR_STORE,
            started_at=started_at,
            embedded_chunks=embedded_chunks,
            validation_metrics=validation_metrics,
        )

        LOGGER_OBJ.info("Manifest - DONE!\n-------------")

        LOGGER_OBJ.info("DONE!")
