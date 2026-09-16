from __future__ import annotations

import abc
import dataclasses
import datetime as dt
import enum
import json
import logging
import pathlib
import typing
import uuid
from pathlib import Path

from ai_agent.rag.pipelines import models
from ai_agent.rag.pipelines.config import PipelineConfig

if typing.TYPE_CHECKING:
    from ai_agent.rag.pipelines.chunk.stages.validator import ChunkingValidationMetrics
    from ai_agent.rag.pipelines.embeddings.stages.validator import EmbeddingValidationMetrics
    from ai_agent.rag.pipelines.vector_store.stages.validator import VectorStoreValidationMetrics

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class StagesEnum(enum.Enum):
    PREPARE = "prepare"
    CHUNK = "chunk"
    EMBEDDING = "embedding"
    VECTOR_STORE = "vector_store"

    def previous(self) -> StagesEnum | None:
        if self == StagesEnum.CHUNK:
            return StagesEnum.PREPARE
        elif self == StagesEnum.EMBEDDING:
            return StagesEnum.CHUNK
        elif self == StagesEnum.VECTOR_STORE:
            return StagesEnum.EMBEDDING
        return None


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ManifestCreatorProtocol(abc.ABC):
    config: PipelineConfig

    @abc.abstractmethod
    def create(self, *, started_at: str, **kwargs) -> dict[str, typing.Any]:
        raise NotImplementedError


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class PrepareManifest(ManifestCreatorProtocol):
    def create(self, *, started_at: str, **kwargs) -> dict[str, typing.Any]:
        raw_documents: typing.Final[list[models.SourceDocument]] = kwargs["raw_documents"]
        deduplication_result: typing.Final[models.DeduplicationResult] = kwargs["deduplication_result"]

        manifest: typing.Final = {
            "run_id": str(uuid.uuid4()),
            "started_at": started_at,
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "paths": {
                "config": str(self.config.paths.config),
                "input_dir": str(self.config.paths.input),
                "prepared_path": str(self.config.paths.prepared_jsonl),
            },
            "count": {
                "input_documents": len(raw_documents),
                "output_documents": len(deduplication_result.documents),
            },
            "deduplication": {
                "exact_duplicates": deduplication_result.exact_duplicates,
                "near_duplicates": deduplication_result.near_duplicates,
            },
            "stage_config": {
                "deduplication_similarity_threshold": self.config.deduplication.similarity_threshold,
                "normalization_unicode_form": self.config.normalization.unicode_form,
            },
        }
        LOGGER_OBJ.info("Manifest prepared.")
        return manifest


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ChunkingManifest(ManifestCreatorProtocol):
    def create(self, *, started_at: str, **kwargs) -> dict[str, typing.Any]:
        prepared_documents: typing.Final[list[models.PreparedDocument]] = kwargs["prepared_documents"]
        chunks: typing.Final[list[models.Chunk]] = kwargs["chunks"]
        validation_metrics: typing.Final[ChunkingValidationMetrics] = kwargs["validation_metrics"]

        manifest: typing.Final = {
            "run_id": str(uuid.uuid4()),
            "started_at": started_at,
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "paths": {
                "config": str(self.config.paths.config),
                "prepared_path": str(self.config.paths.prepared_jsonl),
                "chunks_path": str(self.config.paths.chunks_jsonl),
            },
            "count": {
                "input_documents": len(prepared_documents),
                "output_chunks": len(chunks),
            },
            "stage_config": {
                "strategy": self.config.chunking.strategy,
                "chunk_size": self.config.chunking.chunk_size,
                "chunk_overlap": self.config.chunking.chunk_overlap,
            },
            "validation": dataclasses.asdict(validation_metrics),
        }
        LOGGER_OBJ.info("Manifest prepared.")
        return manifest


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class EmbeddingManifest(ManifestCreatorProtocol):
    def create(self, *, started_at: str, **kwargs) -> dict[str, typing.Any]:
        chunks: typing.Final[list[models.Chunk]] = kwargs["chunks"]
        embedded_chunks: typing.Final[list[models.EmbeddedChunk]] = kwargs["embedded_chunks"]
        validation_metrics: typing.Final[EmbeddingValidationMetrics] = kwargs["validation_metrics"]

        manifest: typing.Final = {
            "run_id": str(uuid.uuid4()),
            "started_at": started_at,
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "paths": {
                "config": str(self.config.paths.config),
                "chunks_path": str(self.config.paths.chunks_jsonl),
                "embeddings_path": str(self.config.paths.embeddings_jsonl),
            },
            "count": {
                "input_chunks": len(chunks),
                "output_embeddings": len(embedded_chunks),
            },
            "stage_config": {
                "model": embedded_chunks[0].metadata.embedding_model if embedded_chunks else None,
                "dimensions": embedded_chunks[0].metadata.embedding_dimensions if embedded_chunks else None,
            },
            "validation": dataclasses.asdict(validation_metrics),
        }

        LOGGER_OBJ.info("Manifest prepared.")
        return manifest


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class VectorStoreManifest(ManifestCreatorProtocol):
    def create(self, *, started_at: str, **kwargs) -> dict[str, typing.Any]:
        embedded_chunks: typing.Final[list[models.EmbeddedChunk]] = kwargs["embedded_chunks"]
        validation_metrics: typing.Final[VectorStoreValidationMetrics] = kwargs["validation_metrics"]

        manifest: typing.Final = {
            "run_id": str(uuid.uuid4()),
            "started_at": started_at,
            "created_at": dt.datetime.now(tz=dt.UTC).isoformat(),
            "paths": {
                "config": str(self.config.paths.config),
                "embeddings_path": str(self.config.paths.embeddings_jsonl),
                "search_results": str(self.config.paths.search_results_json),
            },
            "count": {
                "input_embeddings": len(embedded_chunks),
                "stored_points": validation_metrics.stored_points_count,
            },
            "stage_config": {
                "store_type": self.config.vector_store.store_type,
                "collection_name": self.config.vector_store.collection_name,
                "vectors_dimensions": self.config.vector_store.vectors_dimensions,
                "distance": self.config.vector_store.distance,
            },
            "validation": dataclasses.asdict(validation_metrics)
            | {"overall_validity": validation_metrics.overall_validity},
        }

        LOGGER_OBJ.info("Manifest prepared.")
        return manifest


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ManifestManager:
    config: PipelineConfig

    def _select_path(self, stage: StagesEnum) -> Path:
        if stage == StagesEnum.PREPARE:
            return self.config.paths.prepared
        elif stage == StagesEnum.CHUNK:
            return self.config.paths.chunks
        elif stage == StagesEnum.EMBEDDING:
            return self.config.paths.embeddings
        elif stage == StagesEnum.VECTOR_STORE:
            return self.config.paths.vector_store

    def _select_stage_manifest_creator(self, stage: StagesEnum) -> ManifestCreatorProtocol:
        if stage == StagesEnum.PREPARE:
            return PrepareManifest(config=self.config)
        elif stage == StagesEnum.CHUNK:
            return ChunkingManifest(config=self.config)
        elif stage == StagesEnum.EMBEDDING:
            return EmbeddingManifest(config=self.config)
        elif stage == StagesEnum.VECTOR_STORE:
            return VectorStoreManifest(config=self.config)

    def load(self, stage: StagesEnum) -> dict[str, typing.Any]:
        path: typing.Final = self._select_path(stage) / "manifest.json"
        if not path.exists():
            return {}

        with path.open("r", encoding="utf-8") as file:
            return typing.cast(
                dict[str, typing.Any],
                json.load(file),
            )

    def create_stage_manifest(self, stage: StagesEnum, **kwargs):
        LOGGER_OBJ.info("Exporting manifest...")
        previous_stage: typing.Final = stage.previous()
        previous_stage_data: typing.Final = self.load(previous_stage) if previous_stage else {}
        current_stage_data: typing.Final = self._select_stage_manifest_creator(stage).create(**kwargs)
        full_data: typing.Final = previous_stage_data | {stage.name.lower(): current_stage_data}
        full_data["run_id"] = str(uuid.uuid4())
        self._save(manifest=full_data, output_dir=self._select_path(stage))

    def _save(self, manifest: dict[str, typing.Any], output_dir: pathlib.Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        output_path: typing.Final = output_dir / "manifest.json"
        output_path.write_text(
            json.dumps(
                manifest,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        LOGGER_OBJ.info(f"Manifest exported to {output_path}")
