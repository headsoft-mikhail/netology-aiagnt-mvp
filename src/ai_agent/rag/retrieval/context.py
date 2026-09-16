import dataclasses
import logging
import typing

from qdrant_client.conversions.common_types import ScoredPoint

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ContextBuilder:
    min_score: float

    def build_context(self, retrieval_results: list[ScoredPoint]) -> str:
        chunks: typing.Final[list[str]] = []
        used_chunks: typing.Final[list[str]] = []

        for point in retrieval_results:
            if point.score < self.min_score:
                continue

            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            source = str(payload.get("source", ""))
            text = str(payload.get("text", ""))
            chunks.append(
                "\n".join(
                    [
                        f"[Фрагмент {len(chunks) + 1}]",
                        f"Источник: {source}",
                        f"Текст: {text}",
                    ]
                )
            )
            used_chunks.append(f"{chunk_id}\tScore: {point.score:.3f}\tSource: {source}")

        LOGGER_OBJ.info(
            "Chunks used for context: %s",
            "\n".join(used_chunks) if used_chunks else "none",
        )
        return "\n\n".join(chunks)
