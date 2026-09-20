import dataclasses
import logging
import typing

import tiktoken
from qdrant_client.conversions.common_types import ScoredPoint

LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class ContextBuilder:
    min_score: float
    max_context_tokens: int
    tokenizer_model: str

    def select_points(self, retrieval_results: list[ScoredPoint]) -> list[ScoredPoint]:
        selected_points: typing.Final[list[ScoredPoint]] = []
        used_tokens = 0

        for point in retrieval_results:
            if point.score < self.min_score:
                continue

            rendered_chunk = self._render_chunk(point, len(selected_points) + 1)
            chunk_tokens = self._count_tokens(rendered_chunk)
            if used_tokens + chunk_tokens > self.max_context_tokens:
                break

            selected_points.append(point)
            used_tokens += chunk_tokens

        return selected_points

    def build_context(self, retrieval_results: list[ScoredPoint]) -> str:
        chunks: typing.Final[list[str]] = []
        used_chunks: typing.Final[list[str]] = []

        for point in self.select_points(retrieval_results):
            payload = point.payload or {}
            chunk_id = str(payload.get("chunk_id", ""))
            source = str(payload.get("source", ""))
            chunks.append(self._render_chunk(point, len(chunks) + 1))
            used_chunks.append(f"{chunk_id}\tScore: {point.score:.3f}\tSource: {source}")

        LOGGER_OBJ.info(
            "Chunks used for context: %s",
            "\n".join(used_chunks) if used_chunks else "none",
        )
        return "\n\n".join(chunks)

    def _count_tokens(self, text: str) -> int:
        tokenizer: typing.Final = tiktoken.encoding_for_model(self.tokenizer_model)
        return len(tokenizer.encode(text))

    @staticmethod
    def _render_chunk(point: ScoredPoint, position: int) -> str:
        payload: typing.Final = point.payload or {}
        return "\n".join(
            [
                f"[Фрагмент {position}]",
                f"Источник: {payload.get('source', '')}",
                f"Текст: {payload.get('text', '')}",
            ]
        )
