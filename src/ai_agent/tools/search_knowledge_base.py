import dataclasses
import logging
import typing

import pydantic
from qdrant_client.conversions.common_types import ScoredPoint

from ai_agent import contracts
from ai_agent.rag.retrieval import models
from ai_agent.rag.retrieval.context import ContextBuilder
from ai_agent.tools import protocol

NO_KNOWLEDGE_RESULTS_MESSAGE: typing.Final = "В базе знаний не найдено релевантных материалов."
KNOWLEDGE_BASE_UNAVAILABLE_MESSAGE: typing.Final = "Не удалось выполнить поиск по базе знаний."
SEARCH_KNOWLEDGE_BASE_TOOL_NAME: typing.Final = "search_knowledge_base"
LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


class RetrievalClientProtocol(typing.Protocol):
    def top_k(self, query: str) -> list[ScoredPoint]: ...


def search_knowledge_base(
    args: models.KnowledgeSearchInput,
    retrieval_client: RetrievalClientProtocol,
    context_builder: ContextBuilder,
) -> contracts.KnowledgeBaseSearchResult:
    """Search guides, compatibility notes and store policies in the RAG index.

    Call this Tool for questions that require instructions, selection rules,
    compatibility guidance, troubleshooting or store policies. Do not call it
    solely to obtain live product prices, stock or a filtered product list.
    """
    normalized_query: typing.Final = args.query

    try:
        retrieval_results: typing.Final = retrieval_client.top_k(normalized_query)
        relevant_points: typing.Final = [
            point for point in retrieval_results if point.score >= context_builder.min_score
        ]
        fragments: typing.Final = [_to_fragment(point) for point in relevant_points]
        context: typing.Final = context_builder.build_context(retrieval_results)
    except Exception:
        LOGGER_OBJ.debug("Knowledge base search failed", exc_info=True)
        return contracts.KnowledgeBaseSearchResult(
            status=contracts.ToolStatus.ERROR,
            error=contracts.ToolError(
                code=contracts.ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE,
                message=KNOWLEDGE_BASE_UNAVAILABLE_MESSAGE,
            ),
        )

    if not fragments:
        return contracts.KnowledgeBaseSearchResult(
            status=contracts.ToolStatus.NO_RESULTS,
            error=contracts.ToolError(
                code=contracts.ErrorCode.NO_RESULTS,
                message=NO_KNOWLEDGE_RESULTS_MESSAGE,
            ),
        )

    return contracts.KnowledgeBaseSearchResult(
        status=contracts.ToolStatus.OK,
        fragments=fragments,
        context=context,
        sources=list(dict.fromkeys(fragment.source for fragment in fragments)),
    )


@dataclasses.dataclass(kw_only=True, slots=True)
class KnowledgeBaseSearchTool:
    retrieval_client: RetrievalClientProtocol
    context_builder: ContextBuilder
    name: typing.ClassVar[str] = SEARCH_KNOWLEDGE_BASE_TOOL_NAME
    input_model: typing.ClassVar[type[pydantic.BaseModel]] = models.KnowledgeSearchInput

    def invoke(self, tool_input: pydantic.BaseModel) -> protocol.ToolResult:
        if not isinstance(tool_input, models.KnowledgeSearchInput):
            raise TypeError("KnowledgeBaseSearchTool received an invalid input model.")
        return search_knowledge_base(tool_input, self.retrieval_client, self.context_builder)


def _to_fragment(point: ScoredPoint) -> contracts.KnowledgeFragment:
    payload: typing.Final = point.payload or {}
    return contracts.KnowledgeFragment(
        chunk_id=str(payload.get("chunk_id", "")),
        text=str(payload.get("text", "")),
        source=str(payload.get("source", "")),
        score=point.score,
    )
