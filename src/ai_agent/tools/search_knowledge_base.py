import typing

import pydantic
from qdrant_client.conversions.common_types import ScoredPoint

from ai_agent import contracts
from ai_agent.rag.context import ContextBuilder

NO_KNOWLEDGE_RESULTS_MESSAGE: typing.Final = "В базе знаний не найдено релевантных материалов."
KNOWLEDGE_BASE_UNAVAILABLE_MESSAGE: typing.Final = "Не удалось выполнить поиск по базе знаний."
SEARCH_KNOWLEDGE_BASE_TOOL_NAME: typing.Final = "search_knowledge_base"


class RetrievalClientProtocol(typing.Protocol):
    def top_k(self, query: str) -> list[ScoredPoint]: ...


class KnowledgeSearchInput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    query: str = pydantic.Field(min_length=1)


def search_knowledge_base(
    args: KnowledgeSearchInput,
    retrieval_client: RetrievalClientProtocol,
    context_builder: ContextBuilder,
) -> contracts.KnowledgeSearchResult:
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
    except Exception:  # noqa: BLE001
        return contracts.KnowledgeSearchResult(
            status=contracts.ToolStatus.ERROR,
            error=contracts.ToolError(
                code=contracts.ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE,
                message=KNOWLEDGE_BASE_UNAVAILABLE_MESSAGE,
            ),
        )

    if not fragments:
        return contracts.KnowledgeSearchResult(
            status=contracts.ToolStatus.NO_RESULTS,
            error=contracts.ToolError(
                code=contracts.ErrorCode.NO_RESULTS,
                message=NO_KNOWLEDGE_RESULTS_MESSAGE,
            ),
        )

    return contracts.KnowledgeSearchResult(
        status=contracts.ToolStatus.OK,
        fragments=fragments,
        context=context,
        sources=list(dict.fromkeys(fragment.source for fragment in fragments)),
    )


def _to_fragment(point: ScoredPoint) -> contracts.KnowledgeFragment:
    payload: typing.Final = point.payload or {}
    return contracts.KnowledgeFragment(
        chunk_id=str(payload.get("chunk_id", "")),
        text=str(payload.get("text", "")),
        source=str(payload.get("source", "")),
        score=point.score,
    )
