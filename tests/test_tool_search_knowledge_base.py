import typing

import qdrant_client.models

from ai_agent import contracts
from ai_agent.rag.retrieval import context, models
from ai_agent.tools import search_knowledge_base

QUERY: typing.Final = "Как выбрать роутер?"
SOURCE: typing.Final = "router_selection.txt"


class FakeRetrievalClient:
    def __init__(self, points: list[qdrant_client.models.ScoredPoint]) -> None:
        self.points = points

    def top_k(self, query: str) -> list[qdrant_client.models.ScoredPoint]:
        assert query == QUERY
        return self.points


class FailingRetrievalClient:
    def top_k(self, query: str) -> list[qdrant_client.models.ScoredPoint]:
        raise RuntimeError(query)


def create_point(score: float) -> qdrant_client.models.ScoredPoint:
    return qdrant_client.models.ScoredPoint(
        id=1,
        version=1,
        score=score,
        payload={
            "chunk_id": "chunk-1",
            "text": "Для гигабитного тарифа нужен гигабитный WAN-порт.",
            "source": SOURCE,
        },
    )


def test_search_knowledge_base_returns_context_and_sources() -> None:
    result: typing.Final = search_knowledge_base.search_knowledge_base(
        models.KnowledgeSearchInput(query=QUERY),
        FakeRetrievalClient([create_point(0.9)]),
        context.ContextBuilder(min_score=0.8),
    )

    assert result.status is contracts.ToolStatus.OK
    assert result.sources == [SOURCE]
    assert result.fragments[0].score == 0.9
    assert "гигабитный WAN-порт" in result.context


def test_search_knowledge_base_returns_no_results_below_threshold() -> None:
    result: typing.Final = search_knowledge_base.search_knowledge_base(
        models.KnowledgeSearchInput(query=QUERY),
        FakeRetrievalClient([create_point(0.79)]),
        context.ContextBuilder(min_score=0.8),
    )

    assert result.status is contracts.ToolStatus.NO_RESULTS
    assert result.error is not None
    assert result.error.message == search_knowledge_base.NO_KNOWLEDGE_RESULTS_MESSAGE


def test_search_knowledge_base_returns_safe_retrieval_error() -> None:
    result: typing.Final = search_knowledge_base.search_knowledge_base(
        models.KnowledgeSearchInput(query=QUERY),
        FailingRetrievalClient(),
        context.ContextBuilder(min_score=0.8),
    )

    assert result.status is contracts.ToolStatus.ERROR
    assert result.error is not None
    assert result.error.code is contracts.ErrorCode.KNOWLEDGE_BASE_UNAVAILABLE
    assert result.error.message == search_knowledge_base.KNOWLEDGE_BASE_UNAVAILABLE_MESSAGE
