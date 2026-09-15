import typing

import pytest

from ai_agent import contracts
from ai_agent.rag import config, context, models, retrieval
from ai_agent.tools import search_knowledge_base

RELEVANT_QUERY: typing.Final = "Какой роутер подойдёт для тарифа 1 Гбит/с?"
IRRELEVANT_QUERY: typing.Final = "Сколько стоит страхование жизни?"


@pytest.fixture(scope="module")
def knowledge_search_dependencies() -> tuple[retrieval.RetrievalClient, context.ContextBuilder]:
    retrieval_config: typing.Final = config.load_retrieval_config()
    return (
        retrieval.RetrievalClient(retrieval_config),
        context.ContextBuilder(min_score=retrieval_config.min_score),
    )


def test_retrieval_finds_router_requirements(
    knowledge_search_dependencies: tuple[retrieval.RetrievalClient, context.ContextBuilder],
) -> None:
    retrieval_client, context_builder = knowledge_search_dependencies

    result: typing.Final = search_knowledge_base.search_knowledge_base(
        models.KnowledgeSearchInput(query=RELEVANT_QUERY),
        retrieval_client,
        context_builder,
    )

    assert result.status is contracts.ToolStatus.OK
    assert "1 Гбит/с" in result.context
    assert any("router_selection.txt" in source for source in result.sources)


def test_retrieval_rejects_irrelevant_insurance_query(
    knowledge_search_dependencies: tuple[retrieval.RetrievalClient, context.ContextBuilder],
) -> None:
    retrieval_client, context_builder = knowledge_search_dependencies

    result: typing.Final = search_knowledge_base.search_knowledge_base(
        models.KnowledgeSearchInput(query=IRRELEVANT_QUERY),
        retrieval_client,
        context_builder,
    )

    assert result.status is contracts.ToolStatus.NO_RESULTS
    assert result.fragments == []
