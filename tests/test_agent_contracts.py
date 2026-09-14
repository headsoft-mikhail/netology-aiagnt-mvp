import typing

import pydantic
import pytest

from ai_agent import contracts, state
from ai_agent.catalog import models


def test_agent_request_strips_boundary_values() -> None:
    request: typing.Final = contracts.AgentRequest(
        user_id="  user-1  ",
        query="  Нужен роутер  ",
    )

    assert request.user_id == "user-1"
    assert request.query == "Нужен роутер"


def test_agent_request_rejects_empty_query() -> None:
    with pytest.raises(pydantic.ValidationError):
        contracts.AgentRequest(user_id="user-1", query="   ")


def test_agent_state_uses_typed_catalog_filters() -> None:
    filters: typing.Final = models.ProductSearchInput(
        filters=models.RouterFilters(
            category=models.ProductCategory.ROUTER,
            min_wifi_generation=6,
        )
    )
    agent_state: typing.Final = state.AgentState(
        request=contracts.AgentRequest(user_id="user-1", query="Нужен Wi-Fi 6 роутер"),
        selected_actions=[contracts.AgentAction.SEARCH_PRODUCTS],
        catalog_filters=filters,
    )

    assert agent_state.catalog_filters is not None
    assert isinstance(agent_state.catalog_filters.filters, models.RouterFilters)
    assert agent_state.status is contracts.AgentRunStatus.PENDING
