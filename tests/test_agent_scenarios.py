import pathlib
import typing

import pytest

from ai_agent import agent, contracts
from ai_agent.memory import repository as memory_repository

MEMORY_REQUEST: typing.Final = "Какой бюджет я указывал?"
TEST_USER_ID: typing.Final = "test_user"
TEST_USER_FACTS: typing.Final = "Бюджет до 10 000 рублей"


@pytest.fixture
def agent_runner(tmp_path: pathlib.Path) -> agent.MockAgentRunner:
    db: typing.Final = memory_repository.MemoryRepository(database_path=tmp_path / "memory.db")
    db.save_fact(TEST_USER_ID, TEST_USER_FACTS)
    return agent.MockAgentRunner(db)


def test_agent_uses_long_term_memory(agent_runner: agent.MockAgentRunner) -> None:
    result: typing.Final = agent_runner.run(TEST_USER_ID, MEMORY_REQUEST)

    assert TEST_USER_FACTS in result.answer
    assert result.status is contracts.AgentRunStatus.COMPLETED


def test_agent_requests_clarification_for_unspecific_request(
    agent_runner: agent.MockAgentRunner,
) -> None:
    result: typing.Final = agent_runner.run(TEST_USER_ID, "Помоги мне")

    assert result.status is contracts.AgentRunStatus.NEEDS_INPUT
