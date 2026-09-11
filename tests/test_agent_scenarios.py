import json
import pathlib
import typing

import agent
import constants
import database
import pytest
import tools

MEMORY_REQUEST: typing.Final = "На каком проекте я сейчас работаю и какой мой язык?"
EXISTING_ENGINEER_REQUEST: typing.Final = (
    f"Проверь мои доступы к серверам. Мой ID {constants.EXISTING_ENGINEER_ID}"
)
INEXISTENT_ENGINEER_REQUEST: typing.Final = (
    f"Посмотри права для инженера {constants.INEXISTENT_ENGINEER_ID}"
)


@pytest.fixture
def agent_runner(tmp_path: pathlib.Path) -> agent.MockAgentRunner:
    db: typing.Final = database.SQLiteMemoryStore(str(tmp_path / "memory.db"))
    return agent.MockAgentRunner(db)


def test_agent_uses_long_term_memory(agent_runner: agent.MockAgentRunner) -> None:
    result: typing.Final = agent_runner.run(constants.TEST_USER_ID, MEMORY_REQUEST)

    assert constants.TEST_USER_PROJECT in result["agent_response"]
    assert constants.TEST_USER_LANGUAGE in result["agent_response"]
    assert result["tool_called"] is None


def test_agent_calls_tool_for_existing_engineer(
    agent_runner: agent.MockAgentRunner,
) -> None:
    result: typing.Final = agent_runner.run(
        constants.TEST_USER_ID, EXISTING_ENGINEER_REQUEST
    )

    assert result["tool_called"] == tools.PERMISSIONS_TOOL_NAME
    assert result["tool_args"] == {"engineer_id": constants.EXISTING_ENGINEER_ID}
    assert constants.PRODENV_PERMISSION_R in result["agent_response"]


def test_agent_handles_inexistent_engineer_error(
    agent_runner: agent.MockAgentRunner,
) -> None:
    result: typing.Final = agent_runner.run(
        constants.TEST_USER_ID, INEXISTENT_ENGINEER_REQUEST
    )
    tool_result: typing.Final = json.loads(result["tool_result"])

    assert result["tool_called"] == tools.PERMISSIONS_TOOL_NAME
    assert tool_result["status"] == tools.ERROR_STATUS
    assert tools.INEXISTENT_ENGINEER_ERROR in tool_result["message"]
    assert tools.INEXISTENT_ENGINEER_ERROR in result["agent_response"]
