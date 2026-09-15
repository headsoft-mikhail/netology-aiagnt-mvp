import dataclasses
import sys
import typing

import faker as faker_lib
import pytest

from ai_agent import cli, contracts

TEST_ANSWER: typing.Final = "Подходящий роутер найден."
TEST_QUESTION: typing.Final = "Подбери роутер"
TEST_REQUEST_ID: typing.Final = "cli-test-request"
PREPARE_COMMAND: typing.Final = "just prepare_agent"
INTERNAL_ERROR_DETAILS: typing.Final = "internal path must not be shown"


@dataclasses.dataclass
class FakeAgentRunner:
    status: contracts.AgentRunStatus
    ended_session_id: str | None = None

    def run(self, user_id: str, question: str, *, session_id: str) -> contracts.AgentResponse:
        return contracts.AgentResponse(
            status=self.status,
            answer=TEST_ANSWER,
            request_id=TEST_REQUEST_ID,
            session_id=session_id,
        )

    def end_session(self, session_id: str) -> None:
        self.ended_session_id = session_id


def test_single_question_prints_answer_and_ends_session(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    faker: faker_lib.Faker,
) -> None:
    user_id: typing.Final = faker.uuid4()
    session_id: typing.Final = faker.uuid4()
    runner: typing.Final = FakeAgentRunner(status=contracts.AgentRunStatus.COMPLETED)
    monkeypatch.setattr(cli, "create_agent", lambda: runner)

    exit_code: typing.Final = cli.start_agent(
        user_id,
        session_id,
        question=TEST_QUESTION,
    )

    captured: typing.Final = capsys.readouterr()
    assert exit_code == cli.EXIT_SUCCESS
    assert TEST_ANSWER in captured.out
    assert captured.err == ""
    assert runner.ended_session_id == session_id


def test_failed_single_question_returns_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    faker: faker_lib.Faker,
) -> None:
    runner: typing.Final = FakeAgentRunner(status=contracts.AgentRunStatus.FAILED)
    monkeypatch.setattr(cli, "create_agent", lambda: runner)

    exit_code: typing.Final = cli.start_agent(faker.uuid4(), question=TEST_QUESTION)

    assert exit_code == cli.EXIT_FAILURE


def test_startup_failure_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_create_agent() -> typing.NoReturn:
        raise RuntimeError(INTERNAL_ERROR_DETAILS)

    monkeypatch.setattr(cli, "create_agent", fail_to_create_agent)
    monkeypatch.setattr(sys, "argv", ["ai-agent", TEST_QUESTION])

    exit_code: typing.Final = cli.main()

    captured: typing.Final = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert PREPARE_COMMAND in captured.err
    assert INTERNAL_ERROR_DETAILS not in captured.err
