import sys
import typing

import faker as faker_lib
import pytest

from ai_agent import cli

PREPARE_COMMAND: typing.Final = "just prepare_agent"
INTERNAL_ERROR_DETAILS: typing.Final = "internal path must not be shown"
USER_ID_UNDERSCORE_OPTION: typing.Final = "--user_id"
SESSION_ID_UNDERSCORE_OPTION: typing.Final = "--session_id"
LOG_LEVEL_OPTION: typing.Final = "--log-level"
DEBUG_LEVEL: typing.Final = "debug"


def test_startup_failure_is_reported_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail_to_create_agent() -> typing.NoReturn:
        raise RuntimeError(INTERNAL_ERROR_DETAILS)

    monkeypatch.setattr(cli, "create_agent", fail_to_create_agent)
    monkeypatch.setattr(sys, "argv", ["ai-agent"])

    exit_code: typing.Final = cli.main()

    captured: typing.Final = capsys.readouterr()
    assert exit_code == cli.EXIT_FAILURE
    assert PREPARE_COMMAND in captured.err
    assert INTERNAL_ERROR_DETAILS not in captured.err


def test_cli_accepts_underscore_aliases_for_identifiers(
    monkeypatch: pytest.MonkeyPatch,
    faker: faker_lib.Faker,
) -> None:
    user_id: typing.Final = faker.uuid4()
    session_id: typing.Final = faker.uuid4()
    received_arguments: typing.Final[dict[str, object]] = {}

    def fake_start_agent(
        actual_user_id: str,
        actual_session_id: str | None = None,
        *,
        show_trace: bool = False,
    ) -> int:
        received_arguments.update(
            user_id=actual_user_id,
            session_id=actual_session_id,
            show_trace=show_trace,
        )
        return cli.EXIT_SUCCESS

    monkeypatch.setattr(cli, "start_agent", fake_start_agent)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "ai-agent",
            LOG_LEVEL_OPTION,
            DEBUG_LEVEL,
            USER_ID_UNDERSCORE_OPTION,
            user_id,
            SESSION_ID_UNDERSCORE_OPTION,
            session_id,
        ],
    )

    assert cli.main() == cli.EXIT_SUCCESS
    assert received_arguments == {
        "user_id": user_id,
        "session_id": session_id,
        "show_trace": False,
    }
