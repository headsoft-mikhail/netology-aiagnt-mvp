import json
import typing

import constants
import pytest
import tools
from pydantic import ValidationError


def test_existing_engineer_permissions() -> None:
    tool_input: typing.Final = tools.PermissionCheckInput(
        engineer_id=constants.EXISTING_ENGINEER_ID
    )

    result: typing.Final = tools.get_engineer_permissions(tool_input)

    assert f"#{constants.EXISTING_ENGINEER_ID}" in result
    assert constants.PRODENV_PERMISSION_R in result
    assert constants.TESTENV_PERMISSION_RW in result


def test_inexistent_engineer_returns_structured_error() -> None:
    tool_input: typing.Final = tools.PermissionCheckInput(
        engineer_id=constants.INEXISTENT_ENGINEER_ID
    )

    result: typing.Final = json.loads(tools.get_engineer_permissions(tool_input))

    assert result == {
        "status": tools.ERROR_STATUS,
        "message": tools.INEXISTENT_ENGINEER_ERROR,
    }


def test_engineer_id_rejects_string() -> None:
    string_engineer_id: typing.Final = typing.cast(
        typing.Any, str(constants.EXISTING_ENGINEER_ID)
    )

    with pytest.raises(ValidationError):
        tools.PermissionCheckInput(engineer_id=string_engineer_id)
