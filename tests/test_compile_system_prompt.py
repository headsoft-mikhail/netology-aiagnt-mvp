import pathlib
import typing

import constants
import database
import prompts


def test_compile_system_prompt_contains_existing_user_facts(
    tmp_path: pathlib.Path,
) -> None:
    db: typing.Final = database.SQLiteMemoryStore(str(tmp_path / "memory.db"))

    system_prompt: typing.Final = prompts.compile_system_prompt(
        constants.TEST_USER_ID,
        db,
    )

    assert constants.TEST_USER_FACTS in system_prompt


def test_compile_system_prompt_contains_fallback_for_unknown_user(
    tmp_path: pathlib.Path,
) -> None:
    db: typing.Final = database.SQLiteMemoryStore(str(tmp_path / "memory.db"))

    system_prompt: typing.Final = prompts.compile_system_prompt("unknown_user", db)

    assert prompts.MISSING_USER_FACTS_STR in system_prompt
