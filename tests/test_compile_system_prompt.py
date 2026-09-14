import pathlib
import typing

from ai_agent import prompts
from ai_agent.memory import repository as memory_repository

TEST_USER_ID: typing.Final = "test_user"
TEST_USER_FACTS: typing.Final = "Бюджет до 10 000 рублей"


def test_compile_system_prompt_contains_existing_user_facts(
    tmp_path: pathlib.Path,
) -> None:
    db: typing.Final = memory_repository.MemoryRepository(database_path=tmp_path / "memory.db")
    db.save_fact(TEST_USER_ID, TEST_USER_FACTS)

    system_prompt: typing.Final = prompts.compile_system_prompt(
        TEST_USER_ID,
        db,
    )

    assert TEST_USER_FACTS in system_prompt


def test_compile_system_prompt_contains_fallback_for_unknown_user(
    tmp_path: pathlib.Path,
) -> None:
    db: typing.Final = memory_repository.MemoryRepository(database_path=tmp_path / "memory.db")

    system_prompt: typing.Final = prompts.compile_system_prompt("unknown_user", db)

    assert memory_repository.MISSING_USER_FACTS_STR in system_prompt
