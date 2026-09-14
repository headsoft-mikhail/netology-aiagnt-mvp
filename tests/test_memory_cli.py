import pathlib
import sys
import typing

import pytest

from ai_agent.memory import cli, config, repository

TEST_USER_ID: typing.Final = "test_user"
TEST_USER_FACTS: typing.Final = "Предпочтительный бренд Keenetic"


def test_clear_memory_removes_database(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded_config: typing.Final = config.MemoryConfig(database_path=tmp_path / "memory.db")
    memory: typing.Final = repository.MemoryRepository(database_path=loaded_config.database_path)
    memory.save_fact(TEST_USER_ID, TEST_USER_FACTS)

    monkeypatch.setattr(cli, "memory_config", loaded_config)
    monkeypatch.setattr(sys, "argv", ["ai_agent.memory", "clear"])

    cli.main()

    assert not loaded_config.database_path.exists()
