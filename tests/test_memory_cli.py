import pathlib
import sys
import typing

import faker as faker_lib
import pytest

from ai_agent import contracts
from ai_agent.memory import cli, config, repository

TEST_USER_FACTS: typing.Final = "Предпочтительный бренд Keenetic"
TEST_MEMORY_SOURCE: typing.Final = "test"


def test_clear_memory_removes_database(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    faker: faker_lib.Faker,
) -> None:
    loaded_config: typing.Final = config.MemoryConfig(database_path=tmp_path / "memory.db")
    memory: typing.Final = repository.MemoryRepository(database_path=loaded_config.database_path)
    memory.save_fact(
        faker.uuid4(),
        contracts.MemoryKey.PREFERRED_BRANDS,
        TEST_USER_FACTS,
        source=TEST_MEMORY_SOURCE,
    )

    monkeypatch.setattr(cli, "memory_config", loaded_config)
    monkeypatch.setattr(sys, "argv", ["ai_agent.memory", "clear"])

    cli.main()

    assert not loaded_config.database_path.exists()
