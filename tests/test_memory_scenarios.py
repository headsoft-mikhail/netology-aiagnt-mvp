import pathlib
import typing

import faker as faker_lib
import pytest

from ai_agent import contracts
from ai_agent.memory import repository

INITIAL_BUDGET: typing.Final = "10000"
UPDATED_BUDGET: typing.Final = "15000"
FIRST_PREFERRED_BRAND: typing.Final = "Keenetic"
SECOND_PREFERRED_BRAND: typing.Final = "ASUS"
MEMORY_SOURCE: typing.Final = "explicit_user_request"
SENSITIVE_VALUE: typing.Final = "Пароль Wi-Fi: super-secret"


def test_new_explicit_budget_replaces_previous_value_for_same_user(
    tmp_path: pathlib.Path,
    faker: faker_lib.Faker,
) -> None:
    user_id: typing.Final = faker.uuid4()
    memory: typing.Final = repository.MemoryRepository(database_path=tmp_path / "memory.db")
    memory.save_fact(
        user_id,
        contracts.MemoryKey.BUDGET_RUB,
        INITIAL_BUDGET,
        source=MEMORY_SOURCE,
    )
    memory.save_fact(
        user_id,
        contracts.MemoryKey.BUDGET_RUB,
        UPDATED_BUDGET,
        source=MEMORY_SOURCE,
    )

    facts: typing.Final = memory.get_relevant(user_id, limit=10)

    assert [(fact.key, fact.value) for fact in facts] == [(contracts.MemoryKey.BUDGET_RUB, UPDATED_BUDGET)]


def test_clearing_one_user_does_not_remove_another_users_preferences(
    tmp_path: pathlib.Path,
    faker: faker_lib.Faker,
) -> None:
    first_user_id: typing.Final = faker.uuid4()
    second_user_id: typing.Final = faker.uuid4()
    memory: typing.Final = repository.MemoryRepository(database_path=tmp_path / "memory.db")
    for user_id, brand in (
        (first_user_id, FIRST_PREFERRED_BRAND),
        (second_user_id, SECOND_PREFERRED_BRAND),
    ):
        memory.save_fact(
            user_id,
            contracts.MemoryKey.PREFERRED_BRANDS,
            brand,
            source=MEMORY_SOURCE,
        )

    memory.clear_user(first_user_id)

    assert memory.get_relevant(first_user_id, limit=10) == []
    assert [fact.value for fact in memory.get_relevant(second_user_id, limit=10)] == [SECOND_PREFERRED_BRAND]


def test_memory_rejects_sensitive_value(tmp_path: pathlib.Path, faker: faker_lib.Faker) -> None:
    user_id: typing.Final = faker.uuid4()
    memory: typing.Final = repository.MemoryRepository(database_path=tmp_path / "memory.db")

    with pytest.raises(ValueError, match="Sensitive data"):
        memory.save_fact(
            user_id,
            contracts.MemoryKey.CURRENT_EQUIPMENT,
            SENSITIVE_VALUE,
            source=MEMORY_SOURCE,
        )

    assert memory.get_relevant(user_id, limit=10) == []
