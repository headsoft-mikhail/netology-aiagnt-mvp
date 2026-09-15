import json
import logging
import pathlib
import typing

import faker as faker_lib
import pytest
import qdrant_client.models

from ai_agent import agent, contracts
from ai_agent.catalog import config as catalog_config
from ai_agent.catalog import repository as catalog_repository
from ai_agent.catalog import restore
from ai_agent.llm import client as llm_client
from ai_agent.llm import models as llm_models
from ai_agent.llm import service
from ai_agent.memory import repository as memory_repository
from ai_agent.rag import context
from ai_agent.tools import search_knowledge_base, search_products

TEST_USER_ID: typing.Final = "test_user"
OTHER_USER_ID: typing.Final = "other_user"
TEST_SESSION_ID: typing.Final = "test_session"
TEST_BUDGET: typing.Final = "10000"
TEST_EXCLUDED_BRAND: typing.Final = "D-Link"
ROUTER_PRODUCT_CODE: typing.Final = "RTR-TP-AX23"
MESH_PRODUCT_CODE: typing.Final = "MSH-KN-BUDDY-2"
KNOWLEDGE_QUERY: typing.Final = "требования к роутеру для гигабитного тарифа"
KNOWLEDGE_SOURCE: typing.Final = "router_selection.txt"
KNOWLEDGE_TEXT: typing.Final = "Для тарифа 1 Гбит/с нужен WAN-порт не менее 1 Гбит/с."
MESH_KNOWLEDGE_QUERY: typing.Final = "выбор Mesh-системы для большой площади"
MESH_KNOWLEDGE_SOURCE: typing.Final = "coverage_and_mesh.txt"
MESH_KNOWLEDGE_TEXT: typing.Final = "Для большой площади используйте Mesh-систему из нескольких модулей."
MEMORY_LIMIT: typing.Final = 10
NO_RESULT_MAX_PRICE: typing.Final = 100
CATALOG_RESULT_LIMIT: typing.Final = 3
CONTEXT_MIN_SCORE: typing.Final = 0.8
RETRIEVAL_SCORE: typing.Final = 0.95
TEST_MODEL_NAME: typing.Final = "scripted-test-model"
UNAVAILABLE_MODEL_NAME: typing.Final = "unavailable-test-model"
KNOWLEDGE_CASES: typing.Final = (
    (
        "Какой WAN-порт нужен для тарифа 1 Гбит/с?",
        "Для тарифа 1 Гбит/с нужен WAN-порт не менее 1 Гбит/с.",
    ),
    (
        "Что важно проверить перед покупкой гигабитного роутера?",
        "Проверьте скорость WAN-порта: она должна быть не ниже скорости тарифа.",
    ),
)
CATALOG_CASES: typing.Final = (
    (
        "Покажи USB-адаптер Wi-Fi 6 до 7000 рублей",
        {
            "category": "wifi_adapter",
            "connection_type": "usb",
            "min_wifi_generation": 6,
            "max_price_rub": 7000,
        },
        "ADP-DL-DWA-X1850",
    ),
    (
        "Нужна точка доступа Wi-Fi 7 с PoE",
        {
            "category": "access_point",
            "min_wifi_generation": 7,
            "poe": True,
        },
        "AP-UB-U7P",
    ),
    (
        "Покажи управляемый коммутатор с PoE до 18000 рублей",
        {
            "category": "network_switch",
            "managed": True,
            "poe": True,
            "max_price_rub": 18000,
        },
        "SWT-TP-SG108PE",
    ),
)
DIRECT_DIALOGUE_CASES: typing.Final = (
    ("Подбери устройство", "Уточните категорию и бюджет.", contracts.AgentRunStatus.NEEDS_INPUT),
    ("Нужно улучшить Wi-Fi", "Уточните площадь и текущую проблему.", contracts.AgentRunStatus.NEEDS_INPUT),
    ("Подбери страховку", "Я консультирую только по сетевому оборудованию.", contracts.AgentRunStatus.COMPLETED),
    ("Расскажи прогноз погоды", "Этот вопрос вне тематики магазина.", contracts.AgentRunStatus.COMPLETED),
)
NO_RESULT_CASES: typing.Final = (
    ("Найди роутер дешевле 100 рублей", "router", "В заданном бюджете роутеров нет."),
    ("Найди Mesh-систему дешевле 100 рублей", "mesh_system", "В заданном бюджете Mesh-систем нет."),
    ("Найди точку доступа дешевле 100 рублей", "access_point", "В заданном бюджете точек доступа нет."),
)
MEMORY_UPDATE_CASES: typing.Final = (
    (
        contracts.MemoryKey.PREFERRED_BRANDS,
        "Keenetic",
        "Запомни, что я предпочитаю Keenetic",
    ),
    (
        contracts.MemoryKey.BUDGET_RUB,
        TEST_BUDGET,
        "Запомни мой бюджет 10000 рублей",
    ),
)
INVALID_CATALOG_PRICES: typing.Final = (-100, "недорого")


class ScriptedChatClient:
    def __init__(self, responses: list[dict[str, object]]) -> None:
        self.responses = [json.dumps(response, ensure_ascii=False) for response in responses]
        self.calls: list[tuple[str, str]] = []

    @property
    def model_name(self) -> str:
        return TEST_MODEL_NAME

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str:
        assert json_mode
        self.calls.append((system_prompt, user_prompt))
        return self.responses.pop(0)


class FakeRetrievalClient:
    def top_k(self, query: str) -> list[qdrant_client.models.ScoredPoint]:
        knowledge_by_query: typing.Final = {
            KNOWLEDGE_QUERY: (KNOWLEDGE_SOURCE, KNOWLEDGE_TEXT),
            MESH_KNOWLEDGE_QUERY: (MESH_KNOWLEDGE_SOURCE, MESH_KNOWLEDGE_TEXT),
        }
        source, text = knowledge_by_query[query]
        return [
            qdrant_client.models.ScoredPoint(
                id=1,
                version=1,
                score=RETRIEVAL_SCORE,
                payload={
                    "chunk_id": "router-1",
                    "source": source,
                    "text": text,
                },
            )
        ]


class UnavailableChatClient:
    @property
    def model_name(self) -> str:
        return UNAVAILABLE_MODEL_NAME

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> typing.NoReturn:
        raise llm_client.LLMUnavailableError("LLM unavailable in test")


@pytest.fixture
def repositories(
    tmp_path: pathlib.Path,
) -> tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository]:
    memory: typing.Final = memory_repository.MemoryRepository(database_path=tmp_path / "memory.db")
    catalog_path: typing.Final = tmp_path / "products.db"
    snapshot_path: typing.Final = pathlib.Path(restore.__file__).parent / "data" / "products.csv"
    restore.restore_catalog(
        catalog_config.CatalogConfig(
            snapshot_path=snapshot_path,
            database_path=catalog_path,
        )
    )
    return memory, catalog_repository.ProductsRepository(database_path=catalog_path)


def create_agent_runner(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    responses: list[dict[str, object]],
) -> tuple[agent.AgentRunner, ScriptedChatClient]:
    memory, catalog = repositories
    chat_client: typing.Final = ScriptedChatClient(responses)
    return (
        agent.AgentRunner(
            memory=memory,
            llm=service.LLMService(chat_client=chat_client),
            catalog=catalog,
            retrieval_client=FakeRetrievalClient(),
            context_builder=context.ContextBuilder(min_score=CONTEXT_MIN_SCORE),
        ),
        chat_client,
    )


def test_agent_uses_memory_rag_and_catalog_for_router_selection(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    caplog: pytest.LogCaptureFixture,
) -> None:
    memory, _ = repositories
    memory.save_fact(
        TEST_USER_ID,
        contracts.MemoryKey.BUDGET_RUB,
        TEST_BUDGET,
        source=agent.MEMORY_SOURCE,
    )
    memory.save_fact(
        TEST_USER_ID,
        contracts.MemoryKey.EXCLUDED_BRANDS,
        TEST_EXCLUDED_BRAND,
        source=agent.MEMORY_SOURCE,
    )
    memory_before: typing.Final = memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT)
    runner, chat_client = create_agent_runner(
        repositories,
        [
            {
                "actions": ["search_knowledge_base", "search_products"],
                "knowledge_query": KNOWLEDGE_QUERY,
            },
            {
                "filters": {
                    "category": "router",
                    "max_price_rub": 10000,
                    "excluded_brands": [TEST_EXCLUDED_BRAND],
                    "min_wifi_generation": 6,
                    "min_wan_speed_mbps": 1000,
                },
                "limit": CATALOG_RESULT_LIMIT,
            },
            {
                "answer": f"TP-Link Archer AX23 ({ROUTER_PRODUCT_CODE}) подходит и стоит 6490 рублей.",
                "product_codes": [ROUTER_PRODUCT_CODE],
            },
        ],
    )

    with caplog.at_level(logging.INFO, logger="ai_agent.agent"):
        result: typing.Final = runner.run(
            TEST_USER_ID,
            "Подбери Wi-Fi 6 роутер до 10 000 рублей для тарифа 1 Гбит/с",
            session_id=TEST_SESSION_ID,
        )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert [call.tool_name for call in result.tool_calls] == ["search_knowledge_base", "search_products"]
    assert result.product_codes == [ROUTER_PRODUCT_CODE]
    assert result.sources == [KNOWLEDGE_SOURCE]
    assert result.memory_used
    assert TEST_BUDGET in chat_client.calls[1][1]
    assert TEST_EXCLUDED_BRAND in chat_client.calls[1][1]
    assert memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT) == memory_before
    catalog_tool_input: typing.Final = json.loads(result.tool_calls[1].input_json)
    assert catalog_tool_input["filters"]["max_price_rub"] == int(TEST_BUDGET)
    assert catalog_tool_input["filters"]["excluded_brands"] == [TEST_EXCLUDED_BRAND]
    assert ROUTER_PRODUCT_CODE in result.answer
    event_names: typing.Final = [
        record.message.split()[0] for record in caplog.records if record.message.startswith("event=")
    ]
    assert event_names == [
        "event=request_accepted",
        "event=memory_load_started",
        "event=memory_load_completed",
        "event=llm_request_started",
        "event=llm_request_completed",
        "event=actions_selected",
        "event=memory_update_skipped",
        "event=tool_call_started",
        "event=tool_call_completed",
        "event=rag_retrieval_completed",
        "event=llm_request_started",
        "event=llm_request_completed",
        "event=catalog_filters_validated",
        "event=tool_call_started",
        "event=tool_call_completed",
        "event=context_assembled",
        "event=llm_request_started",
        "event=llm_request_completed",
        "event=request_completed",
    ]
    assert all(
        f"request_id={result.request_id}" in record.message and f"session_id={TEST_SESSION_ID}" in record.message
        for record in caplog.records
        if record.message.startswith("event=")
    )


@pytest.mark.parametrize(
    ("memory_key", "memory_value", "user_query"),
    MEMORY_UPDATE_CASES,
)
def test_agent_saves_only_explicit_memory_update(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    caplog: pytest.LogCaptureFixture,
    memory_key: contracts.MemoryKey,
    memory_value: str,
    user_query: str,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {
                "actions": ["update_memory"],
                "memory_update": {"key": memory_key.value, "value": memory_value},
            }
        ],
    )

    with caplog.at_level(logging.INFO, logger="ai_agent.agent"):
        result: typing.Final = runner.run(
            TEST_USER_ID,
            user_query,
            session_id=TEST_SESSION_ID,
        )
    memory, _ = repositories
    saved_facts: typing.Final = memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.tool_calls == []
    assert [(fact.key, fact.value) for fact in saved_facts] == [(memory_key, memory_value)]
    assert any("event=memory_update_completed" in record.message for record in caplog.records)
    assert all(memory_value not in record.message for record in caplog.records)


def test_agent_combines_rag_and_catalog_for_mesh_selection(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {
                "actions": ["search_knowledge_base", "search_products"],
                "knowledge_query": MESH_KNOWLEDGE_QUERY,
            },
            {
                "filters": {
                    "category": "mesh_system",
                    "max_price_rub": 20000,
                    "min_nodes": 2,
                    "min_coverage_sqm": 400,
                },
                "limit": CATALOG_RESULT_LIMIT,
            },
            {
                "answer": f"Для большой квартиры подходит Mesh-система {MESH_PRODUCT_CODE}.",
                "product_codes": [MESH_PRODUCT_CODE],
            },
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Подбери Mesh-систему из двух модулей для площади 400 м² до 20000 рублей",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.product_codes == [MESH_PRODUCT_CODE]
    assert [call.tool_name for call in result.tool_calls] == ["search_knowledge_base", "search_products"]


def test_agent_deletes_requested_memory_fact_only(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, _ = repositories
    for key, value in (
        (contracts.MemoryKey.BUDGET_RUB, "10000"),
        (contracts.MemoryKey.PREFERRED_BRANDS, "Keenetic"),
    ):
        memory.save_fact(TEST_USER_ID, key, value, source=agent.MEMORY_SOURCE)
    runner, _ = create_agent_runner(
        repositories,
        [{"actions": ["delete_memory"], "memory_key": "budget_rub"}],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Удали сохранённый бюджет",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert [(fact.key, fact.value) for fact in memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT)] == [
        (contracts.MemoryKey.PREFERRED_BRANDS, "Keenetic")
    ]


def test_agent_clears_all_memory_only_for_requesting_user(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, _ = repositories
    memory.save_fact(TEST_USER_ID, contracts.MemoryKey.BUDGET_RUB, "10000", source=agent.MEMORY_SOURCE)
    memory.save_fact(OTHER_USER_ID, contracts.MemoryKey.BUDGET_RUB, "20000", source=agent.MEMORY_SOURCE)
    runner, _ = create_agent_runner(repositories, [{"actions": ["clear_memory"]}])

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Удали все мои сохранённые предпочтения",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT) == []
    assert [fact.value for fact in memory.get_relevant(OTHER_USER_ID, limit=MEMORY_LIMIT)] == ["20000"]


@pytest.mark.parametrize(
    ("user_query", "category", "answer"),
    NO_RESULT_CASES,
)
def test_agent_returns_not_found_without_inventing_product(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    user_query: str,
    category: str,
    answer: str,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {"actions": ["search_products"]},
            {
                "filters": {"category": category, "max_price_rub": NO_RESULT_MAX_PRICE},
                "limit": CATALOG_RESULT_LIMIT,
            },
            {"answer": answer, "product_codes": []},
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        user_query,
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.NOT_FOUND
    assert result.product_codes == []
    assert result.tool_calls[0].status is contracts.ToolStatus.NO_RESULTS


def test_agent_rejects_product_not_returned_by_catalog(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {"actions": ["search_products"]},
            {"filters": {"category": "router", "max_price_rub": 7000}, "limit": 3},
            {"answer": "Рекомендую выдуманный роутер.", "product_codes": ["UNKNOWN-PRODUCT"]},
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Найди роутер до 7000 рублей",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.errors[0].code is contracts.ErrorCode.INVALID_LLM_RESPONSE


def test_agent_uses_short_term_history_and_clears_it_at_session_end(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    runner, chat_client = create_agent_runner(
        repositories,
        [
            {"actions": ["clarify"], "response_message": "Какой у вас бюджет?"},
            {"actions": ["answer"], "response_message": "Бюджет принят."},
            {"actions": ["answer"], "response_message": "Начинаем заново."},
        ],
    )
    first_query: typing.Final = "Подбери роутер"

    clarification_result: typing.Final = runner.run(TEST_USER_ID, first_query, session_id=TEST_SESSION_ID)
    runner.run(TEST_USER_ID, "До 10000 рублей", session_id=TEST_SESSION_ID)

    assert clarification_result.status is contracts.AgentRunStatus.NEEDS_INPUT
    assert clarification_result.tool_calls == []
    assert first_query in chat_client.calls[1][1]
    assert "Какой у вас бюджет?" in chat_client.calls[1][1]

    runner.end_session(TEST_SESSION_ID)
    runner.run(TEST_USER_ID, "Новый вопрос", session_id=TEST_SESSION_ID)

    assert first_query not in chat_client.calls[2][1]


def test_agent_handles_unsupported_plan_without_optional_llm_message(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [{"actions": ["unsupported"], "response_message": None}],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Сколько стоит страхование жизни?",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.answer == llm_models.UNSUPPORTED_MESSAGE
    assert result.tool_calls == []


@pytest.mark.parametrize("invalid_price", INVALID_CATALOG_PRICES)
def test_agent_handles_invalid_catalog_filter_from_llm(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    invalid_price: object,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {"actions": ["search_products"]},
            {
                "filters": {
                    "category": "router",
                    "max_price_rub": invalid_price,
                }
            },
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Найди роутер с некорректной ценой",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls == []
    assert result.errors[0].code is contracts.ErrorCode.INVALID_LLM_RESPONSE


def test_agent_handles_catalog_tool_failure(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    tmp_path: pathlib.Path,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {"actions": ["search_products"]},
            {"filters": {"category": "router"}},
        ],
    )
    runner.catalog = catalog_repository.ProductsRepository(database_path=tmp_path / "missing.db")

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Покажи роутеры в наличии",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls[0].status is contracts.ToolStatus.ERROR
    assert result.errors[0].code is contracts.ErrorCode.CATALOG_UNAVAILABLE


def test_agent_handles_unavailable_llm(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, catalog = repositories
    runner: typing.Final = agent.AgentRunner(
        memory=memory,
        llm=service.LLMService(chat_client=UnavailableChatClient()),
        catalog=catalog,
        retrieval_client=FakeRetrievalClient(),
        context_builder=context.ContextBuilder(min_score=CONTEXT_MIN_SCORE),
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        "Подбери роутер",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls == []
    assert result.errors[0].code is contracts.ErrorCode.LLM_UNAVAILABLE


@pytest.mark.parametrize(
    ("user_query", "answer"),
    KNOWLEDGE_CASES,
)
def test_agent_answers_knowledge_questions_using_rag(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    user_query: str,
    answer: str,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {
                "actions": ["search_knowledge_base"],
                "knowledge_query": KNOWLEDGE_QUERY,
            },
            {"answer": answer, "product_codes": []},
        ],
    )

    result: typing.Final = runner.run(TEST_USER_ID, user_query, session_id=TEST_SESSION_ID)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.answer == answer
    assert result.sources == [KNOWLEDGE_SOURCE]
    assert [call.tool_name for call in result.tool_calls] == [search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME]


@pytest.mark.parametrize(
    ("user_query", "filters", "product_code"),
    CATALOG_CASES,
)
def test_agent_searches_catalog_for_different_product_categories(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    user_query: str,
    filters: dict[str, object],
    product_code: str,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {"actions": ["search_products"]},
            {"filters": filters, "limit": 10},
            {"answer": f"Подходящий товар: {product_code}.", "product_codes": [product_code]},
        ],
    )

    result: typing.Final = runner.run(TEST_USER_ID, user_query, session_id=TEST_SESSION_ID)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.product_codes == [product_code]
    assert [call.tool_name for call in result.tool_calls] == [search_products.SEARCH_PRODUCTS_TOOL_NAME]


@pytest.mark.parametrize(
    ("user_query", "response_message", "expected_status"),
    DIRECT_DIALOGUE_CASES,
)
def test_agent_handles_direct_dialogue_branches(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    user_query: str,
    response_message: str,
    expected_status: contracts.AgentRunStatus,
) -> None:
    action: typing.Final = "clarify" if expected_status is contracts.AgentRunStatus.NEEDS_INPUT else "unsupported"
    runner, _ = create_agent_runner(
        repositories,
        [{"actions": [action], "response_message": response_message}],
    )

    result: typing.Final = runner.run(TEST_USER_ID, user_query, session_id=TEST_SESSION_ID)

    assert result.status is expected_status
    assert result.answer == response_message
    assert result.tool_calls == []


def test_agent_rejects_session_owned_by_another_user(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    faker: faker_lib.Faker,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [{"actions": ["answer"], "response_message": "Сессия создана."}],
    )
    runner.run(TEST_USER_ID, "Начать консультацию", session_id=TEST_SESSION_ID)

    result: typing.Final = runner.run(
        faker.uuid4(),
        "Продолжить чужую консультацию",
        session_id=TEST_SESSION_ID,
    )

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.errors[0].code is contracts.ErrorCode.INVALID_INPUT
    assert result.session_id == TEST_SESSION_ID
