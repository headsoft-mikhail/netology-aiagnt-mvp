import dataclasses
import json
import logging
import pathlib
import sqlite3
import typing

import faker as faker_lib
import pytest
import qdrant_client.models

from ai_agent import agent, contracts
from ai_agent.catalog import repository as catalog_repository
from ai_agent.llm import client as llm_client
from ai_agent.llm import models as llm_models
from ai_agent.llm import service
from ai_agent.memory import repository as memory_repository
from ai_agent.rag.retrieval import context
from ai_agent.tools import registry, search_knowledge_base, search_products
from tests import reporting

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
REQUEST_FAILED_EVENT: typing.Final = "event=request_failed"
LLM_COMPONENT_FIELD: typing.Final = 'component="llm"'
INVALID_LLM_ERROR_CODE_FIELD: typing.Final = 'error_code="invalid_llm_response"'
INVALID_LLM_ERROR_TYPE_FIELD: typing.Final = 'error_type="InvalidLLMResponseError"'
MEMORY_DATABASE_ERROR: typing.Final = "simulated memory database failure"
KNOWLEDGE_CASES: typing.Final = (
    pytest.param(
        "TC-RAG-001",
        "Для тарифа 1 Гбит/с нужен WAN-порт не менее 1 Гбит/с.",
        marks=pytest.mark.report_case("TC-RAG-001"),
    ),
    pytest.param(
        "TC-RAG-002",
        "Проверьте скорость WAN-порта: она должна быть не ниже скорости тарифа.",
        marks=pytest.mark.report_case("TC-RAG-002"),
    ),
)
CATALOG_CASES: typing.Final = (
    pytest.param(
        "TC-CATALOG-001",
        {
            "category": "wifi_adapter",
            "connection_type": "usb",
            "min_wifi_generation": 6,
            "max_price_rub": 7000,
        },
        "ADP-DL-DWA-X1850",
        marks=pytest.mark.report_case("TC-CATALOG-001"),
    ),
    pytest.param(
        "TC-CATALOG-002",
        {
            "category": "access_point",
            "min_wifi_generation": 7,
            "poe": True,
        },
        "AP-UB-U7P",
        marks=pytest.mark.report_case("TC-CATALOG-002"),
    ),
    pytest.param(
        "TC-CATALOG-003",
        {
            "category": "network_switch",
            "managed": True,
            "poe": True,
            "max_price_rub": 18000,
        },
        "SWT-TP-SG108PE",
        marks=pytest.mark.report_case("TC-CATALOG-003"),
    ),
)
DIRECT_DIALOGUE_CASES: typing.Final = (
    pytest.param(
        "TC-CLARIFY-001",
        "Уточните категорию и бюджет.",
        contracts.AgentRunStatus.NEEDS_INPUT,
        marks=pytest.mark.report_case("TC-CLARIFY-001"),
    ),
    pytest.param(
        "TC-CLARIFY-002",
        "Уточните площадь и текущую проблему.",
        contracts.AgentRunStatus.NEEDS_INPUT,
        marks=pytest.mark.report_case("TC-CLARIFY-002"),
    ),
    pytest.param(
        "TC-SCOPE-001",
        "Я консультирую только по сетевому оборудованию.",
        contracts.AgentRunStatus.COMPLETED,
        marks=pytest.mark.report_case("TC-SCOPE-001"),
    ),
    pytest.param(
        "TC-SCOPE-002",
        "Этот вопрос вне тематики магазина.",
        contracts.AgentRunStatus.COMPLETED,
        marks=pytest.mark.report_case("TC-SCOPE-002"),
    ),
)
NO_RESULT_CASES: typing.Final = (
    pytest.param(
        "TC-MVP-002",
        "router",
        "В заданном бюджете роутеров нет.",
        marks=pytest.mark.report_case("TC-MVP-002"),
    ),
    pytest.param(
        "TC-NOT-FOUND-002",
        "mesh_system",
        "В заданном бюджете Mesh-систем нет.",
        marks=pytest.mark.report_case("TC-NOT-FOUND-002"),
    ),
    pytest.param(
        "TC-NOT-FOUND-003",
        "access_point",
        "В заданном бюджете точек доступа нет.",
        marks=pytest.mark.report_case("TC-NOT-FOUND-003"),
    ),
)
MEMORY_UPDATE_CASES: typing.Final = (
    pytest.param(
        "TC-MEMORY-001",
        contracts.MemoryKey.PREFERRED_BRANDS,
        "Keenetic",
        marks=pytest.mark.report_case("TC-MEMORY-001"),
    ),
    pytest.param(
        "TC-MEMORY-002",
        contracts.MemoryKey.BUDGET_RUB,
        TEST_BUDGET,
        marks=pytest.mark.report_case("TC-MEMORY-002"),
    ),
)
INVALID_CATALOG_PRICES: typing.Final = (
    pytest.param(
        "TC-VALIDATION-001",
        -100,
        marks=pytest.mark.report_case("TC-VALIDATION-001"),
    ),
    pytest.param(
        "TC-VALIDATION-002",
        "недорого",
        marks=pytest.mark.report_case("TC-VALIDATION-002"),
    ),
)
INVALID_MEMORY_CASES: typing.Final = (
    pytest.param(
        "TC-MEMORY-008",
        "-10000",
        marks=pytest.mark.report_case("TC-MEMORY-008"),
    ),
    pytest.param(
        "TC-MEMORY-009",
        "десять тысяч рублей",
        marks=pytest.mark.report_case("TC-MEMORY-009"),
    ),
)


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


class EmptyRetrievalClient:
    def top_k(self, query: str) -> list[qdrant_client.models.ScoredPoint]:
        del query
        return []


class FailOnceChatClient:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = json.dumps(response, ensure_ascii=False)
        self.calls: list[tuple[str, str]] = []
        self.failed = False

    @property
    def model_name(self) -> str:
        return TEST_MODEL_NAME

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str:
        assert json_mode
        self.calls.append((system_prompt, user_prompt))
        if not self.failed:
            self.failed = True
            raise llm_client.LLMUnavailableError("Temporary LLM failure in test")
        return self.response


@dataclasses.dataclass(kw_only=True, slots=True)
class UnavailableMemoryRepository(memory_repository.MemoryRepository):
    def get_relevant(self, user_id: str, *, limit: int) -> list[contracts.MemoryFact]:
        del user_id, limit
        raise sqlite3.OperationalError(MEMORY_DATABASE_ERROR)


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
    snapshot_path: typing.Final = pathlib.Path(catalog_repository.__file__).parent / "data" / "products.csv"
    catalog: typing.Final = catalog_repository.ProductsRepository(database_path=catalog_path)
    catalog.restore_from_snapshot(snapshot_path)
    return memory, catalog


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
            tool_registry=create_tool_registry(catalog),
        ),
        chat_client,
    )


def create_tool_registry(
    catalog: catalog_repository.ProductsRepository,
    retrieval_client: search_knowledge_base.RetrievalClientProtocol | None = None,
) -> registry.ToolRegistry:
    return registry.ToolRegistry(
        tools=(
            search_knowledge_base.KnowledgeBaseSearchTool(
                retrieval_client=retrieval_client or FakeRetrievalClient(),
                context_builder=context.ContextBuilder(min_score=CONTEXT_MIN_SCORE),
            ),
            search_products.ProductSearchTool(catalog=catalog),
        )
    )


@pytest.mark.report_case("TC-MVP-001", "TC-MEMORY-003")
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
            reporting.request("TC-MVP-001"),
            session_id=TEST_SESSION_ID,
        )
    reporting.record_agent_response("TC-MVP-001", result)
    reporting.record_agent_response("TC-MEMORY-003", result)

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
    ("case_id", "memory_key", "memory_value"),
    MEMORY_UPDATE_CASES,
)
def test_agent_saves_only_explicit_memory_update(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    caplog: pytest.LogCaptureFixture,
    case_id: str,
    memory_key: contracts.MemoryKey,
    memory_value: str,
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
            reporting.request(case_id),
            session_id=TEST_SESSION_ID,
        )
    memory, _ = repositories
    saved_facts: typing.Final = memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT)
    reporting.record_agent_response(
        case_id,
        result,
        memory_operation="save",
        memory_keys=[memory_key],
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.tool_calls == []
    assert [(fact.key, fact.value) for fact in saved_facts] == [(memory_key, memory_value)]
    assert any("event=memory_update_completed" in record.message for record in caplog.records)
    assert all(memory_value not in record.message for record in caplog.records)


@pytest.mark.report_case("TC-MEMORY-007")
def test_agent_uses_fact_saved_through_previous_dialogue_turn(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    first_query, second_query = reporting.request("TC-MEMORY-007").split(" → ")
    runner, chat_client = create_agent_runner(
        repositories,
        [
            {
                "actions": ["update_memory"],
                "memory_update": {
                    "key": contracts.MemoryKey.BUDGET_RUB.value,
                    "value": TEST_BUDGET,
                },
            },
            {"actions": ["search_products"]},
            {
                "filters": {
                    "category": "router",
                    "max_price_rub": int(TEST_BUDGET),
                    "min_wifi_generation": 6,
                },
                "limit": CATALOG_RESULT_LIMIT,
            },
            {
                "answer": f"С учётом бюджета подходит {ROUTER_PRODUCT_CODE}.",
                "product_codes": [ROUTER_PRODUCT_CODE],
            },
        ],
    )

    save_result: typing.Final = runner.run(TEST_USER_ID, first_query, session_id=TEST_SESSION_ID)
    result: typing.Final = runner.run(TEST_USER_ID, second_query, session_id=TEST_SESSION_ID)
    reporting.record_agent_response(
        "TC-MEMORY-007",
        result,
        memory_operation="read",
        memory_keys=[contracts.MemoryKey.BUDGET_RUB],
    )

    assert save_result.status is contracts.AgentRunStatus.COMPLETED
    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.memory_used
    assert TEST_BUDGET in chat_client.calls[1][1]
    assert TEST_BUDGET in chat_client.calls[2][1]
    catalog_tool_input: typing.Final = json.loads(result.tool_calls[0].input_json)
    assert catalog_tool_input["filters"]["max_price_rub"] == int(TEST_BUDGET)
    assert ROUTER_PRODUCT_CODE in result.answer
    memory, _ = repositories
    assert memory.get_relevant(OTHER_USER_ID, limit=MEMORY_LIMIT) == []


@pytest.mark.parametrize(("case_id", "invalid_value"), INVALID_MEMORY_CASES)
def test_agent_rejects_invalid_numeric_memory_value(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    case_id: str,
    invalid_value: str,
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {
                "actions": ["update_memory"],
                "memory_update": {
                    "key": contracts.MemoryKey.BUDGET_RUB.value,
                    "value": invalid_value,
                },
            }
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request(case_id),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(
        case_id,
        result,
        memory_operation="save_rejected",
        memory_keys=[contracts.MemoryKey.BUDGET_RUB],
    )

    assert result.status is contracts.AgentRunStatus.NEEDS_INPUT
    assert not result.errors
    assert "положительное целое число" in result.answer
    memory, _ = repositories
    assert memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT) == []


@pytest.mark.report_case("TC-MAIN-002")
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
        reporting.request("TC-MAIN-002"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-MAIN-002", result)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.product_codes == [MESH_PRODUCT_CODE]
    assert [call.tool_name for call in result.tool_calls] == ["search_knowledge_base", "search_products"]


@pytest.mark.report_case("TC-STATUS-001")
def test_agent_completes_combined_search_when_catalog_has_product_without_rag_context(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, catalog = repositories
    chat_client: typing.Final = ScriptedChatClient(
        [
            {
                "actions": ["search_knowledge_base", "search_products"],
                "knowledge_query": "несуществующая инструкция",
            },
            {
                "filters": {"category": "router", "max_price_rub": 7000},
                "limit": CATALOG_RESULT_LIMIT,
            },
            {
                "answer": f"В каталоге найден {ROUTER_PRODUCT_CODE}.",
                "product_codes": [ROUTER_PRODUCT_CODE],
            },
        ]
    )
    runner: typing.Final = agent.AgentRunner(
        memory=memory,
        llm=service.LLMService(chat_client=chat_client),
        tool_registry=create_tool_registry(catalog, EmptyRetrievalClient()),
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request("TC-STATUS-001"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-STATUS-001", result)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.tool_calls[0].status is contracts.ToolStatus.NO_RESULTS
    assert result.tool_calls[1].status is contracts.ToolStatus.OK
    assert result.product_codes == [ROUTER_PRODUCT_CODE]


@pytest.mark.report_case("TC-STATUS-002")
def test_agent_returns_not_found_when_combined_search_catalog_is_empty(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    runner, _ = create_agent_runner(
        repositories,
        [
            {
                "actions": ["search_knowledge_base", "search_products"],
                "knowledge_query": KNOWLEDGE_QUERY,
            },
            {
                "filters": {"category": "router", "max_price_rub": NO_RESULT_MAX_PRICE},
                "limit": CATALOG_RESULT_LIMIT,
            },
            {"answer": "Подходящих товаров нет.", "product_codes": []},
        ],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request("TC-STATUS-002"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-STATUS-002", result)

    assert result.status is contracts.AgentRunStatus.NOT_FOUND
    assert result.tool_calls[0].status is contracts.ToolStatus.OK
    assert result.tool_calls[1].status is contracts.ToolStatus.NO_RESULTS
    assert result.product_codes == []


@pytest.mark.report_case("TC-MEMORY-004")
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
        reporting.request("TC-MEMORY-004"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(
        "TC-MEMORY-004",
        result,
        memory_operation="delete",
        memory_keys=[contracts.MemoryKey.BUDGET_RUB],
    )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert [(fact.key, fact.value) for fact in memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT)] == [
        (contracts.MemoryKey.PREFERRED_BRANDS, "Keenetic")
    ]


@pytest.mark.report_case("TC-MEMORY-005", "TC-USER-001")
def test_agent_clears_all_memory_only_for_requesting_user(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, _ = repositories
    memory.save_fact(TEST_USER_ID, contracts.MemoryKey.BUDGET_RUB, "10000", source=agent.MEMORY_SOURCE)
    memory.save_fact(OTHER_USER_ID, contracts.MemoryKey.BUDGET_RUB, "20000", source=agent.MEMORY_SOURCE)
    runner, _ = create_agent_runner(repositories, [{"actions": ["clear_memory"]}])

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request("TC-MEMORY-005"),
        session_id=TEST_SESSION_ID,
    )
    for case_id in ("TC-MEMORY-005", "TC-USER-001"):
        reporting.record_agent_response(
            case_id,
            result,
            memory_operation="clear",
        )

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert memory.get_relevant(TEST_USER_ID, limit=MEMORY_LIMIT) == []
    assert [fact.value for fact in memory.get_relevant(OTHER_USER_ID, limit=MEMORY_LIMIT)] == ["20000"]


@pytest.mark.parametrize(
    ("case_id", "category", "answer"),
    NO_RESULT_CASES,
)
def test_agent_returns_not_found_without_inventing_product(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    case_id: str,
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
        reporting.request(case_id),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(case_id, result)

    assert result.status is contracts.AgentRunStatus.NOT_FOUND
    assert result.product_codes == []
    assert result.tool_calls[0].status is contracts.ToolStatus.NO_RESULTS


@pytest.mark.report_case("TC-GROUNDING-001")
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
        reporting.request("TC-GROUNDING-001"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-GROUNDING-001", result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.errors[0].code is contracts.ErrorCode.INVALID_LLM_RESPONSE


@pytest.mark.report_case("TC-SESSION-001")
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
    first_query, second_query = reporting.request("TC-SESSION-001").split(" → ")

    clarification_result: typing.Final = runner.run(TEST_USER_ID, first_query, session_id=TEST_SESSION_ID)
    continuation_result: typing.Final = runner.run(TEST_USER_ID, second_query, session_id=TEST_SESSION_ID)
    reporting.record_custom_result(
        "TC-SESSION-001",
        {
            "status": continuation_result.status.value,
            "first_answer": clarification_result.answer,
            "second_answer": continuation_result.answer,
            "history_used": first_query in chat_client.calls[1][1],
        },
    )

    assert clarification_result.status is contracts.AgentRunStatus.NEEDS_INPUT
    assert clarification_result.tool_calls == []
    assert first_query in chat_client.calls[1][1]
    assert "Какой у вас бюджет?" in chat_client.calls[1][1]

    runner.end_session(TEST_SESSION_ID)
    runner.run(TEST_USER_ID, "Новый вопрос", session_id=TEST_SESSION_ID)

    assert first_query not in chat_client.calls[2][1]


@pytest.mark.report_case("TC-SESSION-003")
def test_agent_records_controlled_error_as_paired_session_response(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    first_query, second_query = reporting.request("TC-SESSION-003").split(" → ")
    memory, catalog = repositories
    chat_client: typing.Final = FailOnceChatClient({"actions": ["answer"], "response_message": "Диалог продолжен."})
    runner: typing.Final = agent.AgentRunner(
        memory=memory,
        llm=service.LLMService(chat_client=chat_client),
        tool_registry=create_tool_registry(catalog),
    )

    failed_result: typing.Final = runner.run(TEST_USER_ID, first_query, session_id=TEST_SESSION_ID)
    result: typing.Final = runner.run(TEST_USER_ID, second_query, session_id=TEST_SESSION_ID)
    paired_history: typing.Final = (
        first_query in chat_client.calls[1][1] and failed_result.answer in chat_client.calls[1][1]
    )
    reporting.record_custom_result(
        "TC-SESSION-003",
        {
            "status": result.status.value,
            "previous_status": failed_result.status.value,
            "paired_history": paired_history,
            "answer": result.answer,
        },
    )

    assert failed_result.status is contracts.AgentRunStatus.FAILED
    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert paired_history


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


@pytest.mark.parametrize(("case_id", "invalid_price"), INVALID_CATALOG_PRICES)
def test_agent_handles_invalid_catalog_filter_from_llm(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    caplog: pytest.LogCaptureFixture,
    case_id: str,
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

    with caplog.at_level(logging.INFO, logger="ai_agent.agent"):
        result: typing.Final = runner.run(
            TEST_USER_ID,
            reporting.request(case_id),
            session_id=TEST_SESSION_ID,
        )
    reporting.record_agent_response(case_id, result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls == []
    assert result.errors[0].code is contracts.ErrorCode.INVALID_LLM_RESPONSE
    request_failed_log: typing.Final = next(
        record.message for record in caplog.records if record.message.startswith(REQUEST_FAILED_EVENT)
    )
    assert LLM_COMPONENT_FIELD in request_failed_log
    assert INVALID_LLM_ERROR_CODE_FIELD in request_failed_log
    assert INVALID_LLM_ERROR_TYPE_FIELD in request_failed_log
    assert "reason=" in request_failed_log


@pytest.mark.report_case("TC-TOOL-ERROR-001")
def test_agent_handles_catalog_tool_failure(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    tmp_path: pathlib.Path,
) -> None:
    memory, _ = repositories
    chat_client: typing.Final = ScriptedChatClient(
        [
            {"actions": ["search_products"]},
            {"filters": {"category": "router"}},
        ]
    )
    missing_catalog: typing.Final = catalog_repository.ProductsRepository(database_path=tmp_path / "missing.db")
    runner: typing.Final = agent.AgentRunner(
        memory=memory,
        llm=service.LLMService(chat_client=chat_client),
        tool_registry=create_tool_registry(missing_catalog),
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request("TC-TOOL-ERROR-001"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-TOOL-ERROR-001", result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls[0].status is contracts.ToolStatus.ERROR
    assert result.errors[0].code is contracts.ErrorCode.CATALOG_UNAVAILABLE


@pytest.mark.report_case("TC-LLM-ERROR-001")
def test_agent_handles_unavailable_llm(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
) -> None:
    memory, catalog = repositories
    runner: typing.Final = agent.AgentRunner(
        memory=memory,
        llm=service.LLMService(chat_client=UnavailableChatClient()),
        tool_registry=create_tool_registry(catalog),
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request("TC-LLM-ERROR-001"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-LLM-ERROR-001", result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.tool_calls == []
    assert result.errors[0].code is contracts.ErrorCode.LLM_UNAVAILABLE


@pytest.mark.report_case("TC-MEMORY-ERROR-002")
def test_agent_logs_memory_database_failure_with_traceback(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    tmp_path: pathlib.Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _, catalog = repositories
    failing_memory: typing.Final = UnavailableMemoryRepository(database_path=tmp_path / "broken-memory.db")
    runner: typing.Final = agent.AgentRunner(
        memory=failing_memory,
        llm=service.LLMService(chat_client=ScriptedChatClient([])),
        tool_registry=create_tool_registry(catalog),
    )

    with caplog.at_level(logging.DEBUG, logger="ai_agent.agent"):
        result: typing.Final = runner.run(
            TEST_USER_ID,
            reporting.request("TC-MEMORY-ERROR-002"),
            session_id=TEST_SESSION_ID,
        )
    reporting.record_agent_response("TC-MEMORY-ERROR-002", result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.errors[0].code is contracts.ErrorCode.MEMORY_UNAVAILABLE
    request_failed_log: typing.Final = next(
        record for record in caplog.records if record.message.startswith(REQUEST_FAILED_EVENT)
    )
    assert 'component="memory"' in request_failed_log.message
    assert 'error_type="OperationalError"' in request_failed_log.message
    assert MEMORY_DATABASE_ERROR in request_failed_log.message
    assert request_failed_log.exc_info is None
    debug_log: typing.Final = next(record for record in caplog.records if record.message == "memory operation failed")
    assert debug_log.exc_info is not None


@pytest.mark.parametrize(
    ("case_id", "answer"),
    KNOWLEDGE_CASES,
)
def test_agent_answers_knowledge_questions_using_rag(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    case_id: str,
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

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request(case_id),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(case_id, result)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.answer == answer
    assert result.sources == [KNOWLEDGE_SOURCE]
    assert [call.tool_name for call in result.tool_calls] == [search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME]


@pytest.mark.parametrize(
    ("case_id", "filters", "product_code"),
    CATALOG_CASES,
)
def test_agent_searches_catalog_for_different_product_categories(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    case_id: str,
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

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request(case_id),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(case_id, result)

    assert result.status is contracts.AgentRunStatus.COMPLETED
    assert result.product_codes == [product_code]
    assert [call.tool_name for call in result.tool_calls] == [search_products.SEARCH_PRODUCTS_TOOL_NAME]


@pytest.mark.parametrize(
    ("case_id", "response_message", "expected_status"),
    DIRECT_DIALOGUE_CASES,
)
def test_agent_handles_direct_dialogue_branches(
    repositories: tuple[memory_repository.MemoryRepository, catalog_repository.ProductsRepository],
    case_id: str,
    response_message: str,
    expected_status: contracts.AgentRunStatus,
) -> None:
    action: typing.Final = "clarify" if expected_status is contracts.AgentRunStatus.NEEDS_INPUT else "unsupported"
    runner, _ = create_agent_runner(
        repositories,
        [{"actions": [action], "response_message": response_message}],
    )

    result: typing.Final = runner.run(
        TEST_USER_ID,
        reporting.request(case_id),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response(case_id, result)

    assert result.status is expected_status
    assert result.answer == response_message
    assert result.tool_calls == []


@pytest.mark.report_case("TC-SESSION-002")
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
        reporting.request("TC-SESSION-002"),
        session_id=TEST_SESSION_ID,
    )
    reporting.record_agent_response("TC-SESSION-002", result)

    assert result.status is contracts.AgentRunStatus.FAILED
    assert result.errors[0].code is contracts.ErrorCode.INVALID_INPUT
    assert result.session_id == TEST_SESSION_ID
