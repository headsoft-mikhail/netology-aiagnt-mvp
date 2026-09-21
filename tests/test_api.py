import asyncio
import logging
import pathlib
import typing

import fastapi
import httpx
import modern_di
import modern_di.providers
import pytest

from ai_agent import contracts
from ai_agent.agent import AgentRunner
from ai_agent.service import app as service_app
from ai_agent.service import dependencies, routes

TEST_USER_ID: typing.Final = "api-test-user"
TEST_SESSION_ID: typing.Final = "api-test-session"
TEST_MESSAGE: typing.Final = "Как выбрать роутер?"
TEST_ANSWER: typing.Final = "Выберите роутер с гигабитным WAN-портом."
TEST_SOURCE: typing.Final = "router_selection.txt"
TEST_DOCUMENT_ID: typing.Final = "router-selection-document"
TEST_CHUNK_ID: typing.Final = "router-selection-chunk"
LLM_ERROR_MESSAGE: typing.Final = "Не удалось получить корректный ответ языковой модели."


class FakeAgentRunner:
    def __init__(
        self,
        *,
        status: contracts.AgentRunStatus = contracts.AgentRunStatus.COMPLETED,
        errors: list[contracts.ToolError] | None = None,
    ) -> None:
        self.calls: list[tuple[str, str, str | None]] = []
        self.status = status
        self.errors = errors or []

    def run(
        self,
        user_id: str,
        user_message: str,
        *,
        session_id: str | None = None,
    ) -> contracts.AgentResponse:
        self.calls.append((user_id, user_message, session_id))
        return contracts.AgentResponse(
            status=self.status,
            answer=LLM_ERROR_MESSAGE if self.errors else TEST_ANSWER,
            sources=[TEST_SOURCE] if not self.errors else [],
            citations=[
                contracts.Citation(
                    source=TEST_SOURCE,
                    document_id=TEST_DOCUMENT_ID,
                    chunk_id=TEST_CHUNK_ID,
                    score=0.95,
                )
            ]
            if not self.errors
            else [],
            request_id="api-test-request",
            session_id=session_id or TEST_SESSION_ID,
            errors=self.errors,
        )


def test_health_ready_and_chat_endpoints(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    configure_ready_dependencies(tmp_path, monkeypatch)
    caplog.set_level(logging.DEBUG, logger=routes.__name__)

    fake_agent: typing.Final = FakeAgentRunner()
    application: typing.Final = service_app.create_application(
        create_test_container(typing.cast(AgentRunner, fake_agent))
    )

    health_response, ready_response, chat_response = asyncio.run(
        request_application(
            application,
            include_session_id=True,
        )
    )

    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok"}
    assert ready_response.status_code == 200
    assert ready_response.json()["status"] == "ready"
    assert chat_response.status_code == 200
    assert chat_response.json()["answer"] == TEST_ANSWER
    assert chat_response.json()["citations"][0]["chunk_id"] == TEST_CHUNK_ID
    assert fake_agent.calls == [(TEST_USER_ID, TEST_MESSAGE, TEST_SESSION_ID)]
    trace_log: typing.Final = next(record for record in caplog.records if "event=agent_trace" in record.message)
    assert TEST_ANSWER in trace_log.message
    assert TEST_SESSION_ID in trace_log.message


def test_chat_reports_llm_unavailable_as_service_unavailable(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    configure_ready_dependencies(tmp_path, monkeypatch)
    llm_error: typing.Final = contracts.ToolError(
        code=contracts.ErrorCode.LLM_UNAVAILABLE,
        message=LLM_ERROR_MESSAGE,
    )
    fake_agent: typing.Final = FakeAgentRunner(
        status=contracts.AgentRunStatus.FAILED,
        errors=[llm_error],
    )
    application: typing.Final = service_app.create_application(
        create_test_container(typing.cast(AgentRunner, fake_agent))
    )

    _, _, chat_response = asyncio.run(
        request_application(
            application,
            include_session_id=True,
        )
    )

    assert chat_response.status_code == 503
    assert chat_response.json()["status"] == contracts.AgentRunStatus.FAILED
    assert chat_response.json()["errors"] == [llm_error.model_dump(mode="json")]


def configure_ready_dependencies(
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path: typing.Final = tmp_path / "catalog.db"
    vector_store_path: typing.Final = tmp_path / "vector-store"
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    catalog_path.touch(exist_ok=True)
    vector_store_path.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(routes.catalog_config, "database_path", catalog_path)
    monkeypatch.setattr(routes.rag_retrieval_config, "vector_store_path", vector_store_path)

    def vector_store_is_ready(config: object) -> bool:
        del config
        return True

    monkeypatch.setattr(routes.retrieval, "is_vector_store_ready", vector_store_is_ready)


def test_unprepared_application_reports_not_ready() -> None:
    application: typing.Final = service_app.create_application(create_test_container(None))

    _, ready_response, chat_response = asyncio.run(
        request_application(
            application,
            include_session_id=False,
        )
    )

    assert ready_response.status_code == 503
    assert ready_response.json()["agent_ready"] is False
    assert chat_response.status_code == 503
    assert chat_response.json() == {"detail": routes.SERVICE_UNAVAILABLE_MESSAGE}


def create_test_container(agent_runner: AgentRunner | None) -> modern_di.Container:
    def create_test_runtime() -> dependencies.AgentRuntime:
        return dependencies.AgentRuntime(agent_runner=agent_runner)

    class TestDependencies(modern_di.Group):
        agent_runtime = modern_di.providers.Factory(
            create_test_runtime,
            scope=modern_di.Scope.APP,
            cache=True,
        )

    return modern_di.Container(groups=[TestDependencies])


async def request_application(
    application: fastapi.FastAPI,
    *,
    include_session_id: bool,
) -> tuple[httpx.Response, httpx.Response, httpx.Response]:
    payload: typing.Final[dict[str, str]] = {
        "user_id": TEST_USER_ID,
        "message": TEST_MESSAGE,
    }
    if include_session_id:
        payload["session_id"] = TEST_SESSION_ID

    transport: typing.Final = httpx.ASGITransport(app=application)
    async with (
        application.router.lifespan_context(application),
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
    ):
        return (
            await client.get("/health"),
            await client.get("/ready"),
            await client.post("/chat", json=payload),
        )
