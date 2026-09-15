import dataclasses
import json
import logging
import time
import typing
import uuid

from ai_agent import contracts, state
from ai_agent import logging as agent_logging
from ai_agent.catalog import models as catalog_models
from ai_agent.catalog.repository import ProductsRepository
from ai_agent.llm import client as llm_client
from ai_agent.llm import models as llm_models
from ai_agent.llm.service import LLMService
from ai_agent.memory import session
from ai_agent.memory.repository import MemoryRepository
from ai_agent.rag.context import ContextBuilder
from ai_agent.tools import search_knowledge_base, search_products

MEMORY_SOURCE: typing.Final = "explicit_user_request"
MEMORY_SAVED_MESSAGE: typing.Final = "Предпочтение сохранено."
MEMORY_DELETED_MESSAGE: typing.Final = "Предпочтение удалено."
LLM_ERROR_MESSAGE: typing.Final = "Не удалось получить корректный ответ языковой модели."
MEMORY_ERROR_MESSAGE: typing.Final = "Не удалось обратиться к памяти пользователя."
LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class AgentRunner:
    memory: MemoryRepository
    llm: LLMService
    catalog: ProductsRepository
    retrieval_client: search_knowledge_base.RetrievalClientProtocol
    context_builder: ContextBuilder
    memory_limit: int = 10
    _sessions: dict[str, session.SessionMemory] = dataclasses.field(default_factory=dict)

    def run(
        self,
        user_id: str,
        user_message: str,
        *,
        session_id: str | None = None,
    ) -> contracts.AgentResponse:
        actual_session_id: typing.Final = session_id or str(uuid.uuid4())
        request_id: typing.Final = str(uuid.uuid4())
        request: typing.Final = contracts.AgentRequest(
            user_id=user_id,
            query=user_message,
            session_id=actual_session_id,
        )
        agent_state: typing.Final = state.AgentState(
            request=request,
            status=contracts.AgentRunStatus.RUNNING,
        )
        self._log(
            "request_accepted",
            agent_state,
            request_id,
            user_id=request.user_id,
            query_length=len(request.query),
        )
        try:
            current_session: typing.Final = self._get_session(request)
        except ValueError:
            return self._error_response(
                agent_state,
                request_id,
                contracts.ErrorCode.INVALID_INPUT,
                "Указанная сессия принадлежит другому пользователю.",
            )
        current_session.add("user", request.query)

        memory_started_at: typing.Final = time.perf_counter()
        self._log("memory_load_started", agent_state, request_id, user_id=request.user_id)
        try:
            agent_state.relevant_memory = self.memory.get_relevant(
                request.user_id,
                limit=self.memory_limit,
            )
        except Exception:  # noqa: BLE001
            return self._error_response(
                agent_state,
                request_id,
                contracts.ErrorCode.MEMORY_UNAVAILABLE,
                MEMORY_ERROR_MESSAGE,
            )
        self._log(
            "memory_load_completed",
            agent_state,
            request_id,
            records_count=len(agent_state.relevant_memory),
            duration_ms=_duration_ms(memory_started_at),
        )

        planning_started_at: typing.Final = time.perf_counter()
        self._log(
            "llm_request_started",
            agent_state,
            request_id,
            phase="planning",
            model=self.llm.model_name,
        )
        try:
            plan: typing.Final = self.llm.plan(
                request.query,
                agent_state.relevant_memory,
                current_session.messages[:-1],
            )
        except llm_client.LLMError as error:
            return self._llm_error_response(agent_state, request_id, error)
        self._log(
            "llm_request_completed",
            agent_state,
            request_id,
            phase="planning",
            model=self.llm.model_name,
            duration_ms=_duration_ms(planning_started_at),
        )
        agent_state.selected_actions = plan.actions
        self._log(
            "actions_selected",
            agent_state,
            request_id,
            actions=[action.value for action in plan.actions],
        )
        if not any(
            action
            in {
                contracts.AgentAction.UPDATE_MEMORY,
                contracts.AgentAction.DELETE_MEMORY,
                contracts.AgentAction.CLEAR_MEMORY,
            }
            for action in plan.actions
        ):
            self._log(
                "memory_update_skipped",
                agent_state,
                request_id,
                reason="no_explicit_memory_action",
            )

        direct_response: typing.Final = self._handle_direct_action(
            agent_state,
            plan,
            request_id,
        )
        if direct_response is not None:
            current_session.add("assistant", direct_response.answer)
            return direct_response

        if contracts.AgentAction.SEARCH_KNOWLEDGE_BASE in plan.actions:
            knowledge_input: typing.Final[dict[str, object]] = {"query": plan.knowledge_query}
            safe_knowledge_arguments: typing.Final[dict[str, object]] = {
                "query_length": len(typing.cast(str, plan.knowledge_query)),
            }
            tool_started_at: typing.Final = time.perf_counter()
            self._log(
                "tool_call_started",
                agent_state,
                request_id,
                tool=search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
                arguments=safe_knowledge_arguments,
            )
            agent_state.knowledge_result = search_knowledge_base.search_knowledge_base(
                search_knowledge_base.KnowledgeSearchInput(query=typing.cast(str, plan.knowledge_query)),
                self.retrieval_client,
                self.context_builder,
            )
            agent_state.tool_calls.append(
                self._tool_trace(
                    search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
                    knowledge_input,
                    agent_state.knowledge_result,
                )
            )
            self._log(
                "tool_call_completed",
                agent_state,
                request_id,
                level=(
                    logging.WARNING
                    if agent_state.knowledge_result.status is contracts.ToolStatus.NO_RESULTS
                    else logging.INFO
                ),
                tool=search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
                status=agent_state.knowledge_result.status.value,
                duration_ms=_duration_ms(tool_started_at),
                results_count=len(agent_state.knowledge_result.fragments),
            )
            self._log(
                "rag_retrieval_completed",
                agent_state,
                request_id,
                fragments_count=len(agent_state.knowledge_result.fragments),
                sources=agent_state.knowledge_result.sources,
                scores=[round(fragment.score, 3) for fragment in agent_state.knowledge_result.fragments],
            )
            if agent_state.knowledge_result.status is contracts.ToolStatus.ERROR:
                return self._tool_error_response(agent_state, request_id, agent_state.knowledge_result.error)

        if contracts.AgentAction.SEARCH_PRODUCTS in plan.actions:
            filters_started_at: typing.Final = time.perf_counter()
            self._log(
                "llm_request_started",
                agent_state,
                request_id,
                phase="catalog_filters",
                model=self.llm.model_name,
            )
            try:
                agent_state.catalog_filters = self.llm.create_product_search(
                    request.query,
                    agent_state.relevant_memory,
                    agent_state.knowledge_result.context if agent_state.knowledge_result else "",
                )
            except llm_client.LLMError as error:
                return self._llm_error_response(agent_state, request_id, error)
            self._log(
                "llm_request_completed",
                agent_state,
                request_id,
                phase="catalog_filters",
                model=self.llm.model_name,
                duration_ms=_duration_ms(filters_started_at),
            )
            catalog_arguments: typing.Final = agent_state.catalog_filters.model_dump(mode="json")
            self._log(
                "catalog_filters_validated",
                agent_state,
                request_id,
                filters=catalog_arguments,
            )
            catalog_tool_started_at: typing.Final = time.perf_counter()
            self._log(
                "tool_call_started",
                agent_state,
                request_id,
                tool=search_products.SEARCH_PRODUCTS_TOOL_NAME,
                arguments=catalog_arguments,
            )
            agent_state.catalog_result = search_products.search_products(
                agent_state.catalog_filters,
                self.catalog,
            )
            agent_state.tool_calls.append(
                self._tool_trace(
                    search_products.SEARCH_PRODUCTS_TOOL_NAME,
                    catalog_arguments,
                    agent_state.catalog_result,
                )
            )
            self._log(
                "tool_call_completed",
                agent_state,
                request_id,
                level=(
                    logging.WARNING
                    if agent_state.catalog_result.status is contracts.ToolStatus.NO_RESULTS
                    else logging.INFO
                ),
                tool=search_products.SEARCH_PRODUCTS_TOOL_NAME,
                status=agent_state.catalog_result.status.value,
                duration_ms=_duration_ms(catalog_tool_started_at),
                results_count=agent_state.catalog_result.total,
            )
            if agent_state.catalog_result.status is contracts.ToolStatus.ERROR:
                return self._tool_error_response(agent_state, request_id, agent_state.catalog_result.error)

        agent_state.assembled_context = _json_dump(
            {
                "memory": [fact.model_dump(mode="json") for fact in agent_state.relevant_memory],
                "knowledge": (
                    agent_state.knowledge_result.model_dump(mode="json") if agent_state.knowledge_result else None
                ),
                "catalog": agent_state.catalog_result.model_dump(mode="json") if agent_state.catalog_result else None,
            }
        )
        self._log(
            "context_assembled",
            agent_state,
            request_id,
            memory_records=len(agent_state.relevant_memory),
            knowledge_fragments=(len(agent_state.knowledge_result.fragments) if agent_state.knowledge_result else 0),
            products=(agent_state.catalog_result.total if agent_state.catalog_result else 0),
            context_chars=len(agent_state.assembled_context),
        )
        final_answer_started_at: typing.Final = time.perf_counter()
        self._log(
            "llm_request_started",
            agent_state,
            request_id,
            phase="final_answer",
            model=self.llm.model_name,
        )
        try:
            draft: typing.Final = self.llm.create_final_answer(
                request.query,
                agent_state.relevant_memory,
                agent_state.knowledge_result,
                agent_state.catalog_result,
            )
            self._validate_product_codes(draft.product_codes, agent_state.catalog_result)
        except llm_client.LLMError as error:
            return self._llm_error_response(agent_state, request_id, error)
        self._log(
            "llm_request_completed",
            agent_state,
            request_id,
            phase="final_answer",
            model=self.llm.model_name,
            duration_ms=_duration_ms(final_answer_started_at),
        )

        status: typing.Final = (
            contracts.AgentRunStatus.NOT_FOUND
            if self._has_no_results(agent_state)
            else contracts.AgentRunStatus.COMPLETED
        )
        response: typing.Final = self._response(
            agent_state,
            request_id,
            status,
            draft.answer,
            product_codes=draft.product_codes,
        )
        current_session.add("assistant", response.answer)
        return response

    def end_session(self, session_id: str) -> None:
        current_session: typing.Final = self._sessions.pop(session_id, None)
        if current_session is not None:
            current_session.clear()
            agent_logging.event(
                LOGGER_OBJ,
                logging.INFO,
                "session_memory_cleared",
                session_id=session_id,
                user_id=current_session.user_id,
            )

    def _get_session(self, request: contracts.AgentRequest) -> session.SessionMemory:
        session_id: typing.Final = typing.cast(str, request.session_id)
        current_session: typing.Final = self._sessions.get(session_id)
        if current_session is not None:
            if current_session.user_id != request.user_id:
                raise ValueError("Session belongs to another user.")
            return current_session
        created_session: typing.Final = session.SessionMemory(
            user_id=request.user_id,
            session_id=session_id,
        )
        self._sessions[session_id] = created_session
        return created_session

    def _handle_direct_action(
        self,
        agent_state: state.AgentState,
        plan: llm_models.AgentPlan,
        request_id: str,
    ) -> contracts.AgentResponse | None:
        action: typing.Final = plan.actions[0]
        if action is contracts.AgentAction.UPDATE_MEMORY:
            memory_started_at: typing.Final = time.perf_counter()
            self._log("memory_update_started", agent_state, request_id, operation="save")
            try:
                update: typing.Final = typing.cast(contracts.MemoryUpdate, plan.memory_update)
                saved_fact: typing.Final = self.memory.save_fact(
                    agent_state.request.user_id,
                    update.key,
                    update.value,
                    source=MEMORY_SOURCE,
                    session_id=typing.cast(str, agent_state.request.session_id),
                )
                agent_state.relevant_memory = [saved_fact]
            except ValueError, TypeError:
                return self._error_response(
                    agent_state,
                    request_id,
                    contracts.ErrorCode.INVALID_INPUT,
                    "Эти данные нельзя сохранить в памяти.",
                )
            except Exception:  # noqa: BLE001
                return self._error_response(
                    agent_state,
                    request_id,
                    contracts.ErrorCode.MEMORY_UNAVAILABLE,
                    MEMORY_ERROR_MESSAGE,
                )
            self._log(
                "memory_update_completed",
                agent_state,
                request_id,
                operation="save",
                key=saved_fact.key.value,
                duration_ms=_duration_ms(memory_started_at),
            )
            return self._response(
                agent_state,
                request_id,
                contracts.AgentRunStatus.COMPLETED,
                MEMORY_SAVED_MESSAGE,
            )
        if action is contracts.AgentAction.DELETE_MEMORY:
            delete_started_at: typing.Final = time.perf_counter()
            self._log("memory_update_started", agent_state, request_id, operation="delete")
            try:
                self.memory.delete_fact(
                    agent_state.request.user_id,
                    typing.cast(contracts.MemoryKey, plan.memory_key),
                )
            except Exception:  # noqa: BLE001
                return self._error_response(
                    agent_state,
                    request_id,
                    contracts.ErrorCode.MEMORY_UNAVAILABLE,
                    MEMORY_ERROR_MESSAGE,
                )
            self._log(
                "memory_update_completed",
                agent_state,
                request_id,
                operation="delete",
                key=typing.cast(contracts.MemoryKey, plan.memory_key).value,
                duration_ms=_duration_ms(delete_started_at),
            )
            return self._response(
                agent_state,
                request_id,
                contracts.AgentRunStatus.COMPLETED,
                MEMORY_DELETED_MESSAGE,
            )
        if action is contracts.AgentAction.CLEAR_MEMORY:
            clear_started_at: typing.Final = time.perf_counter()
            self._log("memory_update_started", agent_state, request_id, operation="clear")
            try:
                self.memory.clear_user(agent_state.request.user_id)
            except Exception:  # noqa: BLE001
                return self._error_response(
                    agent_state,
                    request_id,
                    contracts.ErrorCode.MEMORY_UNAVAILABLE,
                    MEMORY_ERROR_MESSAGE,
                )
            self._log(
                "memory_update_completed",
                agent_state,
                request_id,
                operation="clear",
                duration_ms=_duration_ms(clear_started_at),
            )
            return self._response(
                agent_state,
                request_id,
                contracts.AgentRunStatus.COMPLETED,
                "Все предпочтения пользователя удалены.",
            )
        if action is contracts.AgentAction.CLARIFY:
            return self._response(
                agent_state,
                request_id,
                contracts.AgentRunStatus.NEEDS_INPUT,
                typing.cast(str, plan.response_message),
            )
        if action in (contracts.AgentAction.ANSWER, contracts.AgentAction.UNSUPPORTED):
            return self._response(
                agent_state,
                request_id,
                contracts.AgentRunStatus.COMPLETED,
                typing.cast(str, plan.response_message),
            )
        return None

    @staticmethod
    def _tool_trace(
        tool_name: str,
        tool_input: dict[str, object],
        tool_result: contracts.KnowledgeSearchResult | catalog_models.ProductSearchResult,
    ) -> contracts.ToolCallTrace:
        return contracts.ToolCallTrace(
            tool_name=tool_name,
            status=tool_result.status,
            input_json=_json_dump(tool_input),
            output_json=tool_result.model_dump_json(),
            error=tool_result.error,
        )

    @staticmethod
    def _validate_product_codes(
        product_codes: list[str],
        catalog_result: catalog_models.ProductSearchResult | None,
    ) -> None:
        available_product_codes: typing.Final = (
            {product.product_code for product in catalog_result.products} if catalog_result else set()
        )
        if not set(product_codes).issubset(available_product_codes):
            raise llm_client.InvalidLLMResponseError("LLM mentioned a product outside the catalog result.")

    @staticmethod
    def _has_no_results(agent_state: state.AgentState) -> bool:
        results: typing.Final = (agent_state.knowledge_result, agent_state.catalog_result)
        return any(result is not None and result.status is contracts.ToolStatus.NO_RESULTS for result in results)

    def _tool_error_response(
        self,
        agent_state: state.AgentState,
        request_id: str,
        error: contracts.ToolError | None,
    ) -> contracts.AgentResponse:
        actual_error: typing.Final = error or contracts.ToolError(
            code=contracts.ErrorCode.INTERNAL_ERROR,
            message="Компонент завершился с неизвестной ошибкой.",
        )
        return self._error_response(agent_state, request_id, actual_error.code, actual_error.message)

    def _llm_error_response(
        self,
        agent_state: state.AgentState,
        request_id: str,
        error: llm_client.LLMError,
    ) -> contracts.AgentResponse:
        LOGGER_OBJ.debug("LLM operation failed")
        self._log(
            "llm_request_failed",
            agent_state,
            request_id,
            model=self.llm.model_name,
            error_type=type(error).__name__,
        )
        code: typing.Final = (
            contracts.ErrorCode.LLM_UNAVAILABLE
            if isinstance(error, llm_client.LLMUnavailableError)
            else contracts.ErrorCode.INVALID_LLM_RESPONSE
        )
        return self._error_response(agent_state, request_id, code, LLM_ERROR_MESSAGE)

    def _error_response(
        self,
        agent_state: state.AgentState,
        request_id: str,
        code: contracts.ErrorCode,
        message: str,
    ) -> contracts.AgentResponse:
        agent_state.errors.append(contracts.ToolError(code=code, message=message))
        agent_logging.event(
            LOGGER_OBJ,
            logging.ERROR,
            "request_failed",
            request_id=request_id,
            session_id=agent_state.request.session_id,
            component=code.value,
            error_type=code.value,
            message=message,
        )
        return self._response(
            agent_state,
            request_id,
            contracts.AgentRunStatus.FAILED,
            message,
        )

    def _response(
        self,
        agent_state: state.AgentState,
        request_id: str,
        status: contracts.AgentRunStatus,
        answer: str,
        *,
        product_codes: list[str] | None = None,
    ) -> contracts.AgentResponse:
        sources: typing.Final = agent_state.knowledge_result.sources if agent_state.knowledge_result else []
        response: typing.Final = contracts.AgentResponse(
            status=status,
            answer=answer,
            sources=sources,
            product_codes=product_codes or [],
            tool_calls=agent_state.tool_calls,
            memory_used=[fact.id for fact in agent_state.relevant_memory],
            request_id=request_id,
            session_id=typing.cast(str, agent_state.request.session_id),
            errors=agent_state.errors,
        )
        agent_state.status = status
        agent_state.final_response = response
        self._log(
            "request_completed",
            agent_state,
            request_id,
            level=(
                logging.WARNING
                if status in {contracts.AgentRunStatus.NEEDS_INPUT, contracts.AgentRunStatus.NOT_FOUND}
                else logging.INFO
            ),
            status=status.value,
            tool_calls=len(response.tool_calls),
            products=len(response.product_codes),
            errors=len(response.errors),
        )
        return response

    @staticmethod
    def _log(
        event_name: str,
        agent_state: state.AgentState,
        request_id: str,
        *,
        level: int = logging.INFO,
        **fields: object,
    ) -> None:
        agent_logging.event(
            LOGGER_OBJ,
            level,
            event_name,
            request_id=request_id,
            session_id=agent_state.request.session_id,
            **fields,
        )


def _json_dump(value: object) -> str:
    return json.dumps(value, ensure_ascii=False)


def _duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
