import dataclasses
import logging
import typing
import uuid

from ai_agent import contracts, state
from ai_agent import logging as agent_logging
from ai_agent.handlers import catalog, direct, final_answer, knowledge, memory, planning, runtime
from ai_agent.llm import models as llm_models
from ai_agent.llm.service import LLMService
from ai_agent.memory import session
from ai_agent.memory.repository import MemoryRepository
from ai_agent.tools import registry

MEMORY_SOURCE: typing.Final = memory.MEMORY_SOURCE
SESSION_OWNER_ERROR_MESSAGE: typing.Final = "Указанная сессия принадлежит другому пользователю."
LOGGER_OBJ: typing.Final = logging.getLogger(__name__)


@dataclasses.dataclass(kw_only=True, slots=True)
class AgentRunner:
    """Coordinate one agent request while handlers perform each domain step."""

    memory: MemoryRepository
    llm: LLMService
    tool_registry: registry.ToolRegistry
    memory_limit: int = 10
    _sessions: dict[str, session.SessionMemory] = dataclasses.field(default_factory=dict)
    _memory_handler: memory.MemoryHandler = dataclasses.field(init=False)
    _planning_handler: planning.PlanningHandler = dataclasses.field(init=False)
    _direct_handler: direct.DirectActionHandler = dataclasses.field(init=False)
    _knowledge_handler: knowledge.KnowledgeSearchHandler = dataclasses.field(init=False)
    _catalog_handler: catalog.CatalogSearchHandler = dataclasses.field(init=False)
    _final_answer_handler: final_answer.FinalAnswerHandler = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        self._memory_handler = memory.MemoryHandler(repository=self.memory, limit=self.memory_limit)
        self._planning_handler = planning.PlanningHandler(llm=self.llm)
        self._direct_handler = direct.DirectActionHandler(memory_handler=self._memory_handler)
        self._knowledge_handler = knowledge.KnowledgeSearchHandler(tool_registry=self.tool_registry)
        self._catalog_handler = catalog.CatalogSearchHandler(
            llm=self.llm,
            tool_registry=self.tool_registry,
        )
        self._final_answer_handler = final_answer.FinalAnswerHandler(llm=self.llm)

    def run(
        self,
        user_id: str,
        user_message: str,
        *,
        session_id: str | None = None,
    ) -> contracts.AgentResponse:
        request: typing.Final = contracts.AgentRequest(
            user_id=user_id,
            query=user_message,
            session_id=session_id or str(uuid.uuid4()),
        )
        agent_state: typing.Final = state.AgentState(
            request=request,
            status=contracts.AgentRunStatus.RUNNING,
        )
        run_context: typing.Final = runtime.RunContext(
            agent_state=agent_state,
            request_id=str(uuid.uuid4()),
            logger=LOGGER_OBJ,
            llm_model_name=self.llm.model_name,
        )
        run_context.log(
            "request_accepted",
            user_id=request.user_id,
            query_length=len(request.query),
        )

        try:
            current_session: typing.Final = self._get_session(request)
        except ValueError:
            return run_context.error_response(
                contracts.ErrorCode.INVALID_INPUT,
                SESSION_OWNER_ERROR_MESSAGE,
                component="session",
            )
        current_session.add("user", request.query)

        memory_response: typing.Final = self._memory_handler.load(run_context)
        if memory_response is not None:
            return memory_response

        planning_result: typing.Final = self._planning_handler.handle(
            run_context,
            current_session.messages[:-1],
        )
        if isinstance(planning_result, contracts.AgentResponse):
            return planning_result
        plan: typing.Final = typing.cast(llm_models.AgentPlan, planning_result)

        direct_response: typing.Final = self._direct_handler.handle(plan, run_context)
        if direct_response is not None:
            current_session.add("assistant", direct_response.answer)
            return direct_response

        if contracts.AgentAction.SEARCH_KNOWLEDGE_BASE in plan.actions:
            knowledge_error: typing.Final = self._knowledge_handler.handle(plan, run_context)
            if knowledge_error is not None:
                return knowledge_error

        if contracts.AgentAction.SEARCH_PRODUCTS in plan.actions:
            catalog_error: typing.Final = self._catalog_handler.handle(run_context)
            if catalog_error is not None:
                return catalog_error

        response: typing.Final = self._final_answer_handler.handle(run_context)
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
