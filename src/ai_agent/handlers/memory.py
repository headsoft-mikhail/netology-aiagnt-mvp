import dataclasses
import time
import typing

from ai_agent import contracts
from ai_agent.handlers import helpers, runtime
from ai_agent.llm import models as llm_models
from ai_agent.memory.repository import MemoryRepository

MEMORY_SOURCE: typing.Final = "explicit_user_request"
MEMORY_SAVED_MESSAGE: typing.Final = "Предпочтение сохранено."
MEMORY_DELETED_MESSAGE: typing.Final = "Предпочтение удалено."
MEMORY_CLEARED_MESSAGE: typing.Final = "Все предпочтения пользователя удалены."
MEMORY_ERROR_MESSAGE: typing.Final = "Не удалось обратиться к памяти пользователя."
INVALID_MEMORY_MESSAGE: typing.Final = (
    "Не удалось сохранить значение. Для бюджета, площади, количества устройств "
    "и скорости тарифа укажите положительное целое число цифрами."
)


@dataclasses.dataclass(kw_only=True, slots=True)
class MemoryHandler:
    repository: MemoryRepository
    limit: int

    @helpers.catch_exception_and_log(
        exceptions=(Exception,),
        component="memory",
        error_code=contracts.ErrorCode.MEMORY_UNAVAILABLE,
        user_message=MEMORY_ERROR_MESSAGE,
    )
    def load(self, context: runtime.RunContext) -> contracts.AgentResponse | None:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_load_started", user_id=context.agent_state.request.user_id)
        context.agent_state.relevant_memory = self.repository.get_relevant(
            context.agent_state.request.user_id,
            limit=self.limit,
        )
        context.log(
            "memory_load_completed",
            records_count=len(context.agent_state.relevant_memory),
            duration_ms=helpers.duration_ms(started_at),
        )
        return None

    def handle_action(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse | None:
        action: typing.Final = plan.actions[0]
        if action is contracts.AgentAction.UPDATE_MEMORY:
            return self._save(context, plan)
        if action is contracts.AgentAction.DELETE_MEMORY:
            return self._delete(context, plan)
        if action is contracts.AgentAction.CLEAR_MEMORY:
            return self._clear(context)
        return None

    @helpers.catch_exception_and_log(
        exceptions=(Exception,),
        component="memory",
        error_code=contracts.ErrorCode.MEMORY_UNAVAILABLE,
        user_message=MEMORY_ERROR_MESSAGE,
    )
    @helpers.catch_exception_and_log(
        exceptions=(ValueError, TypeError),
        component="memory",
        error_code=contracts.ErrorCode.INVALID_INPUT,
        user_message=INVALID_MEMORY_MESSAGE,
        response_status=contracts.AgentRunStatus.NEEDS_INPUT,
    )
    def _save(
        self,
        context: runtime.RunContext,
        plan: llm_models.AgentPlan,
    ) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="save")
        update: typing.Final = typing.cast(contracts.MemoryUpdate, plan.memory_update)
        saved_fact: typing.Final = self.repository.save_fact(
            context.agent_state.request.user_id,
            update.key,
            update.value,
            source=MEMORY_SOURCE,
            session_id=typing.cast(str, context.agent_state.request.session_id),
        )
        context.agent_state.relevant_memory = [saved_fact]
        context.log(
            "memory_update_completed",
            operation="save",
            key=saved_fact.key.value,
            duration_ms=helpers.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_SAVED_MESSAGE)

    @helpers.catch_exception_and_log(
        exceptions=(Exception,),
        component="memory",
        error_code=contracts.ErrorCode.MEMORY_UNAVAILABLE,
        user_message=MEMORY_ERROR_MESSAGE,
    )
    def _delete(
        self,
        context: runtime.RunContext,
        plan: llm_models.AgentPlan,
    ) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="delete")
        memory_key: typing.Final = typing.cast(contracts.MemoryKey, plan.memory_key)
        self.repository.delete_fact(context.agent_state.request.user_id, memory_key)
        context.log(
            "memory_update_completed",
            operation="delete",
            key=memory_key.value,
            duration_ms=helpers.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_DELETED_MESSAGE)

    @helpers.catch_exception_and_log(
        exceptions=(Exception,),
        component="memory",
        error_code=contracts.ErrorCode.MEMORY_UNAVAILABLE,
        user_message=MEMORY_ERROR_MESSAGE,
    )
    def _clear(self, context: runtime.RunContext) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="clear")
        self.repository.clear_user(context.agent_state.request.user_id)
        context.log(
            "memory_update_completed",
            operation="clear",
            duration_ms=helpers.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_CLEARED_MESSAGE)
