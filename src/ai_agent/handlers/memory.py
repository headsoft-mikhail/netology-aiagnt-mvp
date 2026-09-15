import dataclasses
import time
import typing

from ai_agent import contracts
from ai_agent.handlers import runtime
from ai_agent.llm import models as llm_models
from ai_agent.memory.repository import MemoryRepository

MEMORY_SOURCE: typing.Final = "explicit_user_request"
MEMORY_SAVED_MESSAGE: typing.Final = "Предпочтение сохранено."
MEMORY_DELETED_MESSAGE: typing.Final = "Предпочтение удалено."
MEMORY_CLEARED_MESSAGE: typing.Final = "Все предпочтения пользователя удалены."
MEMORY_ERROR_MESSAGE: typing.Final = "Не удалось обратиться к памяти пользователя."
INVALID_MEMORY_MESSAGE: typing.Final = "Эти данные нельзя сохранить в памяти."


@dataclasses.dataclass(kw_only=True, slots=True)
class MemoryHandler:
    repository: MemoryRepository
    limit: int

    def load(self, context: runtime.RunContext) -> contracts.AgentResponse | None:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_load_started", user_id=context.agent_state.request.user_id)
        try:
            context.agent_state.relevant_memory = self.repository.get_relevant(
                context.agent_state.request.user_id,
                limit=self.limit,
            )
        except Exception:  # noqa: BLE001
            return context.error_response(
                contracts.ErrorCode.MEMORY_UNAVAILABLE,
                MEMORY_ERROR_MESSAGE,
                component="memory",
            )
        context.log(
            "memory_load_completed",
            records_count=len(context.agent_state.relevant_memory),
            duration_ms=runtime.duration_ms(started_at),
        )
        return None

    def handle_action(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse | None:
        action: typing.Final = plan.actions[0]
        if action is contracts.AgentAction.UPDATE_MEMORY:
            return self._save(plan, context)
        if action is contracts.AgentAction.DELETE_MEMORY:
            return self._delete(plan, context)
        if action is contracts.AgentAction.CLEAR_MEMORY:
            return self._clear(context)
        return None

    def _save(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="save")
        try:
            update: typing.Final = typing.cast(contracts.MemoryUpdate, plan.memory_update)
            saved_fact: typing.Final = self.repository.save_fact(
                context.agent_state.request.user_id,
                update.key,
                update.value,
                source=MEMORY_SOURCE,
                session_id=typing.cast(str, context.agent_state.request.session_id),
            )
            context.agent_state.relevant_memory = [saved_fact]
        except ValueError, TypeError:
            return context.error_response(
                contracts.ErrorCode.INVALID_INPUT,
                INVALID_MEMORY_MESSAGE,
                component="memory",
            )
        except Exception:  # noqa: BLE001
            return context.error_response(
                contracts.ErrorCode.MEMORY_UNAVAILABLE,
                MEMORY_ERROR_MESSAGE,
                component="memory",
            )
        context.log(
            "memory_update_completed",
            operation="save",
            key=saved_fact.key.value,
            duration_ms=runtime.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_SAVED_MESSAGE)

    def _delete(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="delete")
        memory_key: typing.Final = typing.cast(contracts.MemoryKey, plan.memory_key)
        try:
            self.repository.delete_fact(context.agent_state.request.user_id, memory_key)
        except Exception:  # noqa: BLE001
            return context.error_response(
                contracts.ErrorCode.MEMORY_UNAVAILABLE,
                MEMORY_ERROR_MESSAGE,
                component="memory",
            )
        context.log(
            "memory_update_completed",
            operation="delete",
            key=memory_key.value,
            duration_ms=runtime.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_DELETED_MESSAGE)

    def _clear(self, context: runtime.RunContext) -> contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("memory_update_started", operation="clear")
        try:
            self.repository.clear_user(context.agent_state.request.user_id)
        except Exception:  # noqa: BLE001
            return context.error_response(
                contracts.ErrorCode.MEMORY_UNAVAILABLE,
                MEMORY_ERROR_MESSAGE,
                component="memory",
            )
        context.log(
            "memory_update_completed",
            operation="clear",
            duration_ms=runtime.duration_ms(started_at),
        )
        return context.response(contracts.AgentRunStatus.COMPLETED, MEMORY_CLEARED_MESSAGE)
