import dataclasses
import typing

from ai_agent import contracts
from ai_agent.handlers import memory, runtime
from ai_agent.llm import models as llm_models


@dataclasses.dataclass(kw_only=True, slots=True)
class DirectActionHandler:
    memory_handler: memory.MemoryHandler

    def handle(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse | None:
        memory_response: typing.Final = self.memory_handler.handle_action(plan, context)
        if memory_response is not None:
            return memory_response

        action: typing.Final = plan.actions[0]
        if action is contracts.AgentAction.CLARIFY:
            return context.response(
                contracts.AgentRunStatus.NEEDS_INPUT,
                typing.cast(str, plan.response_message),
            )
        if action in (contracts.AgentAction.ANSWER, contracts.AgentAction.UNSUPPORTED):
            return context.response(
                contracts.AgentRunStatus.COMPLETED,
                typing.cast(str, plan.response_message),
            )
        return None
