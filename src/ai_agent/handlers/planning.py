import dataclasses
import time
import typing

from ai_agent import contracts
from ai_agent.handlers import helpers, runtime
from ai_agent.llm import client as llm_client
from ai_agent.llm import models as llm_models
from ai_agent.llm.service import LLMService

MEMORY_ACTIONS: typing.Final = {
    contracts.AgentAction.UPDATE_MEMORY,
    contracts.AgentAction.DELETE_MEMORY,
    contracts.AgentAction.CLEAR_MEMORY,
}


@dataclasses.dataclass(kw_only=True, slots=True)
class PlanningHandler:
    llm: LLMService

    def handle(
        self,
        context: runtime.RunContext,
        conversation: list[dict[str, str]],
    ) -> llm_models.AgentPlan | contracts.AgentResponse:
        started_at: typing.Final = time.perf_counter()
        context.log("llm_request_started", phase="planning", model=self.llm.model_name)
        try:
            plan: typing.Final = self.llm.plan(
                context.agent_state.request.query,
                context.agent_state.relevant_memory,
                conversation,
            )
        except llm_client.LLMError as error:
            return context.llm_error_response(error)
        context.log(
            "llm_request_completed",
            phase="planning",
            model=self.llm.model_name,
            duration_ms=helpers.duration_ms(started_at),
        )
        context.agent_state.selected_actions = plan.actions
        context.log("actions_selected", actions=[action.value for action in plan.actions])
        if not any(action in MEMORY_ACTIONS for action in plan.actions):
            context.log("memory_update_skipped", reason="no_explicit_memory_action")
        return plan
