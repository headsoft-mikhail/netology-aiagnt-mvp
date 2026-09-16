import dataclasses
import json
import time
import typing

from ai_agent import contracts
from ai_agent.catalog import models as catalog_models
from ai_agent.handlers import helpers, runtime
from ai_agent.llm import client as llm_client
from ai_agent.llm.service import LLMService

type SearchResult = contracts.KnowledgeBaseSearchResult | catalog_models.ProductSearchResult
type ResultField = typing.Literal["knowledge_result", "catalog_result"]


@dataclasses.dataclass(frozen=True, kw_only=True, slots=True)
class ActionResultPolicy:
    result_field: ResultField
    optional_when_actions_present: frozenset[contracts.AgentAction] = frozenset()


ACTION_RESULT_POLICIES: typing.Final = {
    contracts.AgentAction.SEARCH_KNOWLEDGE_BASE: ActionResultPolicy(
        result_field="knowledge_result",
        optional_when_actions_present=frozenset({contracts.AgentAction.SEARCH_PRODUCTS}),
    ),
    contracts.AgentAction.SEARCH_PRODUCTS: ActionResultPolicy(
        result_field="catalog_result",
    ),
}


@dataclasses.dataclass(kw_only=True, slots=True)
class FinalAnswerHandler:
    llm: LLMService

    def handle(self, context: runtime.RunContext) -> contracts.AgentResponse:
        context.agent_state.assembled_context = json.dumps(
            {
                "memory": [fact.model_dump(mode="json") for fact in context.agent_state.relevant_memory],
                "knowledge": (
                    context.agent_state.knowledge_result.model_dump(mode="json")
                    if context.agent_state.knowledge_result
                    else None
                ),
                "catalog": (
                    context.agent_state.catalog_result.model_dump(mode="json")
                    if context.agent_state.catalog_result
                    else None
                ),
            },
            ensure_ascii=False,
        )
        context.log(
            "context_assembled",
            memory_records=len(context.agent_state.relevant_memory),
            knowledge_fragments=(
                len(context.agent_state.knowledge_result.fragments) if context.agent_state.knowledge_result else 0
            ),
            products=context.agent_state.catalog_result.total if context.agent_state.catalog_result else 0,
            context_chars=len(context.agent_state.assembled_context),
        )

        started_at: typing.Final = time.perf_counter()
        context.log("llm_request_started", phase="final_answer", model=self.llm.model_name)
        try:
            draft: typing.Final = self.llm.create_final_answer(
                context.agent_state.request.query,
                context.agent_state.relevant_memory,
                context.agent_state.knowledge_result,
                context.agent_state.catalog_result,
            )
            self._validate_product_codes(draft.product_codes, context)
        except llm_client.LLMError as error:
            return context.llm_error_response(error)
        context.log(
            "llm_request_completed",
            phase="final_answer",
            model=self.llm.model_name,
            duration_ms=helpers.duration_ms(started_at),
        )

        status: typing.Final = self._resolve_status(context)
        return context.response(status, draft.answer, product_codes=draft.product_codes)

    @staticmethod
    def _validate_product_codes(
        product_codes: list[str],
        context: runtime.RunContext,
    ) -> None:
        catalog_result: typing.Final = context.agent_state.catalog_result
        available_product_codes: typing.Final = (
            {product.product_code for product in catalog_result.products} if catalog_result else set()
        )
        if not set(product_codes).issubset(available_product_codes):
            raise llm_client.InvalidLLMResponseError("LLM mentioned a product outside the catalog result.")

    @staticmethod
    def _resolve_status(context: runtime.RunContext) -> contracts.AgentRunStatus:
        selected_actions: typing.Final = set(context.agent_state.selected_actions)
        for action in selected_actions:
            policy = ACTION_RESULT_POLICIES.get(action)
            if policy is None or policy.optional_when_actions_present & selected_actions:
                continue
            result = typing.cast(
                SearchResult | None,
                getattr(context.agent_state, policy.result_field),
            )
            if result is not None and result.status is contracts.ToolStatus.NO_RESULTS:
                return contracts.AgentRunStatus.NOT_FOUND

        return contracts.AgentRunStatus.COMPLETED
