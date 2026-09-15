import dataclasses
import logging
import time
import typing

from ai_agent import contracts
from ai_agent.catalog import models as catalog_models
from ai_agent.handlers import runtime
from ai_agent.llm import client as llm_client
from ai_agent.llm.service import LLMService
from ai_agent.tools import registry, search_products


@dataclasses.dataclass(kw_only=True, slots=True)
class CatalogSearchHandler:
    llm: LLMService
    tool_registry: registry.ToolRegistry

    def handle(self, context: runtime.RunContext) -> contracts.AgentResponse | None:
        filters_started_at: typing.Final = time.perf_counter()
        context.log("llm_request_started", phase="catalog_filters", model=self.llm.model_name)
        try:
            filters: typing.Final = self.llm.create_product_search(
                context.agent_state.request.query,
                context.agent_state.relevant_memory,
                context.agent_state.knowledge_result.context if context.agent_state.knowledge_result else "",
            )
        except llm_client.LLMError as error:
            return context.llm_error_response(error)
        context.agent_state.catalog_filters = filters
        context.log(
            "llm_request_completed",
            phase="catalog_filters",
            model=self.llm.model_name,
            duration_ms=runtime.duration_ms(filters_started_at),
        )
        arguments: typing.Final = filters.model_dump(mode="json")
        context.log("catalog_filters_validated", filters=arguments)

        tool_started_at: typing.Final = time.perf_counter()
        context.log(
            "tool_call_started",
            tool=search_products.SEARCH_PRODUCTS_TOOL_NAME,
            arguments=arguments,
        )
        execution: typing.Final = self.tool_registry.execute(
            search_products.SEARCH_PRODUCTS_TOOL_NAME,
            arguments,
        )
        result: typing.Final = typing.cast(catalog_models.ProductSearchResult, execution.result)
        context.agent_state.catalog_result = result
        context.agent_state.tool_calls.append(execution.trace)
        context.log(
            "tool_call_completed",
            level=logging.WARNING if result.status is contracts.ToolStatus.NO_RESULTS else logging.INFO,
            tool=search_products.SEARCH_PRODUCTS_TOOL_NAME,
            status=result.status.value,
            duration_ms=runtime.duration_ms(tool_started_at),
            results_count=result.total,
        )
        if result.status is contracts.ToolStatus.ERROR:
            return context.tool_error_response(
                result.error,
                component=search_products.SEARCH_PRODUCTS_TOOL_NAME,
            )
        return None
