import dataclasses
import logging
import time
import typing

from ai_agent import contracts
from ai_agent.handlers import runtime
from ai_agent.llm import models as llm_models
from ai_agent.tools import registry, search_knowledge_base


@dataclasses.dataclass(kw_only=True, slots=True)
class KnowledgeSearchHandler:
    tool_registry: registry.ToolRegistry

    def handle(
        self,
        plan: llm_models.AgentPlan,
        context: runtime.RunContext,
    ) -> contracts.AgentResponse | None:
        query: typing.Final = typing.cast(str, plan.knowledge_query)
        arguments: typing.Final[dict[str, object]] = {"query": query}
        started_at: typing.Final = time.perf_counter()
        context.log(
            "tool_call_started",
            tool=search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
            arguments={"query_length": len(query)},
        )
        execution: typing.Final = self.tool_registry.execute(
            search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
            arguments,
        )
        result: typing.Final = typing.cast(
            contracts.KnowledgeBaseSearchResult,
            execution.result,
        )
        context.agent_state.knowledge_result = result
        context.agent_state.tool_calls.append(execution.trace)
        context.log(
            "tool_call_completed",
            level=logging.WARNING if result.status is contracts.ToolStatus.NO_RESULTS else logging.INFO,
            tool=search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
            status=result.status.value,
            duration_ms=runtime.duration_ms(started_at),
            results_count=len(result.fragments),
        )
        context.log(
            "rag_retrieval_completed",
            fragments_count=len(result.fragments),
            sources=result.sources,
            scores=[round(fragment.score, 3) for fragment in result.fragments],
        )
        if result.status is contracts.ToolStatus.ERROR:
            return context.tool_error_response(
                result.error,
                component=search_knowledge_base.SEARCH_KNOWLEDGE_BASE_TOOL_NAME,
            )
        return None
