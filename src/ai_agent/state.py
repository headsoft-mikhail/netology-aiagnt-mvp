import pydantic

from ai_agent import contracts
from ai_agent.catalog import models as catalog_models


class AgentState(pydantic.BaseModel):
    """Полное наблюдаемое состояние одного запуска агента."""

    model_config = pydantic.ConfigDict(extra="forbid")

    request: contracts.AgentRequest
    status: contracts.AgentRunStatus = contracts.AgentRunStatus.PENDING
    selected_actions: list[contracts.AgentAction] = pydantic.Field(default_factory=list)
    relevant_memory: list[contracts.MemoryFact] = pydantic.Field(default_factory=list)
    knowledge_result: contracts.KnowledgeSearchResult | None = None
    catalog_filters: catalog_models.ProductSearchInput | None = None
    catalog_result: catalog_models.ProductSearchResult | None = None
    assembled_context: str = ""
    tool_calls: list[contracts.ToolCallTrace] = pydantic.Field(default_factory=list)
    errors: list[contracts.ToolError] = pydantic.Field(default_factory=list)
    final_response: contracts.AgentResponse | None = None
