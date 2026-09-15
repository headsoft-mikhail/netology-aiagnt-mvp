import typing

import pydantic

from ai_agent import contracts
from ai_agent.catalog import models as catalog_models

type ToolResult = contracts.KnowledgeBaseSearchResult | catalog_models.ProductSearchResult


class Tool(typing.Protocol):
    """Uniform runtime contract for every agent Tool."""

    @property
    def name(self) -> str: ...

    @property
    def input_model(self) -> type[pydantic.BaseModel]: ...

    def invoke(self, tool_input: pydantic.BaseModel) -> ToolResult: ...
