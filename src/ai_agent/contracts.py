import datetime
import enum

import pydantic


class AgentAction(enum.StrEnum):
    ANSWER = "answer"
    CLARIFY = "clarify"
    SEARCH_KNOWLEDGE_BASE = "search_knowledge_base"
    SEARCH_PRODUCTS = "search_products"
    UPDATE_MEMORY = "update_memory"
    DELETE_MEMORY = "delete_memory"
    CLEAR_MEMORY = "clear_memory"
    UNSUPPORTED = "unsupported"


class AgentRunStatus(enum.StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    NEEDS_INPUT = "needs_input"
    NOT_FOUND = "not_found"
    FAILED = "failed"


class ToolStatus(enum.StrEnum):
    OK = "ok"
    NO_RESULTS = "no_results"
    ERROR = "error"


class ErrorCode(enum.StrEnum):
    INVALID_INPUT = "invalid_input"
    NO_RESULTS = "no_results"
    KNOWLEDGE_BASE_UNAVAILABLE = "knowledge_base_unavailable"
    CATALOG_UNAVAILABLE = "catalog_unavailable"
    MEMORY_UNAVAILABLE = "memory_unavailable"
    LLM_UNAVAILABLE = "llm_unavailable"
    INVALID_LLM_RESPONSE = "invalid_llm_response"
    INTERNAL_ERROR = "internal_error"


class MemoryType(enum.StrEnum):
    LONG_TERM = "long_term"


class MemoryKey(enum.StrEnum):
    BUDGET_RUB = "budget_rub"
    PREFERRED_BRANDS = "preferred_brands"
    EXCLUDED_BRANDS = "excluded_brands"
    AREA_SQM = "area_sqm"
    DEVICE_COUNT = "device_count"
    TARIFF_SPEED_MBPS = "tariff_speed_mbps"
    CURRENT_EQUIPMENT = "current_equipment"


class AgentRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    user_id: str = pydantic.Field(min_length=1)
    query: str = pydantic.Field(min_length=1)
    session_id: str | None = pydantic.Field(default=None, min_length=1)


class ToolError(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    code: ErrorCode
    message: str = pydantic.Field(min_length=1)


class MemoryFact(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    id: int = pydantic.Field(gt=0)
    user_id: str = pydantic.Field(min_length=1)
    session_id: str = pydantic.Field(min_length=1)
    memory_type: MemoryType
    key: MemoryKey
    value: str = pydantic.Field(min_length=1)
    source: str = pydantic.Field(min_length=1)
    created_at: datetime.datetime
    updated_at: datetime.datetime
    expires_at: datetime.datetime | None = None


class MemoryUpdate(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    key: MemoryKey
    value: str = pydantic.Field(min_length=1)


class KnowledgeFragment(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    chunk_id: str = pydantic.Field(min_length=1)
    text: str = pydantic.Field(min_length=1)
    source: str = pydantic.Field(min_length=1)
    score: float = pydantic.Field(ge=0, le=1)


class KnowledgeSearchResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    status: ToolStatus
    fragments: list[KnowledgeFragment] = pydantic.Field(default_factory=list)
    context: str = ""
    sources: list[str] = pydantic.Field(default_factory=list)
    error: ToolError | None = None


class ToolCallTrace(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    tool_name: str = pydantic.Field(min_length=1)
    status: ToolStatus
    input_json: str = pydantic.Field(min_length=1)
    output_json: str = pydantic.Field(min_length=1)
    error: ToolError | None = None


class AgentResponse(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    status: AgentRunStatus
    answer: str = pydantic.Field(min_length=1)
    sources: list[str] = pydantic.Field(default_factory=list)
    product_codes: list[str] = pydantic.Field(default_factory=list)
    tool_calls: list[ToolCallTrace] = pydantic.Field(default_factory=list)
    memory_used: list[int] = pydantic.Field(default_factory=list)
    request_id: str = pydantic.Field(min_length=1)
    session_id: str = pydantic.Field(min_length=1)
    errors: list[ToolError] = pydantic.Field(default_factory=list)
