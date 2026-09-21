import typing

import pydantic


class ChatRequest(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(str_strip_whitespace=True)

    user_id: str = pydantic.Field(min_length=1)
    session_id: str | None = pydantic.Field(default=None, min_length=1)
    message: str = pydantic.Field(min_length=1)


class HealthResponse(pydantic.BaseModel):
    status: typing.Literal["ok"] = "ok"


class ReadinessResponse(pydantic.BaseModel):
    status: typing.Literal["ready", "not_ready"]
    agent_ready: bool
    catalog_ready: bool
    knowledge_base_ready: bool
