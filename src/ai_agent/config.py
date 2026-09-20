import typing

import pydantic
import pydantic_settings


class AgentConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="AGENT_",
    )

    max_actions_per_request: int = pydantic.Field(default=2, ge=1, le=2)


agent_config: typing.Final = AgentConfig()
