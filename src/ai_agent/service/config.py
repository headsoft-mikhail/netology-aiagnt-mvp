import typing

import pydantic
import pydantic_settings


class APIConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="API_",
    )

    host: str = pydantic.Field(default="0.0.0.0", min_length=1)
    port: int = pydantic.Field(default=8000, ge=1, le=65535)
    log_level: typing.Literal["critical", "error", "warning", "info", "debug", "trace"] = "info"


api_config: typing.Final = APIConfig()
