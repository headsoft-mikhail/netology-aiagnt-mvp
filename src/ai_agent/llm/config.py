import typing

import pydantic
import pydantic_settings


class MissingAPIKeyError(RuntimeError):
    pass


class LLMConfig(pydantic_settings.BaseSettings):
    model_config = pydantic_settings.SettingsConfigDict(
        env_prefix="LLM_",
    )

    base_url: pydantic.AnyHttpUrl = pydantic.AnyHttpUrl("https://api.groq.com/openai/v1")
    model: str = pydantic.Field(default="openai/gpt-oss-20b", min_length=1)
    api_key: pydantic.SecretStr | None = None
    timeout_seconds: float = pydantic.Field(default=60, gt=0)

    @property
    def required_api_key(self) -> str:
        if self.api_key is None or not self.api_key.get_secret_value().strip():
            raise MissingAPIKeyError("Переменная окружения LLM_API_KEY не задана.")
        return self.api_key.get_secret_value()


llm_config: typing.Final = LLMConfig()
