import dataclasses
import typing

import pydantic
import pydantic_ai
import pydantic_ai.exceptions
import pydantic_ai.models
import pydantic_ai.models.openai
import pydantic_ai.providers.openai

from ai_agent.llm import config


class LLMError(RuntimeError):
    pass


class LLMUnavailableError(LLMError):
    pass


class InvalidLLMResponseError(LLMError):
    pass


class ChatClientProtocol(typing.Protocol):
    @property
    def model_name(self) -> str: ...

    def complete[ResponseModel: pydantic.BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_type: type[ResponseModel],
    ) -> ResponseModel: ...


@dataclasses.dataclass(kw_only=True, slots=True)
class LLMClient:
    config: config.LLMConfig
    _model: pydantic_ai.models.Model | None = dataclasses.field(
        default=None,
        init=False,
        repr=False,
    )

    @property
    def model_name(self) -> str:
        return self.config.model

    def complete[ResponseModel: pydantic.BaseModel](
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        output_type: type[ResponseModel],
    ) -> ResponseModel:
        try:
            agent: typing.Final = pydantic_ai.Agent(
                self._get_model(),
                output_type=pydantic_ai.PromptedOutput(output_type),
                system_prompt=system_prompt,
                model_settings={
                    "temperature": 0,
                    "timeout": self.config.timeout_seconds,
                },
            )
            result: typing.Final = agent.run_sync(user_prompt)
        except pydantic_ai.exceptions.ModelAPIError as error:
            raise LLMUnavailableError("LLM API недоступен.") from error
        except pydantic_ai.exceptions.UnexpectedModelBehavior as error:
            raise InvalidLLMResponseError("LLM API вернул ответ неизвестного формата.") from error
        return result.output

    def _get_model(self) -> pydantic_ai.models.Model:
        if self._model is None:
            try:
                api_key: typing.Final = self.config.required_api_key
                provider: typing.Final = pydantic_ai.providers.openai.OpenAIProvider(
                    api_key=api_key,
                    base_url=str(self.config.base_url),
                )
            except (config.MissingAPIKeyError, pydantic_ai.exceptions.UserError) as error:
                raise LLMUnavailableError(str(error)) from error
            self._model = pydantic_ai.models.openai.OpenAIChatModel(
                self.config.model,
                provider=provider,
            )
        return self._model
