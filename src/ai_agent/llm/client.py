import dataclasses
import typing

import requests

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

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str: ...


@dataclasses.dataclass(kw_only=True, slots=True, frozen=True)
class LLMClient:
    config: config.LLMConfig

    @property
    def model_name(self) -> str:
        return self.config.model

    def complete(self, system_prompt: str, user_prompt: str, *, json_mode: bool) -> str:
        try:
            api_key: typing.Final = self.config.required_api_key
        except config.MissingAPIKeyError as error:
            raise LLMUnavailableError(str(error)) from error

        payload: typing.Final[dict[str, object]] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        try:
            response: typing.Final = requests.post(
                str(self.config.completions_url),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            response.raise_for_status()
        except requests.Timeout as error:
            raise LLMUnavailableError("Превышено время ожидания ответа LLM.") from error
        except requests.RequestException as error:
            raise LLMUnavailableError("LLM API недоступен.") from error

        try:
            response_data: typing.Final = response.json()
            content: typing.Final = response_data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError, ValueError) as error:
            raise InvalidLLMResponseError("LLM API вернул ответ неизвестного формата.") from error
        if not isinstance(content, str) or not content.strip():
            raise InvalidLLMResponseError("LLM API вернул пустой ответ.")
        return content.strip()
