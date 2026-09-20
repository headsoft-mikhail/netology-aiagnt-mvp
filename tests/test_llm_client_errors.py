import types
import typing

import pydantic
import pydantic_ai
import pydantic_ai.exceptions
import pytest

from ai_agent.llm import client, config

TEST_BASE_URL: typing.Final = "https://api.groq.com"
TEST_MODEL: typing.Final = "test-model"
SYSTEM_PROMPT: typing.Final = "System prompt"
USER_PROMPT: typing.Final = "User prompt"
LLM_RESPONSE: typing.Final = "Structured response"


class StructuredOutput(pydantic.BaseModel):
    answer: str


def test_llm_config_reads_environment_variables(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", TEST_BASE_URL)
    monkeypatch.setenv("LLM_MODEL", TEST_MODEL)

    loaded_config: typing.Final = config.LLMConfig()

    assert loaded_config.required_api_key == "secret"
    assert str(loaded_config.base_url) == f"{TEST_BASE_URL}/"
    assert loaded_config.model == TEST_MODEL


def create_client(api_key: str | None = "secret") -> client.LLMClient:
    return client.LLMClient(
        config=config.LLMConfig(
            base_url=TEST_BASE_URL,
            model=TEST_MODEL,
            api_key=api_key,
            timeout_seconds=1,
        )
    )


def test_llm_client_reports_missing_api_key_without_sending_request() -> None:
    with pytest.raises(client.LLMUnavailableError, match="LLM_API_KEY"):
        create_client(api_key=None).complete(
            SYSTEM_PROMPT,
            USER_PROMPT,
            output_type=StructuredOutput,
        )


def test_llm_client_converts_provider_failure_to_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_provider_error(*args: object, **kwargs: object) -> typing.NoReturn:
        raise pydantic_ai.exceptions.ModelAPIError(TEST_MODEL, "provider unavailable")

    monkeypatch.setattr(pydantic_ai.Agent, "run_sync", raise_provider_error)

    with pytest.raises(client.LLMUnavailableError, match="недоступен"):
        create_client().complete(
            SYSTEM_PROMPT,
            USER_PROMPT,
            output_type=StructuredOutput,
        )


def test_llm_client_converts_invalid_structured_output_to_controlled_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_invalid_output(*args: object, **kwargs: object) -> typing.NoReturn:
        raise pydantic_ai.exceptions.UnexpectedModelBehavior("invalid output")

    monkeypatch.setattr(pydantic_ai.Agent, "run_sync", raise_invalid_output)

    with pytest.raises(client.InvalidLLMResponseError, match="неизвестного формата"):
        create_client().complete(
            SYSTEM_PROMPT,
            USER_PROMPT,
            output_type=StructuredOutput,
        )


def test_llm_client_returns_pydantic_ai_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_arguments: typing.Final[dict[str, object]] = {}
    expected_output: typing.Final = StructuredOutput(answer=LLM_RESPONSE)

    def return_structured_output(
        agent: pydantic_ai.Agent,
        user_prompt: str,
        **kwargs: object,
    ) -> types.SimpleNamespace:
        del agent, kwargs
        run_arguments["user_prompt"] = user_prompt
        return types.SimpleNamespace(output=expected_output)

    monkeypatch.setattr(pydantic_ai.Agent, "run_sync", return_structured_output)

    result: typing.Final = create_client().complete(
        SYSTEM_PROMPT,
        USER_PROMPT,
        output_type=StructuredOutput,
    )

    assert result == expected_output
    assert run_arguments == {"user_prompt": USER_PROMPT}
