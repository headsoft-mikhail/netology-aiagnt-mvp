import typing

import pytest
import requests

from ai_agent.llm import client, config

TEST_COMPLETIONS_URL: typing.Final = "https://example.com/chat/completions"
TEST_MODEL: typing.Final = "test-model"


def test_llm_config_reads_variables_supplied_by_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_COMPLETIONS_URL", TEST_COMPLETIONS_URL)
    monkeypatch.setenv("LLM_MODEL", TEST_MODEL)

    loaded_config: typing.Final = config.LLMConfig()

    assert loaded_config.required_api_key == "secret"
    assert str(loaded_config.completions_url) == TEST_COMPLETIONS_URL
    assert loaded_config.model == TEST_MODEL


class InvalidResponse:
    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, object]:
        return {"choices": []}


class HTTPErrorResponse:
    def raise_for_status(self) -> typing.NoReturn:
        raise requests.HTTPError


def create_client(api_key: str | None = "secret") -> client.LLMClient:
    return client.LLMClient(
        config=config.LLMConfig(
            completions_url="https://api.groq.com/openai/v1/chat/completions",
            model="test-model",
            api_key=api_key,
            timeout_seconds=1,
        )
    )


def test_llm_client_reports_missing_api_key_without_sending_request() -> None:
    with pytest.raises(client.LLMUnavailableError, match="LLM_API_KEY"):
        create_client(api_key=None).complete("system", "user", json_mode=True)


def test_llm_client_converts_timeout_to_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*args: object, **kwargs: object) -> typing.NoReturn:
        raise requests.Timeout

    monkeypatch.setattr(requests, "post", raise_timeout)

    with pytest.raises(client.LLMUnavailableError, match="время ожидания"):
        create_client().complete("system", "user", json_mode=True)


def test_llm_client_converts_http_failure_to_controlled_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: HTTPErrorResponse())

    with pytest.raises(client.LLMUnavailableError, match="недоступен"):
        create_client().complete("system", "user", json_mode=True)


def test_llm_client_rejects_unknown_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(requests, "post", lambda *args, **kwargs: InvalidResponse())

    with pytest.raises(client.InvalidLLMResponseError, match="неизвестного формата"):
        create_client().complete("system", "user", json_mode=True)
