import dataclasses
import logging
import time
import typing

from ai_agent import contracts, state
from ai_agent import logging as agent_logging
from ai_agent.llm import client as llm_client

LLM_ERROR_MESSAGE: typing.Final = "Не удалось получить корректный ответ языковой модели."
UNKNOWN_COMPONENT_ERROR_MESSAGE: typing.Final = "Компонент завершился с неизвестной ошибкой."


@dataclasses.dataclass(kw_only=True, slots=True)
class RunContext:
    agent_state: state.AgentState
    request_id: str
    logger: logging.Logger
    llm_model_name: str

    def log(self, event_name: str, *, level: int = logging.INFO, **fields: object) -> None:
        agent_logging.event(
            self.logger,
            level,
            event_name,
            request_id=self.request_id,
            session_id=self.agent_state.request.session_id,
            **fields,
        )

    def tool_error_response(
        self,
        error: contracts.ToolError | None,
        *,
        component: str,
    ) -> contracts.AgentResponse:
        actual_error: typing.Final = error or contracts.ToolError(
            code=contracts.ErrorCode.INTERNAL_ERROR,
            message=UNKNOWN_COMPONENT_ERROR_MESSAGE,
        )
        return self.error_response(
            actual_error.code,
            actual_error.message,
            component=component,
        )

    def llm_error_response(self, error: llm_client.LLMError) -> contracts.AgentResponse:
        self.logger.log(
            logging.DEBUG,
            "LLM operation failed",
            exc_info=(type(error), error, error.__traceback__),
        )
        self.log(
            "llm_request_failed",
            model=self.llm_model_name,
            error_type=type(error).__name__,
            reason=str(error),
        )
        code: typing.Final = (
            contracts.ErrorCode.LLM_UNAVAILABLE
            if isinstance(error, llm_client.LLMUnavailableError)
            else contracts.ErrorCode.INVALID_LLM_RESPONSE
        )
        return self.error_response(
            code,
            LLM_ERROR_MESSAGE,
            component="llm",
            error_type=type(error).__name__,
            reason=str(error),
        )

    def error_response(
        self,
        code: contracts.ErrorCode,
        message: str,
        *,
        component: str,
        error_type: str | None = None,
        reason: str | None = None,
    ) -> contracts.AgentResponse:
        self.agent_state.errors.append(contracts.ToolError(code=code, message=message))
        fields: typing.Final[dict[str, object]] = {
            "component": component,
            "error_code": code.value,
            "message": message,
        }
        if error_type is not None:
            fields["error_type"] = error_type
        if reason is not None:
            fields["reason"] = reason
        self.log(
            "request_failed",
            level=logging.ERROR,
            **fields,
        )
        return self.response(contracts.AgentRunStatus.FAILED, message)

    def response(
        self,
        status: contracts.AgentRunStatus,
        answer: str,
        *,
        product_codes: list[str] | None = None,
    ) -> contracts.AgentResponse:
        sources: typing.Final = self.agent_state.knowledge_result.sources if self.agent_state.knowledge_result else []
        response: typing.Final = contracts.AgentResponse(
            status=status,
            answer=answer,
            sources=sources,
            product_codes=product_codes or [],
            tool_calls=self.agent_state.tool_calls,
            memory_used=[fact.id for fact in self.agent_state.relevant_memory],
            request_id=self.request_id,
            session_id=typing.cast(str, self.agent_state.request.session_id),
            errors=self.agent_state.errors,
        )
        self.agent_state.status = status
        self.agent_state.final_response = response
        self.log(
            "request_completed",
            level=(
                logging.WARNING
                if status in {contracts.AgentRunStatus.NEEDS_INPUT, contracts.AgentRunStatus.NOT_FOUND}
                else logging.INFO
            ),
            status=status.value,
            tool_calls=len(response.tool_calls),
            products=len(response.product_codes),
            errors=len(response.errors),
        )
        return response


def duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
