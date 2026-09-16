import functools
import logging
import time
import typing

from ai_agent import contracts

HandlerOperation = typing.TypeVar(
    "HandlerOperation",
    bound=typing.Callable[..., contracts.AgentResponse | None],
)


class ExceptionHandlingContext(typing.Protocol):
    logger: logging.Logger

    def log(self, event_name: str, *, level: int = logging.INFO, **fields: object) -> None: ...

    def error_response(
        self,
        code: contracts.ErrorCode,
        message: str,
        *,
        component: str,
        error_type: str | None = None,
        reason: str | None = None,
    ) -> contracts.AgentResponse: ...

    def response(
        self,
        status: contracts.AgentRunStatus,
        answer: str,
        *,
        product_codes: list[str] | None = None,
    ) -> contracts.AgentResponse: ...


def catch_exception_and_log(
    *,
    exceptions: tuple[type[Exception], ...],
    component: str,
    error_code: contracts.ErrorCode,
    user_message: str,
    response_status: contracts.AgentRunStatus = contracts.AgentRunStatus.FAILED,
) -> typing.Callable[[HandlerOperation], HandlerOperation]:
    """Convert expected handler failures into logged, controlled responses.

    Decorated methods must receive an ``ExceptionHandlingContext`` as their
    second positional argument, immediately after ``self``.
    """

    def decorator(operation: HandlerOperation) -> HandlerOperation:
        @functools.wraps(operation)
        def wrapped(
            handler: object,
            context: ExceptionHandlingContext,
            *args: object,
            **kwargs: object,
        ):
            try:
                return operation(handler, context, *args, **kwargs)
            except exceptions as error:
                context.logger.log(
                    logging.DEBUG,
                    f"{component} operation failed",
                    exc_info=(type(error), error, error.__traceback__),
                )
                if response_status is contracts.AgentRunStatus.FAILED:
                    return context.error_response(
                        error_code,
                        user_message,
                        component=component,
                        error_type=type(error).__name__,
                        reason=str(error),
                    )
                context.log(
                    "input_rejected",
                    level=logging.WARNING,
                    component=component,
                    error_code=error_code.value,
                    error_type=type(error).__name__,
                    reason=str(error),
                )
                return context.response(response_status, user_message)

        return typing.cast(HandlerOperation, wrapped)

    return decorator


def duration_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 2)
