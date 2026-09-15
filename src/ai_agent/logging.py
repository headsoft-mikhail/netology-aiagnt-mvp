import json
import logging
import sys
import typing

LOG_FORMAT: typing.Final = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure(*, verbose: bool = False, debug: bool = False) -> None:
    level: typing.Final = logging.DEBUG if debug else logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        stream=sys.stderr,
        force=True,
    )


def event(
    logger: logging.Logger,
    level: int,
    name: str,
    *,
    request_id: str | None = None,
    session_id: str | None = None,
    **fields: object,
) -> None:
    parts: typing.Final = [f"event={name}"]
    if request_id is not None:
        parts.append(f"request_id={request_id}")
    if session_id is not None:
        parts.append(f"session_id={session_id}")
    parts.extend(f"{key}={_serialize(value)}" for key, value in fields.items())
    logger.log(level, " ".join(parts))


def _serialize(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str)
