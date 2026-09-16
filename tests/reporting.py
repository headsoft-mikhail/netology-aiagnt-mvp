import json
import pathlib
import typing

from ai_agent import contracts

REPORT_PATH: typing.Final = pathlib.Path(__file__).with_name("tests_report.json")
PENDING_ACTUAL: typing.Final = {
    "status": "pending",
    "details": "Сценарий ещё не выполнен в текущем запуске pytest.",
}
CASE_ACTUALS: typing.Final[dict[str, dict[str, object]]] = {}
CASE_TOOL_CALLS: typing.Final[dict[str, list[dict[str, object]]]] = {}


def load_report() -> dict[str, object]:
    return typing.cast(
        dict[str, object],
        json.loads(REPORT_PATH.read_text(encoding="utf-8")),
    )


def get_case(case_id: str) -> dict[str, object]:
    report: typing.Final = load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])
    try:
        return next(case for case in cases if case["id"] == case_id)
    except StopIteration as error:
        raise KeyError(f"Unknown report case: {case_id}") from error


def request(case_id: str) -> str:
    return typing.cast(str, get_case(case_id)["request"])


def initialize_report() -> None:
    CASE_ACTUALS.clear()
    CASE_TOOL_CALLS.clear()
    report: typing.Final = load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])
    for case in cases:
        case["actual"] = PENDING_ACTUAL
        case["tool_calls"] = []
        case["passed"] = False
    _write_report(report)


def record_agent_response(
    case_id: str,
    response: contracts.AgentResponse,
    *,
    memory_operation: str | None = None,
    memory_keys: list[contracts.MemoryKey] | None = None,
) -> None:
    CASE_ACTUALS[case_id] = {
        "status": response.status.value,
        "answer": response.answer,
        "sources": response.sources,
        "product_codes": response.product_codes,
        "memory": {
            "used_fact_ids": response.memory_used,
            "operation": memory_operation,
            "keys": [key.value for key in memory_keys or []],
        },
        "errors": [error.model_dump(mode="json") for error in response.errors],
    }
    CASE_TOOL_CALLS[case_id] = [
        {
            "name": call.tool_name,
            "status": call.status.value,
            "input": json.loads(call.input_json),
            "output": json.loads(call.output_json),
            "error": call.error.model_dump(mode="json") if call.error else None,
        }
        for call in response.tool_calls
    ]


def record_custom_result(
    case_id: str,
    actual: dict[str, object],
    *,
    tool_calls: list[dict[str, object]] | None = None,
) -> None:
    CASE_ACTUALS[case_id] = actual
    CASE_TOOL_CALLS[case_id] = tool_calls or []


def write_results(case_results: dict[str, bool]) -> set[str]:
    report: typing.Final = load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])
    missing_actuals: typing.Final[set[str]] = set()
    for case in cases:
        case_id = typing.cast(str, case["id"])
        if case_id not in CASE_ACTUALS:
            missing_actuals.add(case_id)
            case["actual"] = {
                "status": "not_recorded",
                "details": "Тест не зарегистрировал фактический результат текущего запуска.",
            }
            case["tool_calls"] = []
        else:
            case["actual"] = CASE_ACTUALS[case_id]
            case["tool_calls"] = CASE_TOOL_CALLS[case_id]
        case["passed"] = case_results.get(case_id, False) and case_id not in missing_actuals
    _write_report(report)
    return missing_actuals


def _write_report(report: dict[str, object]) -> None:
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
