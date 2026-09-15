import json
import pathlib
import typing

REPORT_PATH: typing.Final = pathlib.Path(__file__).with_name("tests_report.json")


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


def write_results(case_results: dict[str, bool]) -> None:
    report: typing.Final = load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])
    for case in cases:
        case_id = typing.cast(str, case["id"])
        case["passed"] = case_results.get(case_id, False)
    REPORT_PATH.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
