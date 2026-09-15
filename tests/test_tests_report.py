import typing

from tests import reporting

REQUIRED_CASE_FIELDS: typing.Final = {
    "id",
    "type",
    "user_id",
    "session_id",
    "request",
    "expected",
    "actual",
    "tool_calls",
    "passed",
}
MINIMUM_REPORT_CASES: typing.Final = 20
MINIMUM_CASES_PER_DIALOGUE_GROUP: typing.Final = 2
DIALOGUE_TYPE_GROUPS: typing.Final = (
    {"main"},
    {"alternative"},
    {"error", "memory_error", "session_error"},
    {"memory", "memory_error"},
    {"session", "session_error"},
)


def test_report_contains_complete_results_for_all_dialogue_groups() -> None:
    report: typing.Final = reporting.load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])

    assert len(cases) >= MINIMUM_REPORT_CASES
    assert len({case["id"] for case in cases}) == len(cases)
    assert all(REQUIRED_CASE_FIELDS.issubset(case) for case in cases)
    assert all(case["user_id"] and case["session_id"] for case in cases)
    required_cases: typing.Final = [case for case in cases if case.get("required") is True]
    assert [case["id"] for case in required_cases] == report["required_test_ids"]
    assert [case["type"] for case in required_cases] == ["main", "error"]
    for dialogue_types in DIALOGUE_TYPE_GROUPS:
        assert sum(case["type"] in dialogue_types for case in cases) >= MINIMUM_CASES_PER_DIALOGUE_GROUP
