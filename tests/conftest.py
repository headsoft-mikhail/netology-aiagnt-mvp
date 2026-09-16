import typing

import pytest

from tests import reporting

REPORT_CASE_MARKER: typing.Final = "report_case"
NODE_CASES: typing.Final[dict[str, tuple[str, ...]]] = {}
CASE_RESULTS: typing.Final[dict[str, bool]] = {}


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{REPORT_CASE_MARKER}(*case_ids): bind a test result to report cases",
    )


def pytest_sessionstart(session: pytest.Session) -> None:
    del session
    NODE_CASES.clear()
    CASE_RESULTS.clear()
    reporting.initialize_report()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    for item in items:
        case_ids = tuple(
            typing.cast(str, case_id) for marker in item.iter_markers(REPORT_CASE_MARKER) for case_id in marker.args
        )
        if case_ids:
            NODE_CASES[item.nodeid] = case_ids

    report: typing.Final = reporting.load_report()
    cases: typing.Final = typing.cast(list[dict[str, object]], report["tests"])
    configured_case_ids: typing.Final = {typing.cast(str, case["id"]) for case in cases}
    collected_case_ids: typing.Final = {case_id for case_ids in NODE_CASES.values() for case_id in case_ids}
    if configured_case_ids != collected_case_ids:
        missing: typing.Final = sorted(configured_case_ids - collected_case_ids)
        unknown: typing.Final = sorted(collected_case_ids - configured_case_ids)
        raise pytest.UsageError(f"Report case mapping mismatch: missing={missing}, unknown={unknown}")


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    if report.when != "call" or report.nodeid not in NODE_CASES:
        return
    for case_id in NODE_CASES[report.nodeid]:
        CASE_RESULTS[case_id] = CASE_RESULTS.get(case_id, True) and report.passed


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    del exitstatus
    missing_actuals: typing.Final = reporting.write_results(CASE_RESULTS)
    if missing_actuals:
        session.exitstatus = pytest.ExitCode.TESTS_FAILED
