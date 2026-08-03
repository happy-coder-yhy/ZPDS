import json

import pytest

from zpds.core.decisions import Disposition, ReasonCode
from zpds.qc import get_stage_checker
from zpds.qc.stage9_hand import check


def test_stage9_converts_only_hand_reasons(tmp_path) -> None:
    report_path = tmp_path / "hand_cleaning_report.json"
    report_path.write_text(
        json.dumps(
            {
                "schema_version": "zpds.hand_video_cleaning.v1",
                "excluded_spans": [
                    {
                        "start_frame": 10,
                        "end_frame": 20,
                        "start_timestamp_ns": 1_000_000_000,
                        "end_timestamp_ns": 2_000_000_000,
                        "duration_s": 1.0,
                        "reasons": ["black_frame", "hand_absent", "no_operation"],
                        "severity": "warn",
                        "disposition": "split",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    decisions = check(report_path)

    assert [decision.reason for decision in decisions] == [
        ReasonCode.HAND_ABSENT,
        ReasonCode.NO_OPERATION,
    ]
    assert all(decision.disposition is Disposition.SPLIT for decision in decisions)
    assert all(decision.end_frame_idx == 20 for decision in decisions)


def test_stage9_is_registered_and_not_applicable_without_report() -> None:
    checker = get_stage_checker(9)
    assert checker is not None
    assert checker({}) == []


def test_stage9_rejects_unknown_schema(tmp_path) -> None:
    report_path = tmp_path / "report.json"
    report_path.write_text('{"schema_version": "unknown"}', encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        check(report_path)
