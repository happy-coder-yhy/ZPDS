"""quality_issues.json 0.2.0 人工审核版结构测试。

覆盖：schema 版本、issue_id 稳定分配、review 区默认值、
原字段完整透传。
"""

from __future__ import annotations

import json
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.writers.quality_writer import (
    SCHEMA_VERSION,
    write_quality_issues,
)


def _make_issue(
    issue_type: str = "timestamp_gap",
    start_ns: int = 0,
    end_ns: int = 500_000_000,
    decision: str = "split",
) -> QualityIssue:
    return QualityIssue(
        issue_type=issue_type,
        stream_id="ego_rgb",
        start_ns=start_ns,
        end_ns=end_ns,
        severity="warning",
        decision=decision,
        details={"gap_s": 0.5, "threshold_s": 0.5},
    )


def _write(tmp_path: Path, issues: list[QualityIssue]) -> dict:
    path = write_quality_issues(
        tmp_path / "quality_issues.json",
        issues,
        source_session_id="session_001",
    )
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_schema_version_is_020(tmp_path: Path) -> None:
    payload = _write(tmp_path, [_make_issue()])
    assert payload["schema_version"] == "0.2.0"
    assert SCHEMA_VERSION == "0.2.0"


def test_issue_id_sequential(tmp_path: Path) -> None:
    issues = [
        _make_issue(issue_type="timestamp_gap", start_ns=0),
        _make_issue(issue_type="imu_gap", start_ns=1_000_000_000),
        _make_issue(issue_type="bad_frame", start_ns=2_000_000_000),
    ]
    payload = _write(tmp_path, issues)
    ids = [entry["issue_id"] for entry in payload["issues"]]
    assert ids == ["iss_000001", "iss_000002", "iss_000003"]


def test_review_pending_default(tmp_path: Path) -> None:
    payload = _write(tmp_path, [_make_issue()])
    assert payload["issues"][0]["review"] == {"status": "pending", "note": ""}


def test_original_fields_preserved(tmp_path: Path) -> None:
    payload = _write(tmp_path, [_make_issue()])
    entry = payload["issues"][0]
    assert entry["issue_type"] == "timestamp_gap"
    assert entry["stream_id"] == "ego_rgb"
    assert entry["start_ns"] == 0
    assert entry["end_ns"] == 500_000_000
    assert entry["duration_ns"] == 500_000_000
    assert entry["severity"] == "warning"
    assert entry["decision"] == "split"
    assert entry["details"] == {"gap_s": 0.5, "threshold_s": 0.5}


def test_issue_id_is_first_key(tmp_path: Path) -> None:
    """issue_id 位于条目最前，平台侧阅读/引用友好。"""
    payload = _write(tmp_path, [_make_issue()])
    entry = payload["issues"][0]
    assert list(entry.keys())[0] == "issue_id"
    assert list(entry.keys())[-1] == "review"


def test_issue_count_and_summary(tmp_path: Path) -> None:
    issues = [_make_issue(decision="split"), _make_issue(decision="keep_with_flag")]
    payload = _write(tmp_path, issues)
    assert payload["issue_count"] == 2
    assert payload["summary"]["total"] == 2
    assert payload["summary"]["by_decision"]["split"] == 1
