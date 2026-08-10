"""平台审核结果应用测试（人工审核闭环消费端）。

覆盖：approved / rejected / modified / added 四种审核结果、
未知 status 容错、非法字段容错、与 0.2.0 writer 的往返一致性。
"""

from __future__ import annotations

import json
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.writers.quality_writer import write_quality_issues
from zpds_prepare.writers.review_applier import ReviewStats, apply_review


def _issue(
    issue_type: str,
    decision: str = "keep_with_flag",
    start_ns: int = 0,
    end_ns: int = 1000,
) -> QualityIssue:
    return QualityIssue(
        issue_type=issue_type,
        stream_id="ego_rgb",
        start_ns=start_ns,
        end_ns=end_ns,
        severity="warning",
        decision=decision,
        details={"k": "v"},
    )


def _original_three() -> list[QualityIssue]:
    return [
        _issue("timestamp_gap", decision="split", start_ns=0),
        _issue("imu_gap", decision="split", start_ns=2_000_000_000),
        _issue("bad_frame", decision="keep_with_flag", start_ns=4_000_000_000),
    ]


def _review_entry(issue_id: str, status: str, **overrides) -> dict:
    entry = {
        "issue_id": issue_id,
        "issue_type": "timestamp_gap",
        "stream_id": "ego_rgb",
        "start_ns": 0,
        "end_ns": 1000,
        "severity": "warning",
        "decision": "keep_with_flag",
        "details": {},
        "review": {"status": status, "note": ""},
    }
    entry.update(overrides)
    return entry


def test_approved_keeps_original_decision() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "approved", decision="split"),
        _review_entry("iss_000002", "approved", decision="split"),
        _review_entry("iss_000003", "approved", decision="keep_with_flag"),
    ]}
    merged, stats = apply_review(original, payload)
    assert [i.issue_type for i in merged] == ["timestamp_gap", "imu_gap", "bad_frame"]
    # 原决策保持不变（即使审核版里写了别的 decision，approved 不采纳）
    assert merged[0].decision == "split"
    assert stats.approved == 3
    assert stats.total == 3


def test_rejected_removes_issue() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "approved"),
        _review_entry("iss_000002", "rejected", note="误报"),
        _review_entry("iss_000003", "approved"),
    ]}
    merged, stats = apply_review(original, payload)
    assert [i.issue_type for i in merged] == ["timestamp_gap", "bad_frame"]
    assert stats.rejected == 1


def test_modified_rebuilds_from_entry_fields() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "modified", decision="trim", severity="error",
                      start_ns=500, end_ns=800, details={"note": "平台调整"}),
        _review_entry("iss_000002", "approved"),
        _review_entry("iss_000003", "approved"),
    ]}
    merged, stats = apply_review(original, payload)
    assert stats.modified == 1
    m = merged[0]
    assert m.issue_type == "timestamp_gap"
    assert m.decision == "trim"
    assert m.severity == "error"
    assert (m.start_ns, m.end_ns) == (500, 800)
    assert m.details == {"note": "平台调整"}


def test_added_new_entry() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "approved"),
        _review_entry("iss_000002", "approved"),
        _review_entry("iss_000003", "approved"),
        _review_entry("iss_100001", "added", issue_type="hand_occluded",
                      decision="split", start_ns=9_000_000_000, end_ns=9_500_000_000),
    ]}
    merged, stats = apply_review(original, payload)
    assert stats.added == 1
    assert [i.issue_type for i in merged] == [
        "timestamp_gap", "imu_gap", "bad_frame", "hand_occluded",
    ]
    assert merged[-1].decision == "split"
    assert merged[-1].start_ns == 9_000_000_000


def test_pending_or_unknown_status_keeps_original() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "approved"),
        _review_entry("iss_000002", "whatever_status"),
        _review_entry("iss_000003", "pending"),
    ]}
    merged, stats = apply_review(original, payload)
    assert len(merged) == 3
    assert stats.kept == 2
    assert [i.decision for i in merged] == ["split", "split", "keep_with_flag"]


def test_modified_with_invalid_decision_falls_back_to_original() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "modified", decision="not_a_real_decision"),
        _review_entry("iss_000002", "approved"),
        _review_entry("iss_000003", "approved"),
    ]}
    merged, stats = apply_review(original, payload)
    # 平台改坏字段 → 容错保留原样，不丢弃
    assert len(merged) == 3
    assert merged[0].decision == "split"
    assert stats.kept == 1


def test_rejected_of_unknown_id_skipped() -> None:
    original = _original_three()
    payload = {"issues": [
        _review_entry("iss_000001", "approved"),
        _review_entry("iss_000002", "approved"),
        _review_entry("iss_000003", "approved"),
        _review_entry("iss_999999", "rejected"),
    ]}
    merged, stats = apply_review(original, payload)
    assert len(merged) == 3
    assert stats.rejected == 1


def test_roundtrip_with_writer(tmp_path: Path) -> None:
    """0.2.0 writer 写出 → 平台审核（1 approved + 1 rejected）→ apply 结果正确。"""
    original = _original_three()
    path = write_quality_issues(
        tmp_path / "quality_issues.json", original, source_session_id="s1"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    # 模拟平台：修改第 2 条为 rejected，第 1 条改 decision
    payload["issues"][1]["review"]["status"] = "rejected"
    payload["issues"][0]["review"]["status"] = "modified"
    payload["issues"][0]["decision"] = "trim"

    merged, stats = apply_review(original, payload)
    assert stats.rejected == 1
    assert stats.modified == 1
    assert [i.issue_type for i in merged] == ["timestamp_gap", "bad_frame"]
    assert merged[0].decision == "trim"


def test_stats_dict() -> None:
    stats = ReviewStats(approved=2, rejected=1)
    assert stats.to_dict() == {
        "approved": 2, "rejected": 1, "modified": 0, "added": 0,
        "kept": 0, "total": 3,
    }
