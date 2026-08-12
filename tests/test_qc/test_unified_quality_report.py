"""统一质检报告和审核后执行契约测试。"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import plan_segments
from zpds_prepare.readers.session_model import Session, VideoStream
from zpds_prepare.review.reviewed_report import (
    ReviewValidationError,
    build_reviewed_candidates_document,
    validate_reviewed_report,
)
from zpds_prepare.writers.quality_report_writer import (
    SCHEMA_VERSION,
    compute_immutable_hash,
    validate_generated_report,
    write_quality_report,
)

TOP_LEVEL_KEYS = {
    "schema_version",
    "quality_issue_schema_version",
    "report_metadata",
    "dataset",
    "source_assets",
    "stream_inventory",
    "check_coverage",
    "issues",
    "evidence_index",
    "proposed_cleaning",
    "summary",
    "overall_result",
    "report_review",
    "integrity",
}


def _session(source: Path) -> Session:
    timestamps = [1_000_000_000, 2_000_000_000, 3_000_000_000, 4_000_000_000]
    video = VideoStream(
        stream_id="camera0",
        timestamps_ns=timestamps,
        index_frames=[{"seq": index, "timestamp_ns": value} for index, value in enumerate(timestamps)],
        video_path=str(source),
        fps=1.0,
        width=640,
        height=480,
        frame_count=4,
    )
    return Session(
        session_id="dunjia_sample",
        source_path=str(source),
        meta={},
        video_streams={"camera0": video},
    )


def _issue() -> QualityIssue:
    return QualityIssue(
        issue_type="timestamp_gap",
        stream_id="camera0",
        start_ns=2_000_000_000,
        end_ns=3_000_000_000,
        severity="error",
        decision="split",
        details={"gap_s": 1.0, "threshold_s": 0.5},
    )


def _write(tmp_path: Path) -> tuple[dict, Path]:
    source = tmp_path / "sample.mcap"
    source.write_bytes(b"stable source bytes")
    session = _session(source)
    issues = [_issue()]
    candidates = plan_segments(
        issues,
        session_start_ns=session.session_start_ns,
        session_end_ns=session.session_end_ns,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    report_path = write_quality_report(
        tmp_path / "quality_report.json",
        issues=issues,
        candidates=candidates,
        session=session,
        dataset_path=str(source),
        profile="dunjia",
        cascade_executed=True,
        scene_executed=False,
    )
    return json.loads(report_path.read_text(encoding="utf-8")), source


def _approve(document: dict) -> dict:
    approved = copy.deepcopy(document)
    approved["issues"][0]["review"].update({
        "status": "approved",
        "reviewer_id": "auditor-1",
        "reviewed_at": "2026-08-10T18:00:00+08:00",
        "decision": "accept_recommendation",
        "evidence_checked": True,
    })
    for segment in approved["proposed_cleaning"]["segments"]:
        segment["status"] = "approved"
    approved["report_review"].update({
        "status": "approved",
        "reviewer_id": "auditor-1",
        "reviewed_at": "2026-08-10T18:05:00+08:00",
        "final_result": "approved_for_cleaning",
    })
    return approved


def test_writer_matches_unified_top_level_contract(tmp_path: Path) -> None:
    document, _ = _write(tmp_path)
    assert document["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "zpds.preclean_quality_report.v1.10"
    assert set(document) == TOP_LEVEL_KEYS
    assert "enum_definitions" not in document
    assert document["dataset"]["units"]["length"] == "m"
    assert document["dataset"]["units"]["time"] == "ns"
    assert document["dataset"]["source_type"] == "mcap"
    assert document["check_coverage"]["checks"][-1]["status"] == "not_run"
    assert document["issues"][0]["evidence_refs"] == ["ev_000001"]
    assert document["issues"][0]["decision"] == "split"
    assert document["issues"][0]["proposed_action"]["action"] == "split"
    assert document["issues"][0]["review"]["status"] == "pending"
    assert document["source_assets"][0].keys() == {
        "asset_id",
        "uri",
        "format",
        "readable",
        "size_bytes",
        "sha256",
    }
    assert document["stream_inventory"][0]["source_locator"]
    assert document["stream_inventory"][0]["is_primary"] is True
    assert sum(row["is_primary"] for row in document["stream_inventory"]) == 1
    evidence = document["evidence_index"][0]
    assert evidence["type"] == "timestamp_range"
    assert evidence["locator_method"] == "range_overlap"
    assert "start_ns" in evidence and "end_ns" in evidence
    assert "timestamp_ns" not in evidence
    assert "uri" not in evidence
    assert document["integrity"]["report_content_sha256"]
    validate_generated_report(document)


def test_validator_rejects_issue_outside_stream_timeline(tmp_path: Path) -> None:
    document, _ = _write(tmp_path)
    document["issues"][0]["start_ns"] = 0
    document["issues"][0]["end_ns"] = 1
    document["issues"][0]["duration_ns"] = 1

    with pytest.raises(ValueError, match="时间范围超出对应 Stream"):
        validate_generated_report(document)


def test_umi_marks_both_equal_cameras_primary_independent_of_stream_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "umi_sample.mcap"
    source.write_bytes(b"stable source bytes")
    timestamps = [1_000_000_000, 2_000_000_000]

    def video(stream_id: str) -> VideoStream:
        return VideoStream(
            stream_id=stream_id,
            timestamps_ns=timestamps,
            index_frames=[
                {"seq": index, "timestamp_ns": value}
                for index, value in enumerate(timestamps)
            ],
            video_path=str(source),
            fps=1.0,
            width=640,
            height=480,
            frame_count=2,
        )

    session = Session(
        session_id="umi_sample",
        source_path=str(source),
        meta={},
        video_streams={
            "robot1_camera0": video("robot1_camera0"),
            "robot0_camera0": video("robot0_camera0"),
        },
    )
    report_path = write_quality_report(
        tmp_path / "umi_quality_report.json",
        issues=[],
        candidates=plan_segments(
            [],
            session_start_ns=session.session_start_ns,
            session_end_ns=session.session_end_ns,
            min_duration_ns=1,
            max_duration_ns=10_000_000_000,
        ),
        session=session,
        dataset_path=str(source),
        profile="umi",
        cascade_executed=True,
        scene_executed=False,
    )

    document = json.loads(report_path.read_text(encoding="utf-8"))
    primary_by_stream = {
        row["stream_id"]: row["is_primary"]
        for row in document["stream_inventory"]
    }
    assert primary_by_stream == {
        "robot1_camera0": True,
        "robot0_camera0": True,
    }
    validate_generated_report(document)


def test_point_evidence_uses_timestamp_locator(tmp_path: Path) -> None:
    source = tmp_path / "sample.mcap"
    source.write_bytes(b"stable source bytes")
    session = _session(source)
    issue = QualityIssue(
        issue_type="bad_frame",
        stream_id="camera0",
        start_ns=2_000_000_000,
        end_ns=2_000_000_000,
        severity="error",
        decision="keep_with_flag",
        details={"frame": 1},
    )
    report_path = write_quality_report(
        tmp_path / "point_report.json",
        issues=[issue],
        candidates=plan_segments(
            [issue],
            session_start_ns=session.session_start_ns,
            session_end_ns=session.session_end_ns,
            min_duration_ns=1,
            max_duration_ns=10_000_000_000,
        ),
        session=session,
        dataset_path=str(source),
        profile="dunjia",
        cascade_executed=True,
        scene_executed=False,
    )
    evidence = json.loads(report_path.read_text("utf-8"))["evidence_index"][0]
    document = json.loads(report_path.read_text("utf-8"))
    assert document["issues"][0]["decision"] == "split"
    assert evidence["type"] == "timestamp_point"
    assert evidence["locator_method"] == "nearest_timestamp"
    assert evidence["timestamp_ns"] == 2_000_000_000
    assert evidence["tolerance_ns"] == 1_000_000_000
    assert "start_ns" not in evidence and "end_ns" not in evidence


def test_manual_review_action_uses_unified_issue_type(tmp_path: Path) -> None:
    source = tmp_path / "manual_review.mcap"
    source.write_bytes(b"stable source bytes")
    session = _session(source)
    issue = _issue()
    issue.issue_type = "timestamp_gap"
    issue.decision = "manual_review"

    report_path = write_quality_report(
        tmp_path / "manual_review_report.json",
        issues=[issue],
        candidates=plan_segments(
            [issue],
            session_start_ns=session.session_start_ns,
            session_end_ns=session.session_end_ns,
            min_duration_ns=1,
            max_duration_ns=10_000_000_000,
        ),
        session=session,
        dataset_path=str(source),
        profile="dunjia",
        cascade_executed=True,
        scene_executed=False,
    )

    document = json.loads(report_path.read_text(encoding="utf-8"))
    report_issue = document["issues"][0]
    assert report_issue["issue_type"] == "manual_review_required"
    assert report_issue["details"]["original_issue_type"] == "timestamp_gap"
    validate_generated_report(document)


@pytest.mark.parametrize(
    ("internal_severity", "report_severity"),
    [
        ("warn", "warning"),
        ("info", "warning"),
        ("warning", "warning"),
        ("error", "error"),
        ("critical", "critical"),
    ],
)
def test_writer_normalizes_internal_severity_values(
    tmp_path: Path,
    internal_severity: str,
    report_severity: str,
) -> None:
    source = tmp_path / f"{internal_severity}.mcap"
    source.write_bytes(b"stable source bytes")
    session = _session(source)
    issue = _issue()
    issue.severity = internal_severity

    report_path = write_quality_report(
        tmp_path / f"{internal_severity}_report.json",
        issues=[issue],
        candidates=plan_segments(
            [issue],
            session_start_ns=session.session_start_ns,
            session_end_ns=session.session_end_ns,
            min_duration_ns=1,
            max_duration_ns=10_000_000_000,
        ),
        session=session,
        dataset_path=str(source),
        profile="dunjia",
        cascade_executed=True,
        scene_executed=False,
    )

    document = json.loads(report_path.read_text(encoding="utf-8"))
    assert document["issues"][0]["severity"] == report_severity
    validate_generated_report(document)


@pytest.mark.parametrize(
    ("excluded", "expected"),
    [
        ((0, 2), [(2, 10)]),
        ((4, 6), [(0, 4), (6, 10)]),
        ((8, 10), [(0, 8)]),
        ((0, 10), []),
    ],
)
def test_split_range_derives_boundary_or_middle_segments(
    excluded: tuple[int, int],
    expected: list[tuple[int, int]],
) -> None:
    issue = QualityIssue(
        issue_type="timestamp_gap",
        stream_id="camera0",
        start_ns=excluded[0],
        end_ns=excluded[1],
        severity="error",
        decision="split",
    )
    candidates = plan_segments(
        [issue],
        session_start_ns=0,
        session_end_ns=10,
        min_duration_ns=1,
        max_duration_ns=100,
    )
    assert [(row.source_start_ns, row.source_end_ns) for row in candidates] == expected


def test_review_fields_can_change_without_breaking_immutable_hash(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    approved = _approve(document)
    validate_reviewed_report(approved, profile="dunjia", dataset_path=source)
    candidates = build_reviewed_candidates_document(
        approved,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    assert candidates["candidate_count"] == 2
    assert candidates["review_report_id"] == document["report_metadata"]["report_id"]


def test_modified_action_is_applied(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    approved = _approve(document)
    review = approved["issues"][0]["review"]
    review["status"] = "modified"
    review["decision"] = "modify_recommendation"
    review["modified_action"] = "keep"
    validate_reviewed_report(approved, profile="dunjia", dataset_path=source)
    candidates = build_reviewed_candidates_document(
        approved,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    assert candidates["candidate_count"] == 1
    assert candidates["segments"][0]["issues_in_span"][0]["decision"] == "keep"


def test_current_schema_rejects_keep_with_flag(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    approved = _approve(document)
    review = approved["issues"][0]["review"]
    review["status"] = "modified"
    review["decision"] = "modify_recommendation"
    review["modified_action"] = "keep_with_flag"
    with pytest.raises(ReviewValidationError, match="动作非法"):
        validate_reviewed_report(approved, profile="dunjia", dataset_path=source)


def test_v16_trim_is_accepted_as_split(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    legacy = copy.deepcopy(document)
    legacy["schema_version"] = "zpds.preclean_quality_report.v1.6"
    legacy["issues"][0]["decision"] = "trim"
    legacy["issues"][0]["proposed_action"]["action"] = "trim"
    legacy["integrity"]["report_content_sha256"] = compute_immutable_hash(legacy)
    approved = _approve(legacy)
    validate_reviewed_report(approved, profile="dunjia", dataset_path=source)
    candidates = build_reviewed_candidates_document(
        approved,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    assert candidates["candidate_count"] == 2


def test_v17_exclude_range_is_accepted_as_split(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    legacy = copy.deepcopy(document)
    legacy["schema_version"] = "zpds.preclean_quality_report.v1.7"
    legacy["issues"][0]["decision"] = "exclude_range"
    legacy["issues"][0]["proposed_action"]["action"] = "exclude_range"
    legacy["integrity"]["report_content_sha256"] = compute_immutable_hash(legacy)
    approved = _approve(legacy)
    validate_reviewed_report(approved, profile="dunjia", dataset_path=source)
    candidates = build_reviewed_candidates_document(
        approved,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    assert candidates["candidate_count"] == 2


def test_v18_keep_with_flag_is_accepted_as_split(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    legacy = copy.deepcopy(document)
    legacy["schema_version"] = "zpds.preclean_quality_report.v1.8"
    legacy["issues"][0]["decision"] = "keep_with_flag"
    legacy["issues"][0]["proposed_action"]["action"] = "keep_with_flag"
    legacy["integrity"]["report_content_sha256"] = compute_immutable_hash(legacy)
    approved = _approve(legacy)
    validate_reviewed_report(approved, profile="dunjia", dataset_path=source)
    candidates = build_reviewed_candidates_document(
        approved,
        min_duration_ns=1,
        max_duration_ns=10_000_000_000,
    )
    assert candidates["candidate_count"] == 2


def test_immutable_detection_content_cannot_be_modified(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    approved = _approve(document)
    approved["issues"][0]["start_ns"] += 1
    with pytest.raises(ReviewValidationError, match="哈希不一致"):
        validate_reviewed_report(approved, profile="dunjia", dataset_path=source)


def test_different_source_is_rejected(tmp_path: Path) -> None:
    document, _ = _write(tmp_path)
    approved = _approve(document)
    different = tmp_path / "different.mcap"
    different.write_bytes(b"different source")
    with pytest.raises(ReviewValidationError, match="源数据 SHA256 不一致"):
        validate_reviewed_report(approved, profile="dunjia", dataset_path=different)


def test_pending_report_cannot_execute(tmp_path: Path) -> None:
    document, source = _write(tmp_path)
    with pytest.raises(ReviewValidationError, match="必须为 approved"):
        validate_reviewed_report(document, profile="dunjia", dataset_path=source)
