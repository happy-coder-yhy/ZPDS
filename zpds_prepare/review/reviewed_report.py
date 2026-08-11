"""校验平台返回的统一质检报告，并生成审核后的候选切分。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import plan_segments
from zpds_prepare.quality_report_contract import (
    ALLOWED_ACTIONS,
    ALLOWED_REPORT_FINAL_RESULTS,
    ALLOWED_REVIEW_DECISIONS,
    ALLOWED_REVIEW_REASON_CODES,
    SCHEMA_VERSION,
    SUPPORTED_REVIEW_SCHEMA_VERSIONS,
    normalize_action,
)
from zpds_prepare.writers.quality_report_writer import (
    compute_immutable_hash,
    sha256_path,
    validate_generated_report,
)


class ReviewValidationError(ValueError):
    """审核报告不完整、不一致或不允许执行。"""


def load_reviewed_report(path: str | Path) -> dict[str, Any]:
    review_path = Path(path)
    try:
        document = json.loads(review_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReviewValidationError(f"无法读取审核报告 {review_path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ReviewValidationError("审核报告顶层必须是 JSON object")
    return document


def validate_reviewed_report(
    document: dict[str, Any],
    *,
    profile: str,
    dataset_path: str | Path,
) -> None:
    errors: list[str] = []
    report_schema = document.get("schema_version")
    if report_schema not in SUPPORTED_REVIEW_SCHEMA_VERSIONS:
        errors.append(
            "schema_version 必须为 "
            + " 或 ".join(sorted(SUPPORTED_REVIEW_SCHEMA_VERSIONS))
        )
    else:
        validation_document = json.loads(json.dumps(document, ensure_ascii=False))
        validation_document["schema_version"] = SCHEMA_VERSION
        if report_schema != SCHEMA_VERSION:
            for issue in validation_document.get("issues") or []:
                issue["decision"] = normalize_action(str(issue.get("decision")))
                proposed = issue.get("proposed_action") or {}
                proposed["action"] = normalize_action(str(proposed.get("action")))
            overall = validation_document.get("overall_result") or {}
            overall["recommended_action"] = normalize_action(
                str(overall.get("recommended_action"))
            )
        try:
            validate_generated_report(validation_document)
        except ValueError as exc:
            errors.append(str(exc))

    dataset = document.get("dataset") or {}
    if dataset.get("profile") != profile:
        errors.append(
            f"profile 不一致: 报告={dataset.get('profile')!r}, 命令行={profile!r}"
        )
    time_range = dataset.get("time_range") or {}
    start_ns = time_range.get("start_ns")
    end_ns = time_range.get("end_ns")
    if not isinstance(start_ns, int) or not isinstance(end_ns, int) or start_ns >= end_ns:
        errors.append("dataset.time_range 非法")

    expected_hash = (document.get("integrity") or {}).get("report_content_sha256")
    actual_hash = compute_immutable_hash(document)
    if not expected_hash or expected_hash != actual_hash:
        errors.append("报告不可修改内容哈希不一致")

    source_assets = document.get("source_assets") or []
    if not source_assets or not source_assets[0].get("sha256"):
        errors.append("缺少源数据 SHA256")
    else:
        try:
            source_hash = sha256_path(dataset_path)
        except (OSError, FileNotFoundError) as exc:
            errors.append(str(exc))
        else:
            if source_hash != source_assets[0]["sha256"]:
                errors.append("当前 --dataset 与质检时源数据 SHA256 不一致")
    dataset_source = Path(dataset_path).resolve()
    source_root = dataset_source if dataset_source.is_dir() else dataset_source.parent
    for asset in source_assets[1:]:
        asset_uri = Path(str(asset.get("uri", "")))
        asset_path = asset_uri if asset_uri.is_absolute() else source_root / asset_uri
        try:
            asset_hash = sha256_path(asset_path)
        except (OSError, FileNotFoundError) as exc:
            errors.append(str(exc))
        else:
            if asset_hash != asset.get("sha256"):
                errors.append(f"源资产 {asset.get('asset_id')} SHA256 不一致")

    report_review = document.get("report_review") or {}
    if report_review.get("status") != "approved":
        errors.append("report_review.status 必须为 approved")
    if report_review.get("final_result") != "approved_for_cleaning":
        errors.append("report_review.final_result 必须批准进入清洗")
    elif report_review.get("final_result") not in ALLOWED_REPORT_FINAL_RESULTS:
        errors.append("report_review.final_result 非法")
    if not report_review.get("reviewer_id") or not report_review.get("reviewed_at"):
        errors.append("报告级审核缺少 reviewer_id 或 reviewed_at")

    issue_ids: set[str] = set()
    for issue in document.get("issues") or []:
        issue_id = issue.get("issue_id")
        if not issue_id or issue_id in issue_ids:
            errors.append(f"Issue ID 缺失或重复: {issue_id!r}")
            continue
        issue_ids.add(issue_id)
        review = issue.get("review") or {}
        status = review.get("status")
        if status not in {"approved", "rejected", "modified"}:
            errors.append(f"{issue_id} 未完成审核: {status!r}")
            continue
        if not review.get("reviewer_id") or not review.get("reviewed_at"):
            errors.append(f"{issue_id} 缺少 reviewer_id 或 reviewed_at")
        decision = review.get("decision")
        expected_decision = {
            "approved": "accept_recommendation",
            "rejected": "reject_issue",
            "modified": "modify_recommendation",
        }.get(status)
        if decision not in ALLOWED_REVIEW_DECISIONS or decision != expected_decision:
            errors.append(
                f"{issue_id} review.decision 与 status 不一致: {decision!r}"
            )
        reason_code = review.get("reason_code")
        if reason_code is not None and reason_code not in ALLOWED_REVIEW_REASON_CODES:
            errors.append(f"{issue_id} reason_code 非法: {reason_code!r}")
        if status in {"approved", "modified"} and not review.get("evidence_checked"):
            errors.append(f"{issue_id} 尚未确认 Evidence")
        raw_action = (
            issue.get("decision")
            if status == "approved"
            else review.get("modified_action")
        )
        action = (
            normalize_action(str(raw_action))
            if report_schema != SCHEMA_VERSION
            else str(raw_action)
        )
        if status == "rejected":
            continue
        if action not in ALLOWED_ACTIONS:
            errors.append(f"{issue_id} 动作非法: {action!r}")
        if action == "manual_review":
            errors.append(f"{issue_id} 仍要求人工审核")
        issue_start = issue.get("start_ns")
        issue_end = issue.get("end_ns")
        if (
            isinstance(start_ns, int)
            and isinstance(end_ns, int)
            and (
                not isinstance(issue_start, int)
                or not isinstance(issue_end, int)
                or issue_start < start_ns
                or issue_end > end_ns
                or issue_start > issue_end
            )
        ):
            errors.append(f"{issue_id} 时间范围超出 Session")

    for segment in (document.get("proposed_cleaning") or {}).get("segments") or []:
        if segment.get("status") not in {"approved", "rejected", "modified"}:
            errors.append(
                f"{segment.get('candidate_id')} 未完成审核: {segment.get('status')!r}"
            )

    if errors:
        raise ReviewValidationError("审核报告校验失败:\n- " + "\n- ".join(errors))


def _reviewed_issues(document: dict[str, Any]) -> list[QualityIssue]:
    reviewed: list[QualityIssue] = []
    for row in document.get("issues") or []:
        review = row["review"]
        status = review["status"]
        if status == "rejected":
            continue
        raw_action = (
            row["decision"]
            if status == "approved"
            else review["modified_action"]
        )
        action = (
            normalize_action(str(raw_action))
            if document.get("schema_version") != SCHEMA_VERSION
            else str(raw_action)
        )
        if action == "reject":
            raise ReviewValidationError(f"{row['issue_id']} 拒绝整个 Session，不能生成 Segment")
        details = dict(row.get("details") or {})
        details["review_status"] = status
        details["reviewer_id"] = review.get("reviewer_id")
        details["review_note"] = review.get("note", "")
        details["reviewed_action"] = action
        # 隔离区和普通排除区都从本次正式清洗结果中扣除。
        planning_action = "split" if action == "quarantine" else action
        reviewed.append(QualityIssue(
            issue_type=str(details.get("original_issue_type", row["issue_type"])),
            stream_id=str(row["stream_id"]),
            start_ns=int(row["start_ns"]),
            end_ns=int(row["end_ns"]),
            severity=str(row["severity"]),
            decision=str(planning_action),
            details=details,
        ))
    return reviewed


def build_reviewed_candidates_document(
    document: dict[str, Any],
    *,
    min_duration_ns: int,
    max_duration_ns: int,
) -> dict[str, Any]:
    """把审核后的统一报告转换为 batch_prepare 内部候选文档。"""

    time_range = document["dataset"]["time_range"]
    start_ns = int(time_range["start_ns"])
    end_ns = int(time_range["end_ns"])
    issues = _reviewed_issues(document)
    candidates = plan_segments(
        issues=issues,
        session_start_ns=start_ns,
        session_end_ns=end_ns,
        min_duration_ns=min_duration_ns,
        max_duration_ns=max_duration_ns,
        no_split=False,
    )
    return {
        "schema_version": "0.1.0-reviewed",
        "source_session_id": document["dataset"]["source_session_id"],
        "source_start_ns": start_ns,
        "source_end_ns": end_ns,
        "source_duration_s": round((end_ns - start_ns) / 1_000_000_000, 3),
        "candidate_count": len(candidates),
        "total_effective_duration_s": round(
            sum(candidate.duration_ns for candidate in candidates) / 1_000_000_000,
            3,
        ),
        "review_report_id": document["report_metadata"]["report_id"],
        "segments": [candidate.to_dict() for candidate in candidates],
    }


__all__ = [
    "ReviewValidationError",
    "build_reviewed_candidates_document",
    "load_reviewed_report",
    "validate_reviewed_report",
]
