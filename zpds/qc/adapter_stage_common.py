"""Adapter-backed Stage 0～2 的共享序列化与 Decision 映射。"""

from pathlib import Path
from typing import Any

from zpds.adapters import IssueLevel, ValidationIssue, ValidationReport
from zpds.config import DEFAULT_SCHEMA_REGISTRY
from zpds.core.decisions import (
    Decision,
    DecisionType,
    Evidence,
    ReasonCode,
    Severity,
)
from zpds.core.types import SessionInventory
from zpds.pipeline import StageContext
from zpds.storage import LocalStorage


def raw_session_path(storage: LocalStorage, context: StageContext) -> tuple[str, Path]:
    for reference in context.input_refs:
        if reference.startswith("raw://"):
            return reference, storage.raw_path(reference)
    raise ValueError("StageContext must retain at least one raw:// session reference")


def retained_outputs(context: StageContext, report_reference: str) -> tuple[str, ...]:
    raw_refs = tuple(item for item in context.input_refs if item.startswith("raw://"))
    return (*raw_refs, report_reference)


def inventory_to_dict(inventory: SessionInventory) -> dict[str, Any]:
    return {
        "session_id": inventory.session_id,
        "source_profile": inventory.source_profile,
        "session_uri": inventory.session_uri,
        "assets": [
            {
                "asset_id": asset.asset_id,
                "uri": asset.uri,
                "relative_path": asset.relative_path,
                "size_bytes": asset.size_bytes,
                "sha256": asset.sha256,
                "media_type": asset.media_type,
                "required": asset.required,
            }
            for asset in inventory.assets
        ],
        "streams": [
            {
                "stream_id": stream.stream_id,
                "kind": stream.kind.value,
                "role": stream.role,
                "clock_id": stream.clock_id,
                "width": stream.width,
                "height": stream.height,
                "fps": stream.fps,
                "sample_rate_hz": stream.sample_rate_hz,
                "codec": stream.codec,
                "container": stream.container,
                "topic": stream.topic,
                "encoding": stream.encoding,
                "dtype": stream.dtype,
                "frame_id": stream.frame_id,
                "metadata": stream.metadata,
            }
            for stream in inventory.streams
        ],
        "clocks": [
            {
                "clock_id": clock.clock_id,
                "domain": clock.domain.value,
                "source": clock.source,
                "unit": clock.unit,
                "authoritative": clock.authoritative,
                "notes": clock.notes,
            }
            for clock in inventory.clocks
        ],
        "calibrations": [
            {
                "calibration_id": calibration.calibration_id,
                "kind": calibration.kind,
                "uri": calibration.uri,
                "parent_frame": calibration.parent_frame,
                "child_frame": calibration.child_frame,
                "format": calibration.format,
                "source_recorded": calibration.source_recorded,
                "metadata": calibration.metadata,
            }
            for calibration in inventory.calibrations
        ],
        "total_frames": inventory.total_frames,
        "duration_s": inventory.duration_s,
        "clock_domain": inventory.clock_domain.value,
        "metadata": inventory.metadata,
    }


def report_to_dict(report: ValidationReport) -> dict[str, Any]:
    return {
        "passed": report.passed,
        "checked_assets": report.checked_assets,
        "checked_records": report.checked_records,
        "decoded_records": report.decoded_records,
        "issues": [
            {
                "code": issue.code,
                "level": issue.level.value,
                "message": issue.message,
                "path": issue.path,
                "stream_id": issue.stream_id,
                "details": issue.details,
            }
            for issue in report.issues
        ],
        "metadata": report.metadata,
    }


def validate_source_inventory(value: dict[str, Any]) -> None:
    _validate_registered(value, "source_inventory")


def validate_validation_report(value: dict[str, Any]) -> None:
    _validate_registered(value, "validation_report")


def _validate_registered(value: dict[str, Any], object_type: str) -> None:
    errors = DEFAULT_SCHEMA_REGISTRY.validate(value, object_type, "0.1.0")
    if errors:
        raise ValueError(f"Invalid {object_type}: {'; '.join(errors)}")


def decisions_from_report(
    report: ValidationReport,
    *,
    stage_id: int,
    evidence_uri: str,
) -> tuple[Decision, ...]:
    return tuple(
        _decision_from_issue(issue, stage_id=stage_id, evidence_uri=evidence_uri)
        for issue in report.issues
    )


def _decision_from_issue(
    issue: ValidationIssue,
    *,
    stage_id: int,
    evidence_uri: str,
) -> Decision:
    severity = {
        IssueLevel.INFO: Severity.INFO,
        IssueLevel.WARN: Severity.WARN,
        IssueLevel.ERROR: Severity.ERROR,
        IssueLevel.FATAL: Severity.FATAL,
    }[issue.level]
    decision = {
        IssueLevel.INFO: DecisionType.KEEP,
        IssueLevel.WARN: DecisionType.KEEP_WITH_FLAG,
        IssueLevel.ERROR: DecisionType.QUARANTINE,
        IssueLevel.FATAL: DecisionType.REJECT,
    }[issue.level]
    reason = _reason_code(issue.code, stage_id)
    evidence = Evidence(
        uri=evidence_uri,
        kind=f"stage_{stage_id}_validation",
        description=issue.message,
    )
    return Decision(
        stage=stage_id,
        reason=reason,
        severity=severity,
        decision=decision,
        message="" if decision is DecisionType.KEEP else issue.message,
        evidence=[] if decision is DecisionType.KEEP else [evidence],
        detail={"adapter_code": issue.code, **issue.details},
    )


def _reason_code(code: str, stage_id: int) -> ReasonCode:
    exact = {
        "required_file_missing": ReasonCode.FILE_MISSING,
        "guida_session_missing": ReasonCode.FILE_MISSING,
        "a2d_session_missing": ReasonCode.FILE_MISSING,
        "segment_container_missing": ReasonCode.REQUIRED_STREAM_MISSING,
        "imu_file_missing": ReasonCode.REQUIRED_STREAM_MISSING,
        "imu_declared_path_missing": ReasonCode.REQUIRED_STREAM_MISSING,
        "timestamp_regression": ReasonCode.TIMESTAMP_REGRESSION,
        "timestamp_gap": ReasonCode.TIMESTAMP_GAP,
        "untrusted_pickle": ReasonCode.UNTRUSTED_INPUT,
        "a2d_camera_tuple_incomplete": ReasonCode.REQUIRED_STREAM_MISSING,
    }
    if code in exact:
        return exact[code]
    if stage_id == 0:
        return ReasonCode.FILE_MISSING
    if stage_id == 2:
        return ReasonCode.CLOCK_MISALIGN
    return ReasonCode.CONTAINER_CORRUPT
