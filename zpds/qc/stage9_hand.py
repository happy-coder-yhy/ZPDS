"""Stage 9: 将手部清洗报告转换为标准 QC Decision。"""

from __future__ import annotations

import json
from pathlib import Path

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.qc.cascade import register_stage

_STAGE9_REASONS = {
    ReasonCode.HAND_ABSENT,
    ReasonCode.HAND_TRACK_LOST,
    ReasonCode.HAND_OCCLUDED,
    ReasonCode.HAND_POSE_INCOMPLETE,
    ReasonCode.NO_OPERATION,
    ReasonCode.FLOW_INCONSISTENT,
}


def check(path: str | Path) -> list[Decision]:
    """读取 ``hand_cleaning_report.json`` 并返回 Stage 9 区间决策。"""
    report_path = Path(path).expanduser().resolve()
    if not report_path.is_file():
        raise FileNotFoundError(f"手部清洗报告不存在: {report_path}")
    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)
    if report.get("schema_version") != "zpds.hand_video_cleaning.v1":
        raise ValueError(f"未知手部清洗报告 schema: {report.get('schema_version')!r}")
    decisions: list[Decision] = []
    for span in report.get("excluded_spans", []):
        try:
            disposition = Disposition(span["disposition"])
            severity = Severity(span.get("severity", Severity.WARN.value))
        except ValueError as error:
            raise ValueError(f"手部清洗报告包含未知枚举值: {span}") from error
        for raw_reason in span.get("reasons", []):
            try:
                reason = ReasonCode(raw_reason)
            except ValueError:
                continue
            if reason not in _STAGE9_REASONS:
                continue
            decisions.append(
                Decision(
                    stage=9,
                    reason=reason,
                    severity=severity,
                    message=(
                        f"Hand QC span [{span['start_frame']}, {span['end_frame']}) "
                        f"was excluded as {disposition.value}"
                    ),
                    frame_idx=int(span["start_frame"]),
                    timestamp_ns=int(span["start_timestamp_ns"]),
                    end_frame_idx=int(span["end_frame"]),
                    end_timestamp_ns=int(span["end_timestamp_ns"]),
                    disposition=disposition,
                    detail={
                        "duration_s": float(span["duration_s"]),
                        "evidence_uri": str(path),
                    },
                )
            )
    return decisions


@register_stage(9)
def _check_stage9(context: dict) -> list[Decision]:
    report_path = context.get("hand_cleaning_report_path")
    if not report_path:
        return []
    return check(report_path)


__all__ = ["check"]
