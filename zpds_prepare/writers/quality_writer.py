"""
quality_issues.json 写入器。
"""

import json
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import get_issue_summary


def write_quality_issues(
    output_path: Path,
    issues: list[QualityIssue],
    source_session_id: str,
) -> Path:
    """将所有 QualityIssue 汇总写入 JSON 文件。

    Args:
        output_path: 输出文件路径 (如 output/quality_issues.json)
        issues: Issue 列表
        source_session_id: 来源 Session ID

    Returns:
        实际写入的文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = get_issue_summary(issues)

    payload = {
        "schema_version": "0.1.0",
        "source_session_id": source_session_id,
        "issue_count": summary["total"],
        "summary": summary,
        "issues": [issue.to_dict() for issue in issues],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path
