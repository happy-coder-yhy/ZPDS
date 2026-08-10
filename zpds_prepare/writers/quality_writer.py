"""
quality_issues.json 写入器（0.2.0 — 人工审核版）。

0.2.0 在 0.1.0 基础上新增，供平台侧人工审核：
- 每条 issue 分配稳定 ``issue_id``（``iss_000001`` 递增），审核按条引用
- 每条 issue 附 ``review`` 区（``status: pending``），平台审核后改为
  approved / rejected / modified，或新增 added 条目（平台自定 id）

平台审核后返回同一结构 JSON，由消费端（main.py ``--review`` /
``scripts/apply_review.py``）应用后重新规划切分。
"""

import json
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import get_issue_summary

SCHEMA_VERSION = "0.2.0"


def write_quality_issues(
    output_path: Path,
    issues: list[QualityIssue],
    source_session_id: str,
) -> Path:
    """将所有 QualityIssue 汇总写入 JSON 文件（0.2.0 人工审核版）。

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
        "schema_version": SCHEMA_VERSION,
        "source_session_id": source_session_id,
        "issue_count": summary["total"],
        "summary": summary,
        "issues": [
            _issue_to_review_dict(issue, index)
            for index, issue in enumerate(issues, start=1)
        ],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path


def _issue_to_review_dict(issue: QualityIssue, index: int) -> dict:
    """0.2.0 审核版 issue 条目：issue_id + 原字段 + review 区。

    ``issue_id`` 按写出顺序稳定分配（``iss_000001`` 递增），审核后
    平台按此 id 精确引用；``review.status`` 初始为 ``pending``。
    """
    return {
        "issue_id": f"iss_{index:06d}",
        **issue.to_dict(),
        "review": {"status": "pending", "note": ""},
    }
