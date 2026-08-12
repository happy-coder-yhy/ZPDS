"""
quality_issues.json 写入器（0.2.1 — 人工审核版 + 状态分层）。

0.2.0 在 0.1.0 基础上新增，供平台侧人工审核：
- 每条 issue 分配稳定 ``issue_id``（``iss_000001`` 递增），审核按条引用
- 每条 issue 附 ``review`` 区（``status: pending``），平台审核后改为
  approved / rejected / modified，或新增 added 条目（平台自定 id）

平台审核后返回同一结构 JSON，由消费端（main.py ``--review`` /
``scripts/apply_review.py``）应用后重新规划切分。

0.2.1（状态分层，对应服务器问题 13「QC FAIL 与 Prepared PASS 语义混用」）：
- 顶层新增 ``quality_status``（数据质量：pass/warn/fail，由 issue 严重度派生）
- 顶层新增 ``processing_status``（处理状态：complete/degraded/failed，由各
  分析环节结果派生）与 ``processing``（逐环节详情）——与 Prepared 阶段的
  ``package_validation`` 解耦，三态不再混叫 PASS/FAIL
"""

import json
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import get_issue_summary

SCHEMA_VERSION = "0.2.1"


def derive_quality_status(issues: list[QualityIssue]) -> str:
    """数据质量状态（quality_status）：error → fail；warn → warn；否则 pass。"""
    severities = {i.severity for i in issues}
    if "error" in severities:
        return "fail"
    if severities & {"warn", "warning"}:
        return "warn"
    return "pass"


def derive_processing_status(
    steps: dict[str, dict] | None,
) -> tuple[str, dict]:
    """处理状态（processing_status）与逐环节详情。

    - complete: 启用环节全部成功（含未启用环节不参与判定）
    - degraded: 有环节降级/跳过（如 Hands degraded、scene 失败降级）
    - failed:   有环节失败（当前架构下失败即中断，产物中极少出现）
    """
    steps = dict(steps or {})
    statuses = {v.get("status") for v in steps.values()}
    if "failed" in statuses:
        return "failed", steps
    if "degraded" in statuses:
        return "degraded", steps
    return "complete", steps


def write_quality_issues(
    output_path: Path,
    issues: list[QualityIssue],
    source_session_id: str,
    *,
    quality_status: str | None = None,
    processing_status: str | None = None,
    processing_steps: dict[str, dict] | None = None,
) -> Path:
    """将所有 QualityIssue 汇总写入 JSON 文件（0.2.1 人工审核版）。

    Args:
        output_path: 输出文件路径 (如 output/quality_issues.json)
        issues: Issue 列表
        source_session_id: 来源 Session ID
        quality_status: 数据质量状态（不传时按 issue 严重度自动派生）
        processing_status: 处理状态（不传时按 processing_steps 派生；
            两者都不传则不写出 processing 区，保持 0.2.0 结构兼容）
        processing_steps: 逐环节详情 {"hands": {"status": ..., "detail": ...}}

    Returns:
        实际写入的文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    summary = get_issue_summary(issues)

    payload = {
        "schema_version": SCHEMA_VERSION,
        "source_session_id": source_session_id,
        "issue_count": summary["total"],
        "quality_status": quality_status or derive_quality_status(issues),
        "summary": summary,
        "issues": [
            _issue_to_review_dict(issue, index)
            for index, issue in enumerate(issues, start=1)
        ],
    }
    if processing_steps is not None or processing_status is not None:
        if processing_status is None:
            processing_status, processing_steps = derive_processing_status(
                processing_steps
            )
        payload["processing_status"] = processing_status
        payload["processing"] = processing_steps or {}

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
