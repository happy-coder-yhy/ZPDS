"""
平台审核结果应用器（人工审核闭环消费端）。

0.2.0 审核版 quality_issues.json 由平台审核后返回，``review.status`` 取值：
- ``approved``: 条目原样保留（原决策不变）
- ``rejected``: 移除该 issue——重新切分时自然不含它
- ``modified``: 平台直接修改条目字段（decision/severity/start_ns/end_ns/details），
  以审核版字段重建（与平台约定：修改直接改条目字段，不在 review 区写覆盖值）
- ``added``: 平台新增条目（issue_id 平台自定，不在原集合中即视为新增）
- 未知/缺失 status: 按 pending 容错，保留原样

入口：
- main.py ``--review <path>``: 管线内应用审核结果后重新 plan_segments
- scripts/apply_review.py: 独立 CLI（联调用）
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue

VALID_DECISIONS = {"trim", "split", "keep_with_flag", "quarantine"}
VALID_SEVERITIES = {"info", "warning", "error", "critical"}


@dataclass
class ReviewStats:
    """审核应用统计。"""

    approved: int = 0
    rejected: int = 0
    modified: int = 0
    added: int = 0
    kept: int = 0

    @property
    def total(self) -> int:
        return self.approved + self.rejected + self.modified + self.added + self.kept

    def to_dict(self) -> dict[str, int]:
        return {
            "approved": self.approved,
            "rejected": self.rejected,
            "modified": self.modified,
            "added": self.added,
            "kept": self.kept,
            "total": self.total,
        }


def apply_review(
    original: list[QualityIssue],
    reviewed_payload: dict[str, Any],
) -> tuple[list[QualityIssue], ReviewStats]:
    """按平台审核结果合并 issues。

    Args:
        original: 原始 QualityIssue 列表（写出 quality_issues.json 的那份）
        reviewed_payload: 平台返回的 0.2.0 审核版 json（顶层含 ``issues`` 数组）

    Returns:
        (合并后的 QualityIssue 列表, 应用统计)，供 plan_segments 消费。
    """
    original_by_id = {
        f"iss_{index:06d}": issue
        for index, issue in enumerate(original, start=1)
    }
    entries = reviewed_payload.get("issues") or []
    stats = ReviewStats()
    result: list[QualityIssue] = []

    for entry in entries:
        issue_id = str(entry.get("issue_id", ""))
        status = str((entry.get("review") or {}).get("status", "pending")).lower()
        orig = original_by_id.get(issue_id)

        # rejected：直接移除（无论是否原集合中的条目）
        if status == "rejected":
            stats.rejected += 1
            continue

        rebuilt = _issue_from_entry(entry)

        # modified：以审核版字段重建；字段非法时容错保留原样
        if status == "modified" and orig is not None:
            if rebuilt is None:
                result.append(orig)
                stats.kept += 1
            else:
                result.append(rebuilt)
                stats.modified += 1
            continue

        # 原集合中的条目：approved / pending / 未知 status 均保留原决策
        if orig is not None:
            result.append(orig)
            if status == "approved":
                stats.approved += 1
            else:
                stats.kept += 1
            continue

        # 原集合中不存在 → 平台新增（added）；字段非法则丢弃并计入 kept 外（无法表示，直接忽略）
        if rebuilt is not None:
            result.append(rebuilt)
            stats.added += 1

    return result, stats


def _issue_from_entry(entry: dict[str, Any]) -> QualityIssue | None:
    """从审核版条目重建 QualityIssue；字段缺失/非法返回 None。"""
    try:
        decision = str(entry["decision"])
        if decision not in VALID_DECISIONS:
            return None
        severity = str(entry.get("severity", "warning"))
        if severity not in VALID_SEVERITIES:
            return None
        start_ns = int(entry.get("start_ns", 0))
        end_ns = int(entry.get("end_ns", start_ns))
        if end_ns < start_ns:
            end_ns = start_ns
        return QualityIssue(
            issue_type=str(entry["issue_type"]),
            stream_id=str(entry.get("stream_id", "")),
            start_ns=start_ns,
            end_ns=end_ns,
            severity=severity,
            decision=decision,
            details=dict(entry.get("details") or {}),
        )
    except (KeyError, TypeError, ValueError):
        return None


__all__ = ["ReviewStats", "apply_review"]
