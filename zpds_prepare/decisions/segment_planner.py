"""
候选 Segment 生成器。

根据 QualityIssue 列表，将原始 Session 切分为
一个或多个候选 Segment。
"""

from dataclasses import dataclass, field
from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue


@dataclass
class CandidateSegment:
    """一个候选 Segment 的描述。"""
    candidate_id: str
    source_start_ns: int
    source_end_ns: int
    duration_ns: int
    reason: str
    issues_in_span: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "source_start_ns": self.source_start_ns,
            "source_end_ns": self.source_end_ns,
            "duration_ns": self.duration_ns,
            "duration_s": round(self.duration_ns / 1_000_000_000, 3),
            "reason": self.reason,
            "issues_in_span": self.issues_in_span,
        }


def plan_segments(
    issues: list[QualityIssue],
    session_start_ns: int,
    session_end_ns: int,
    min_duration_ns: int = 1_000_000_000,
    max_duration_ns: int = 120_000_000_000,
    no_split: bool = False,
) -> list[CandidateSegment]:
    """根据 Issues 生成候选 Segment。

    算法：
    1. 收集需排除的时间范围
    2. 合并首尾或中间的排除范围
    3. 从统一主时间轴中扣除这些范围
    4. 过滤掉过短 / 过长的段

    Args:
        issues: 所有 QualityIssue 的列表
        session_start_ns: 原始 Session 起始时间 (设备时钟)
        session_end_ns: 原始 Session 结束时间 (设备时钟)
        min_duration_ns: 最短有效 Segment（默认 1s）
        max_duration_ns: 最长有效 Segment（默认 120s）
        no_split: True 时由调用方将 split/旧范围动作降级为
                  keep，产出单一连续 segment。

    Returns:
        CandidateSegment 列表，按时间排序；no_split=True 时最多返回一个。
    """
    # ---- 1. 分离旧 trim 和范围排除动作 ----
    # 注：no_split 降级已在调用方通过 downgrade_split_issues() 完成
    head_trims: list[QualityIssue] = []
    tail_trims: list[QualityIssue] = []
    splits: list[QualityIssue] = []

    for issue in issues:
        if issue.decision == "trim":
            # 判断在开头还是结尾
            mid_point = session_start_ns + (session_end_ns - session_start_ns) // 2
            if issue.end_ns <= mid_point:
                head_trims.append(issue)
            else:
                tail_trims.append(issue)
        elif issue.decision in {
            "split",
            "exclude_range",
            "quarantine",
            "keep_with_flag",
        }:
            splits.append(issue)

    # ---- 2. 计算有效范围 ----
    valid_start = session_start_ns
    valid_end = session_end_ns

    if head_trims:
        # 头部裁剪：取最后一个头部 trim 区间的结束点
        valid_start = max(valid_start, max(iss.end_ns for iss in head_trims))

    if tail_trims:
        # 尾部裁剪：取第一个尾部 trim 区间的起始点
        valid_end = min(valid_end, min(iss.start_ns for iss in tail_trims))

    # ---- 3. 收集切分点 ----
    # 每个 split issue 是一个缺口 (start_ns, end_ns)
    # 缺口内部是无效的，缺口两侧是可用的
    split_boundaries: list[tuple[int, int]] = []
    for iss in splits:
        # 只处理在有效范围内的 split
        if iss.start_ns >= valid_start and iss.end_ns <= valid_end:
            split_boundaries.append((iss.start_ns, iss.end_ns))
        elif iss.start_ns < valid_start < iss.end_ns:
            # 缺口跨越有效起点 → 推进起点
            valid_start = max(valid_start, iss.end_ns)
        elif iss.start_ns < valid_end < iss.end_ns:
            # 缺口跨越有效终点 → 回缩终点
            valid_end = min(valid_end, iss.start_ns)

    split_boundaries.sort(key=lambda x: x[0])

    # ---- 4. 生成有效区间 ----
    valid_spans: list[tuple[int, int, str]] = []
    current = valid_start

    for gap_start, gap_end in split_boundaries:
        if current < gap_start:
            valid_spans.append((current, gap_start, "valid_span_before_long_gap"))
        current = max(current, gap_end)

    # 最后一个区间
    if current < valid_end:
        reason = "valid_span_after_long_gap" if valid_spans else "full_session"
        valid_spans.append((current, valid_end, reason))

    if not valid_spans and valid_start < valid_end and not split_boundaries:
        valid_spans.append((valid_start, valid_end, "full_session"))

    if (head_trims or tail_trims) and len(valid_spans) == 1:
        start_ns, end_ns, reason = valid_spans[0]
        if reason == "full_session":
            valid_spans[0] = (
                start_ns,
                end_ns,
                "trimmed_by_quality_boundary",
            )

    # ---- 5. 生成候选 Segment ----
    candidates = []
    for idx, (start_ns, end_ns, reason) in enumerate(valid_spans):
        duration_ns = end_ns - start_ns

        # 过滤过短
        if duration_ns < min_duration_ns:
            continue

        # 超长警告（不强制拒绝）
        truncated = False
        if duration_ns > max_duration_ns:
            # 不出错，但标记
            truncated = True

        # 收集落在这个区间内的保留类 Issues。
        span_issues = []
        for iss in issues:
            if iss.decision == "keep":
                if iss.start_ns >= start_ns and iss.end_ns <= end_ns:
                    span_issues.append(iss.to_dict())
            elif iss.decision in {"split", "exclude_range"}:
                # 排除范围已在上面处理，不再写入有效 Segment。
                pass

        reason_full = reason
        if truncated:
            reason_full += f" (exceeds max_duration, {duration_ns / 1e9:.1f}s)"

        candidates.append(CandidateSegment(
            candidate_id=f"candidate_{idx + 1:06d}",
            source_start_ns=start_ns,
            source_end_ns=end_ns,
            duration_ns=duration_ns,
            reason=reason_full,
            issues_in_span=span_issues,
        ))

    return candidates


def downgrade_split_issues(issues: list[QualityIssue]) -> int:
    """将范围排除决策降级为 keep。

    供 ``no_split`` 模式使用：调用方在 ``get_issue_summary()`` 和
    ``plan_segments()`` 之前调用此函数，确保汇总和分段都反映降级后的状态。

    Returns:
        被降级的 issue 数量。
    """
    count = 0
    for iss in issues:
        if iss.decision in {"split", "exclude_range", "keep_with_flag"}:
            iss.decision = "keep"
            count += 1
    if count:
        import logging
        _log = logging.getLogger(__name__)
        _log.info(
            "no_split 模式：%d 个范围排除决策降级为 keep", count,
        )
    return count


def get_issue_summary(issues: list[QualityIssue]) -> dict:
    """按类型统计 Issue。

    Returns:
        {"total": int, "by_type": {type: count}, "by_decision": {decision: count}}
    """
    by_type: dict[str, int] = {}
    by_decision: dict[str, int] = {}

    for iss in issues:
        by_type[iss.issue_type] = by_type.get(iss.issue_type, 0) + 1
        by_decision[iss.decision] = by_decision.get(iss.decision, 0) + 1

    return {
        "total": len(issues),
        "by_type": by_type,
        "by_decision": by_decision,
    }
