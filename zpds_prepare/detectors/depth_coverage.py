"""检查必需深度流对 RGB 公共时间轴的覆盖情况。"""

from __future__ import annotations

from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue


def detect_depth_coverage(
    video_streams: dict[str, Any],
    depth_streams: dict[str, Any],
    tolerance_ns: int = 80_000_000,
) -> list[QualityIssue]:
    """对比深度范围与所有 RGB 流的公共时间范围。

    深度晚开始或早结束时生成 ``trim`` issue，让现有 Segment Planner
    自动收缩候选边界；Raw 数据不会被删除。
    """
    videos = [
        stream
        for stream in video_streams.values()
        if stream.timestamps_ns
    ]
    if not videos or not depth_streams:
        return []

    visual_start_ns = max(int(stream.timestamps_ns[0]) for stream in videos)
    visual_end_ns = min(int(stream.timestamps_ns[-1]) for stream in videos)
    if visual_end_ns <= visual_start_ns:
        return []

    issues: list[QualityIssue] = []
    visual_duration_ns = visual_end_ns - visual_start_ns

    for stream_id, depth in depth_streams.items():
        if not depth.timestamps_ns:
            continue
        depth_start_ns = int(depth.timestamps_ns[0])
        depth_end_ns = int(depth.timestamps_ns[-1])
        overlap_start_ns = max(visual_start_ns, depth_start_ns)
        overlap_end_ns = min(visual_end_ns, depth_end_ns)
        overlap_ns = max(0, overlap_end_ns - overlap_start_ns)
        coverage_ratio = overlap_ns / visual_duration_ns
        common_details = {
            "visual_stream_ids": sorted(video_streams),
            "visual_start_ns": visual_start_ns,
            "visual_end_ns": visual_end_ns,
            "depth_start_ns": depth_start_ns,
            "depth_end_ns": depth_end_ns,
            "coverage_ratio": round(coverage_ratio, 6),
            "tolerance_ns": tolerance_ns,
            "boundary_policy": "required_visual_overlap",
        }

        if depth_start_ns - visual_start_ns > tolerance_ns:
            issues.append(
                QualityIssue(
                    issue_type="depth_late_start",
                    stream_id=stream_id,
                    start_ns=visual_start_ns,
                    end_ns=depth_start_ns,
                    severity="warning",
                    decision="trim",
                    details={
                        **common_details,
                        "missing_head_ns": depth_start_ns - visual_start_ns,
                    },
                )
            )

        if visual_end_ns - depth_end_ns > tolerance_ns:
            issues.append(
                QualityIssue(
                    issue_type="depth_early_end",
                    stream_id=stream_id,
                    start_ns=depth_end_ns,
                    end_ns=visual_end_ns,
                    severity="warning",
                    decision="trim",
                    details={
                        **common_details,
                        "missing_tail_ns": visual_end_ns - depth_end_ns,
                    },
                )
            )

    return issues


__all__ = ["detect_depth_coverage"]
