"""
segment_candidates.json 写入器。
"""

import json
from pathlib import Path

from zpds_prepare.decisions.segment_planner import CandidateSegment


def write_segment_candidates(
    output_path: Path,
    candidates: list[CandidateSegment],
    source_session_id: str,
    source_start_ns: int,
    source_end_ns: int,
) -> Path:
    """将候选 Segment 列表写入 JSON 文件。

    Args:
        output_path: 输出文件路径 (如 output/segment_candidates.json)
        candidates: 候选 Segment 列表
        source_session_id: 来源 Session ID
        source_start_ns: 原始 Session 起始时间
        source_end_ns: 原始 Session 结束时间

    Returns:
        实际写入的文件路径
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "schema_version": "0.1.0",
        "source_session_id": source_session_id,
        "source_start_ns": source_start_ns,
        "source_end_ns": source_end_ns,
        "source_duration_s": round(
            (source_end_ns - source_start_ns) / 1_000_000_000, 3
        ),
        "candidate_count": len(candidates),
        "total_effective_duration_s": round(
            sum(c.duration_ns for c in candidates) / 1_000_000_000, 3
        ),
        "segments": [c.to_dict() for c in candidates],
    }

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return output_path
