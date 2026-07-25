"""
A2D Segment 候选生成器。

根据 A2D profile 的 required_streams，计算公共有效范围，
运行全部机器人质量检测器、生成候选 Segment。

用法:
    from zpds_prepare.segmentation.a2d_segmenter import generate_a2d_candidates

    candidates = generate_a2d_candidates(session, config=cfg)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import (
    CandidateSegment,
    plan_segments,
    get_issue_summary,
)
from zpds_prepare.detectors.robot import (
    detect_timeseries_structure,
    detect_joint_quality,
    detect_action_quality,
    detect_alignment_quality,
    detect_timeseries_gaps,
    detect_camera_gaps,
)
from zpds_prepare.readers.session_model import Session, VideoStream, TimeSeriesStream
from zpds_prepare.writers.candidate_writer import write_segment_candidates

logger = logging.getLogger(__name__)

# Profile 定义
A2D_REQUIRED_STREAMS = [
    "head_rgb", "hand_left_rgb", "hand_right_rgb",
    "robot_state", "robot_action",
]
A2D_OPTIONAL_STREAMS = ["gripper_state", "gripper_action"]

# 默认 Segment 参数
DEFAULT_MIN_DURATION_S = 1.0
DEFAULT_MAX_DURATION_S = 300.0  # A2D 任务可能长达几分钟


# ======================================================================
# 公共有效范围
# ======================================================================

def compute_public_valid_range(
    session: Session,
    required_streams: list[str] | None = None,
) -> tuple[int, int]:
    """计算必需流的公共有效时间范围。

    session_start_ns = max(stream_start_ns for each required stream)
    session_end_ns   = min(stream_end_ns   for each required stream)

    Args:
        session: Session 对象。
        required_streams: 必需流 ID 列表。默认使用 A2D_REQUIRED_STREAMS。

    Returns:
        (start_ns, end_ns) 公共有效范围。
    """
    if required_streams is None:
        required_streams = A2D_REQUIRED_STREAMS

    starts: list[int] = []
    ends: list[int] = []
    missing: list[str] = []

    for stream_id in required_streams:
        ts_list = _get_stream_timestamps(session, stream_id)
        if ts_list is None or len(ts_list) == 0:
            missing.append(stream_id)
            continue
        starts.append(ts_list[0])
        ends.append(ts_list[-1])

    if missing:
        raise ValueError(
            f"缺少必需流: {missing}。"
            f"可用流: video={list(session.video_streams.keys())}, "
            f"ts={list(session.time_series_streams.keys())}"
        )

    start_ns = max(starts)
    end_ns = min(ends)

    if start_ns >= end_ns:
        raise ValueError(
            f"公共有效范围无效: start={start_ns} >= end={end_ns}。"
            f"各流范围: {dict(zip(required_streams, zip(starts, ends)))}"
        )

    return start_ns, end_ns


def _get_stream_timestamps(session: Session, stream_id: str) -> list[int] | None:
    """从 Session 中获取指定流的 timestamp 列表。

    支持 VideoStream 和 TimeSeriesStream。
    """
    # 先查 video_streams
    vs = session.video_streams.get(stream_id)
    if vs is not None:
        return vs.timestamps_ns

    # 再查 time_series_streams
    ts = session.time_series_streams.get(stream_id)
    if ts is not None:
        return ts.timestamps_ns

    return None


# ======================================================================
# 主入口
# ======================================================================

def generate_a2d_candidates(
    session: Session,
    config: dict | None = None,
    alignment_dir: Path | None = None,
) -> tuple[list[CandidateSegment], list[QualityIssue], dict]:
    """为 A2D Session 生成候选 Segment。

    流程:
        1. 计算公共有效范围（required streams 的交集）
        2. 运行全部机器人检测器
        3. 运行流缺口检测（TS 缺口 + 相机缺失）
        4. 运行对齐质量检测（如 alignment parquet 可用）
        5. 调用 plan_segments()

    Args:
        session: 已读取的 Session 对象。
        config: 配置字典，支持:
            - min_duration_s: 最短 Segment 秒数 (默认 1.0)
            - max_duration_s: 最长 Segment 秒数 (默认 300.0)
            - robot: 机器人检测器子配置 (gap / freeze / alignment 等)
        alignment_dir: camera_robot_alignment.parquet 所在目录。
                       默认 output/a2d/{episode_id}/prepared/seg_000001/maps/

    Returns:
        (candidates, all_issues, summary)
    """
    if config is None:
        config = {}

    robot_cfg = config.get("robot", {})
    min_duration_s = float(config.get("min_duration_s", DEFAULT_MIN_DURATION_S))
    max_duration_s = float(config.get("max_duration_s", DEFAULT_MAX_DURATION_S))

    all_issues: list[QualityIssue] = []

    # ---- 1. 公共有效范围 ----
    session_start_ns, session_end_ns = compute_public_valid_range(session)
    session_id = session.session_id

    logger.info(
        "公共有效范围: %.3fs → %.3fs (%.3fs)",
        session_start_ns / 1e9,
        session_end_ns / 1e9,
        (session_end_ns - session_start_ns) / 1e9,
    )

    # ---- 2. 机器人检测器 ----
    # 2a. 时序结构 (robot_state, robot_action, gripper_state, gripper_action)
    for stream_id, ts_stream in session.time_series_streams.items():
        issues = detect_timeseries_structure(ts_stream, config=robot_cfg)
        all_issues.extend(issues)

    # 2b. 关节质量 (robot_state, gripper_state)
    for stream_id, ts_stream in session.time_series_streams.items():
        if ts_stream.role == "state":
            issues = detect_joint_quality(ts_stream, config=robot_cfg.get("joint", {}))
            all_issues.extend(issues)

    # 2c. 动作质量 (robot_action, gripper_action)
    for stream_id, ts_stream in session.time_series_streams.items():
        if ts_stream.role == "action":
            state_stream = _find_state_stream(session, ts_stream)
            issues = detect_action_quality(
                ts_stream,
                state_stream=state_stream,
                config=robot_cfg.get("action", {}),
            )
            all_issues.extend(issues)

    # ---- 3. 流缺口检测 ----
    gap_cfg = robot_cfg.get("gap", {})

    # 3a. TimeSeries 缺口
    for stream_id, ts_stream in session.time_series_streams.items():
        issues = detect_timeseries_gaps(ts_stream, config=gap_cfg)
        all_issues.extend(issues)

    # 3b. 相机缺失
    for stream_id, vs in session.video_streams.items():
        issues = detect_camera_gaps(vs, config=gap_cfg)
        all_issues.extend(issues)

    # ---- 4. 对齐质量 (如 alignment parquet 可用) ----
    alignment_df = _load_alignment(session, alignment_dir)
    if alignment_df is not None and not alignment_df.empty:
        issues = detect_alignment_quality(
            alignment_df,
            config=robot_cfg.get("alignment", {}),
        )
        all_issues.extend(issues)

    # ---- 5. 汇总 ----
    summary = get_issue_summary(all_issues)
    logger.info("总异常: %d, 按处置: %s", summary["total"], summary["by_decision"])

    # ---- 6. 规划候选 Segment ----
    candidates = plan_segments(
        issues=all_issues,
        session_start_ns=session_start_ns,
        session_end_ns=session_end_ns,
        min_duration_ns=int(min_duration_s * 1_000_000_000),
        max_duration_ns=int(max_duration_s * 1_000_000_000),
    )

    return candidates, all_issues, summary


def _find_state_stream(
    session: Session, action_stream: TimeSeriesStream,
) -> TimeSeriesStream | None:
    """找到与 action 流对应的 state 流。

    匹配规则: robot_action → robot_state, gripper_action → gripper_state
    """
    ts = session.time_series_streams
    if "action" in action_stream.stream_id:
        state_id = action_stream.stream_id.replace("action", "state")
        return ts.get(state_id)
    return None


def _load_alignment(
    session: Session, alignment_dir: Path | None,
) -> pd.DataFrame | None:
    """尝试加载 camera_robot_alignment.parquet。

    默认路径: output/a2d/{episode_id}/prepared/seg_000001/maps/
    """
    if alignment_dir is not None:
        path = Path(alignment_dir) / "camera_robot_alignment.parquet"
    else:
        # 从 session_id 推断: a2d_8032 → output/a2d/8032
        ep_id = session.session_id.replace("a2d_", "")
        path = Path(f"output/a2d/{ep_id}/prepared/seg_000001/maps/camera_robot_alignment.parquet")

    if not path.is_file():
        logger.info("对齐文件不可用，跳过对齐质量检测: %s", path)
        return None

    return pd.read_parquet(path)


# ======================================================================
# 便捷写出
# ======================================================================

def run_and_write(
    session: Session,
    output_dir: Path,
    config: dict | None = None,
    alignment_dir: Path | None = None,
) -> Path:
    """完整流程：检测 → 规划 → 写出 segment_candidates.json。

    Returns:
        输出文件路径。
    """
    candidates, all_issues, summary = generate_a2d_candidates(
        session, config, alignment_dir,
    )

    # 计算源范围
    session_start_ns, session_end_ns = compute_public_valid_range(session)

    # 写出 quality_issues.json
    from zpds_prepare.writers.quality_writer import write_quality_issues
    qi_path = write_quality_issues(
        output_path=output_dir / "quality_issues.json",
        issues=all_issues,
        source_session_id=session.session_id,
    )
    logger.info("quality_issues → %s", qi_path)

    # 写出 segment_candidates.json
    sc_path = write_segment_candidates(
        output_path=output_dir / "segment_candidates.json",
        candidates=candidates,
        source_session_id=session.session_id,
        source_start_ns=session_start_ns,
        source_end_ns=session_end_ns,
    )
    logger.info("segment_candidates → %s", sc_path)

    # 打印摘要
    _print_summary(candidates, summary, session_start_ns, session_end_ns)

    return sc_path


def _print_summary(
    candidates: list[CandidateSegment],
    summary: dict,
    session_start_ns: int,
    session_end_ns: int,
) -> None:
    """打印候选 Segment 摘要。"""
    print(f"\n{'=' * 60}")
    print(f"  公共有效范围: {session_start_ns:,} → {session_end_ns:,}")
    print(f"  时长: {(session_end_ns - session_start_ns) / 1e9:.2f}s")
    print(f"  异常总数: {summary['total']}")
    if summary["total"] > 0:
        print(f"  按类型: {summary['by_type']}")
        print(f"  按处置: {summary['by_decision']}")
    print(f"  候选 Segment: {len(candidates)}")
    for c in candidates:
        print(f"    {c.candidate_id}: "
              f"{c.source_start_ns:,} → {c.source_end_ns:,} "
              f"({c.duration_ns / 1e9:.2f}s, {c.reason})")
        for iss in c.issues_in_span:
            print(f"      ⚠ [{iss['decision']}] {iss['issue_type']}: "
                  f"{(iss['end_ns'] - iss['start_ns']) / 1e9:.2f}s")
    print(f"{'=' * 60}")


__all__ = [
    "compute_public_valid_range",
    "generate_a2d_candidates",
    "run_and_write",
    "A2D_REQUIRED_STREAMS",
    "A2D_OPTIONAL_STREAMS",
]
