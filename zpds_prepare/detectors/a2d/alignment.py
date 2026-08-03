"""B7: A2D 相机—机器人对齐。

从 Session 中已有的相机帧和 HDF5 时间戳，构建可审计的逐帧映射表，
记录映射方法、误差、不确定度和连续组。

硬约束：
  - 不能把第 N 个 HDF5 行视为第 N 个相机帧（必须基于时间戳或显式索引映射）
  - 跨 gap/reset 不插值
  - 映射不可信时保留 RGB，但阻断 ``robot_bc_ready``

映射方法:
  - ``aligned_joints_index`` — 相机帧目录索引 → HDF5 同索引时间戳（A2D 当前方法）
  - ``nearest_neighbor`` — 最近邻时间戳匹配（未来扩展用）
  - ``inferred`` — 中间帧线性插值（仅在连续组内）
  - ``unavailable`` — 无可用映射
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class CameraRobotAlignmentRow:
    """单帧对齐记录。"""

    camera_stream_id: str
    source_frame_index: int
    camera_timestamp_ns: int
    robot_row: int
    robot_timestamp_ns: int
    mapping_method: str  # "aligned_joints_index" | "nearest_neighbor" | "inferred" | "unavailable"
    error_ns: int  # camera_ts - robot_ts (带符号)
    uncertainty_ns: int  # 绝对不确定度
    continuity_group: int  # 连续组 ID
    evidence_uri: str = ""


@dataclass
class StreamAlignmentSummary:
    """单个相机流的对齐摘要。"""

    stream_id: str
    total_camera_frames: int
    mapped_frames: int
    unmapped_frames: int
    continuity_groups: int  # 连续组数
    method_distribution: dict[str, int] = field(default_factory=dict)
    error_abs_p50_ns: float = float("nan")
    error_abs_p95_ns: float = float("nan")
    error_abs_max_ns: float = float("nan")
    rms_error_ns: float = float("nan")


@dataclass
class A2DAlignmentReport:
    """A2D 相机-机器人对齐报告。"""

    episode_id: str
    source_path: str
    schema_version: str = "zpds.a2d_alignment.v1"

    # 每相机流对齐
    streams: dict[str, StreamAlignmentSummary] = field(default_factory=dict)

    # HDF5 信息
    hdf5_sample_count: int = 0
    hdf5_timestamp_valid: bool = False

    # 对齐矩阵（可序列化为 Parquet）
    alignment_rows: list[CameraRobotAlignmentRow] = field(default_factory=list)

    # 聚合
    overall_disposition: str = "pass"
    robot_bc_ready: bool | None = None  # None = 未判定
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 常量和配置
# ---------------------------------------------------------------------------

# 连续组断裂条件：
#   相机帧索引间隔 > 1（缺失帧）
#   HDF5 时间戳间隔 > 标称间隔的 3 倍
_GAP_FACTOR_FRAME = 1  # 相机帧索引差 > 1 → gap
_GAP_FACTOR_TIMESTAMP = 3.0  # HDF5 时间戳间隔 > 3× 中位数 → gap
_DEFAULT_UNCERTAINTY_NS = 16_666_667  # 30fps 半帧 ≈ 16.7ms 默认不确定度


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_a2d_alignment(
    session: Any,
) -> A2DAlignmentReport:
    """构建 A2D 相机→机器人对齐表。

    从 Session 中的 video_streams 和 time_series_streams 提取每帧映射，
    检测间隙、记录映射方法和不确定度。

    Args:
        session: ``Session`` 对象（来自 ``a2d_reader.read_session()``）

    Returns:
        A2DAlignmentReport 含每流对齐摘要和逐帧映射行。
    """
    root = Path(session.source_path)
    report = A2DAlignmentReport(
        episode_id=session.session_id.replace("a2d_", ""),
        source_path=str(root),
    )

    # ---- 0. 前置检查 ----
    ts_streams = session.time_series_streams
    robot_state = ts_streams.get("robot_state")
    if robot_state is None:
        report.issues.append("robot_state 流不存在，无法建立对齐")
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return report

    robot_ts = np.array(robot_state.timestamps_ns, dtype=np.int64)
    if len(robot_ts) == 0:
        report.issues.append("robot_state 时间戳为空")
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return report

    report.hdf5_sample_count = len(robot_ts)
    report.hdf5_timestamp_valid = _check_monotonic(robot_ts)

    if not report.hdf5_timestamp_valid:
        report.issues.append("HDF5 时间戳非单调递增")

    # ---- 1. 为每路 RGB 相机构建对齐 ----
    rgb_streams = {
        sid: vs
        for sid, vs in session.video_streams.items()
        if sid.endswith("_rgb")
    }

    if not rgb_streams:
        report.issues.append("无 RGB 相机流")
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return report

    all_rows: list[CameraRobotAlignmentRow] = []

    for stream_id, video_stream in rgb_streams.items():
        stream_rows, summary = _align_stream(
            stream_id, video_stream, robot_ts, root,
        )
        all_rows.extend(stream_rows)
        report.streams[stream_id] = summary

    report.alignment_rows = all_rows

    # ---- 2. 聚合判定 ----
    _evaluate_disposition(report)

    return report


# ---------------------------------------------------------------------------
# 单流对齐
# ---------------------------------------------------------------------------


def _align_stream(
    stream_id: str,
    video_stream: Any,
    robot_ts: np.ndarray,
    episode_root: Path,
) -> tuple[list[CameraRobotAlignmentRow], StreamAlignmentSummary]:
    """为单个相机流建立对齐表。"""
    rows: list[CameraRobotAlignmentRow] = []
    index_frames = video_stream.index_frames or []

    # 相机帧索引列表
    camera_indices = sorted(
        f["frame_index"] for f in index_frames
    )
    if not camera_indices:
        return rows, StreamAlignmentSummary(
            stream_id=stream_id,
            total_camera_frames=0,
            mapped_frames=0,
            unmapped_frames=0,
            continuity_groups=0,
        )

    # HDF5 时间戳信息
    h5_count = len(robot_ts)
    median_dt = _median_interval(robot_ts)
    gap_threshold_ns = int(median_dt * _GAP_FACTOR_TIMESTAMP)

    group_id = 0
    prev_cam_idx: int | None = None
    prev_robot_row: int | None = None
    prev_robot_ts: int | None = None

    mapped = 0
    unmapped = 0
    methods: dict[str, int] = {}
    group_map_counts: list[int] = []
    errors_abs: list[float] = []

    for cam_idx in camera_indices:
        # 映射方法: aligned_joints_index
        #   相机帧目录编号 cam_idx 对应 HDF5 第 cam_idx 行
        if cam_idx < h5_count:
            robot_row = cam_idx
            robot_timestamp = int(robot_ts[cam_idx])
            mapping_method = "aligned_joints_index"
            error_ns = 0  # 直接索引映射，误差定义为 0
            uncertainty_ns = _DEFAULT_UNCERTAINTY_NS
            mapped += 1
        else:
            robot_row = -1
            robot_timestamp = 0
            mapping_method = "unavailable"
            error_ns = 0
            uncertainty_ns = int(1e9)  # 1 秒级不确定度
            unmapped += 1

        # 连续组检测: 相机帧索引跳跃
        if prev_cam_idx is not None:
            cam_gap = cam_idx - prev_cam_idx > _GAP_FACTOR_FRAME
            h5_gap = (
                prev_robot_ts is not None
                and robot_timestamp > 0
                and prev_robot_ts > 0
                and robot_timestamp - prev_robot_ts > gap_threshold_ns
            )
            if cam_gap or h5_gap:
                group_id += 1

        # 相机时间戳 = robot_timestamp（直接对齐时）
        cam_ts = robot_timestamp if mapping_method != "unavailable" else 0

        rows.append(
            CameraRobotAlignmentRow(
                camera_stream_id=stream_id,
                source_frame_index=cam_idx,
                camera_timestamp_ns=cam_ts,
                robot_row=robot_row,
                robot_timestamp_ns=robot_timestamp,
                mapping_method=mapping_method,
                error_ns=error_ns,
                uncertainty_ns=uncertainty_ns,
                continuity_group=group_id,
                evidence_uri=f"episode://{episode_root.name}/camera/{cam_idx}",
            )
        )

        methods[mapping_method] = methods.get(mapping_method, 0) + 1
        if robot_timestamp > 0 and robot_row >= 0:
            errors_abs.append(abs(error_ns))

        prev_cam_idx = cam_idx
        prev_robot_row = robot_row
        prev_robot_ts = robot_timestamp if robot_timestamp > 0 else prev_robot_ts

    # 摘要
    err_arr = np.array(errors_abs) if errors_abs else np.array([0.0])
    summary = StreamAlignmentSummary(
        stream_id=stream_id,
        total_camera_frames=len(camera_indices),
        mapped_frames=mapped,
        unmapped_frames=unmapped,
        continuity_groups=group_id + 1,
        method_distribution=methods,
        error_abs_p50_ns=float(np.percentile(err_arr, 50)),
        error_abs_p95_ns=float(np.percentile(err_arr, 95)),
        error_abs_max_ns=float(err_arr.max()),
        rms_error_ns=float(np.sqrt(np.mean(err_arr**2))),
    )

    return rows, summary


# ---------------------------------------------------------------------------
# 聚合判定
# ---------------------------------------------------------------------------


def _evaluate_disposition(report: A2DAlignmentReport) -> None:
    """根据对齐质量判定 disposition 和 robot_bc_ready。"""
    if not report.streams:
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        return

    total_mapped = sum(s.mapped_frames for s in report.streams.values())
    total_frames = sum(s.total_camera_frames for s in report.streams.values())
    mapping_ratio = total_mapped / max(total_frames, 1)

    if mapping_ratio == 0:
        report.overall_disposition = "reject"
        report.robot_bc_ready = False
        report.issues.append("零帧可映射")
    elif mapping_ratio < 0.8:
        report.overall_disposition = "keep_with_flag"
        report.robot_bc_ready = False
        report.issues.append(
            f"映射覆盖率不足: {mapping_ratio:.1%}"
        )
    elif report.issues:
        report.overall_disposition = "keep_with_flag"
        report.robot_bc_ready = True  # 可映射但有告警
    else:
        report.overall_disposition = "pass"
        report.robot_bc_ready = True

    # 多连续组 → 不确定度上升
    max_groups = max(
        (s.continuity_groups for s in report.streams.values()),
        default=0,
    )
    if max_groups > 1:
        report.issues.append(
            f"检测到 {max_groups} 个不连续组，跨 group 映射不可用"
        )
        report.robot_bc_ready = False


# ---------------------------------------------------------------------------
# Parquet 写出
# ---------------------------------------------------------------------------


def write_alignment_parquet(
    report: A2DAlignmentReport,
    output_path: str | Path,
) -> Path:
    """将对齐表写出为 camera_robot_alignment.parquet。

    Schema:
        camera_stream_id: str
        source_frame_index: int64
        camera_timestamp_ns: int64
        robot_row: int64
        robot_timestamp_ns: int64
        mapping_method: str
        error_ns: int64
        uncertainty_ns: int64
        continuity_group: int64
        evidence_uri: str
    """
    if not report.alignment_rows:
        raise ValueError("对齐表为空，无法写出")

    df = pd.DataFrame([
        {
            "camera_stream_id": r.camera_stream_id,
            "source_frame_index": r.source_frame_index,
            "camera_timestamp_ns": r.camera_timestamp_ns,
            "robot_row": r.robot_row,
            "robot_timestamp_ns": r.robot_timestamp_ns,
            "mapping_method": r.mapping_method,
            "error_ns": r.error_ns,
            "uncertainty_ns": r.uncertainty_ns,
            "continuity_group": r.continuity_group,
            "evidence_uri": r.evidence_uri,
        }
        for r in report.alignment_rows
    ])

    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

    return path


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _check_monotonic(arr: np.ndarray) -> bool:
    if len(arr) < 2:
        return True
    return bool(np.all(np.diff(arr) > 0))


def _median_interval(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return float("nan")
    return float(np.median(np.diff(arr)))


__all__ = [
    "A2DAlignmentReport",
    "CameraRobotAlignmentRow",
    "StreamAlignmentSummary",
    "check_a2d_alignment",
    "write_alignment_parquet",
]
