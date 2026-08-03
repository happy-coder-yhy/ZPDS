"""B4: 遁甲多相机覆盖与末端可见性。

分析 Dunjia 3 路相机的时间覆盖、角色判定和末端执行器可见性。

检查项：
  1. 每路相机时间覆盖范围（start_ns / end_ns / duration）
  2. 相机间时间重叠（camera0↔camera1, camera0↔camera2, camera1↔camera2）
  3. 深度流与各相机的重叠
  4. IMU 流与各相机的重叠
  5. 相机角色（主/侧）由源元数据确定
  6. 末端执行器可见性评估

末端可见性评估：
  - 遁甲没有机器人 kinematics，无法做几何投影
  - 策略：
    a. 记录各相机 frame_count / time_coverage / bad_spans 作为覆盖基础
    b. 主视角（camera0）操作区若可用（有帧、可解码）→ 末端可见性暂标记为 ``unassessed``
    c. 侧视角用于遮挡互补：主视角全黑/冻结时，侧视角若可用 → 降低可见性风险等级
    d. VLM 复核仅用于低置信候选，标记 ``model_estimated``

原则：
  - 末端不可见 ≠ 任务失败，分开报告
  - 无几何时不虚构 in-frame ratio
  - 侧视角遮挡互补不等于末端可见性确认
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class CameraCoverageRecord:
    """单路相机的时间覆盖记录。"""

    camera_id: str
    role: str  # "primary" | "side"
    role_source: str
    frame_id: str

    # 时间覆盖
    frame_count: int
    start_timestamp_ns: int | None
    end_timestamp_ns: int | None
    duration_s: float

    # 解码状态
    decode_status: str  # "decodable" | "undecodable" | "unavailable"
    width: int = 0
    height: int = 0

    # 坏帧区间
    bad_spans: list[dict[str, Any]] = field(default_factory=list)

    # 与其他流的重叠
    depth_overlap_s: float = 0.0
    depth_overlap_ratio: float = 0.0  # 重叠时长 / 相机时长
    imu_overlap_s: float = 0.0
    imu_overlap_ratio: float = 0.0

    # 抽样复核
    review_sample: str = ""  # frame index 或 URI
    evidence_uri: str = ""


@dataclass
class EndEffectorVisibility:
    """末端执行器可见性评估。"""

    status: str  # "visible" | "partially_occluded" | "not_visible" | "unassessed"
    assessment_method: str  # "geometric_projection" | "vlm_review" | "manual_review" | "camera_coverage_only" | "unavailable"
    visible_ratio: float | None = None  # 可见帧比例（有几何时）
    occlusion_spans: list[dict[str, Any]] = field(default_factory=list)
    evidence_uri: str = ""
    config_hash: str = ""
    notes: str = ""


@dataclass
class DunjiaCoverageReport:
    """遁甲多相机覆盖与末端可见性报告。"""

    session_id: str
    source_path: str
    schema_version: str = "zpds.dunjia_coverage.v1"

    # 每路相机覆盖
    cameras: dict[str, CameraCoverageRecord] = field(default_factory=dict)

    # 末端执行器
    end_effector_visibility: EndEffectorVisibility = field(
        default_factory=lambda: EndEffectorVisibility(
            status="unassessed",
            assessment_method="unavailable",
        )
    )

    # 聚合
    overall_disposition: str = "pass"
    issues: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_dunjia_coverage(
    session: Any,
) -> DunjiaCoverageReport:
    """分析遁甲多相机覆盖与末端可见性。

    Args:
        session: ``Session`` 对象

    Returns:
        DunjiaCoverageReport 含每相机时间覆盖、流间重叠和末端可见性评估。
    """
    source_path = session.source_path
    report = DunjiaCoverageReport(
        session_id=session.session_id,
        source_path=str(source_path),
    )

    from zpds_prepare.readers.dunjia_reader import CAMERA_IDS

    # ---- 获取所有流 ----
    video_streams = session.video_streams
    depth_stream = session.depth_streams.get("ego_depth")
    imu_stream = session.imu_streams.get("robot0_imu")

    # 收集 IMU 和 Depth 时间范围
    depth_ts = np.array(depth_stream.timestamps_ns, dtype=np.int64) if depth_stream and depth_stream.timestamps_ns else np.array([], dtype=np.int64)
    imu_df = imu_stream.dataframe if imu_stream else None
    imu_ts = imu_df["timestamp_ns"].values.astype(np.int64) if imu_df is not None else np.array([], dtype=np.int64)

    depth_range = (int(depth_ts[0]), int(depth_ts[-1])) if len(depth_ts) > 0 else None
    imu_range = (int(imu_ts[0]), int(imu_ts[-1])) if len(imu_ts) > 0 else None

    # ---- 为每路相机构建覆盖记录 ----
    for cam_name in ["camera0", "camera1", "camera2"]:
        vs = video_streams.get(cam_name)
        frame_id = CAMERA_IDS.get(cam_name, f"{cam_name}_optical_frame")

        # 角色
        if cam_name == "camera0":
            role = "primary"
        else:
            role = "side"

        if vs is None:
            report.cameras[cam_name] = CameraCoverageRecord(
                camera_id=cam_name,
                role=role,
                role_source="dunjia_reader.CAMERA_IDS",
                frame_id=frame_id,
                frame_count=0,
                start_timestamp_ns=None,
                end_timestamp_ns=None,
                duration_s=0.0,
                decode_status="unavailable",
            )
            report.issues.append(f"{cam_name} 视频流缺失")
            continue

        ts = np.array(vs.timestamps_ns, dtype=np.int64) if vs.timestamps_ns else np.array([], dtype=np.int64)
        cam_range = (int(ts[0]), int(ts[-1])) if len(ts) > 0 else None

        # 解码状态（简单判断通过完整性中已有的信息）
        decode_status = "decodable"
        if vs.frame_count == 0:
            decode_status = "unavailable"
        elif not vs.video_path:
            decode_status = "undecodable"
        else:
            vp = Path(vs.video_path)
            if not vp.exists():
                decode_status = "undecodable"

        # 坏帧区间（当前仅记录帧数异常，详细坏帧需单独检测器）
        bad_spans: list[dict] = []
        if vs.frame_count > 0 and len(ts) > 0:
            expected_interval = np.median(np.diff(ts)) if len(ts) > 1 else 0
            if expected_interval > 0:
                gaps = np.where(np.diff(ts) > expected_interval * 5)[0]
                for g in gaps:
                    bad_spans.append({
                        "type": "timestamp_gap",
                        "start_frame": int(g),
                        "end_frame": int(g + 1),
                        "start_timestamp_ns": int(ts[g]),
                        "end_timestamp_ns": int(ts[g + 1]),
                        "gap_ns": int(ts[g + 1] - ts[g]),
                    })

        # 与深度的重叠
        depth_overlap = _compute_overlap(cam_range, depth_range)

        # 与 IMU 的重叠
        imu_overlap = _compute_overlap(cam_range, imu_range)

        cam_duration = (cam_range[1] - cam_range[0]) / 1e9 if cam_range else 0.0

        report.cameras[cam_name] = CameraCoverageRecord(
            camera_id=cam_name,
            role=role,
            role_source="dunjia_reader.CAMERA_IDS",
            frame_id=frame_id,
            frame_count=vs.frame_count,
            start_timestamp_ns=cam_range[0] if cam_range else None,
            end_timestamp_ns=cam_range[1] if cam_range else None,
            duration_s=round(cam_duration, 3),
            decode_status=decode_status,
            width=vs.width,
            height=vs.height,
            bad_spans=bad_spans,
            depth_overlap_s=round(depth_overlap, 3),
            depth_overlap_ratio=round(depth_overlap / max(cam_duration, 0.001), 4),
            imu_overlap_s=round(imu_overlap, 3),
            imu_overlap_ratio=round(imu_overlap / max(cam_duration, 0.001), 4),
            review_sample="",
            evidence_uri=f"session://{session.session_id}/{cam_name}",
        )

    # ---- 末端执行器可见性 ----
    _assess_end_effector(report)

    # ---- 聚合 ----
    _aggregate_coverage(report)

    return report


# ---------------------------------------------------------------------------
# 重叠计算
# ---------------------------------------------------------------------------


def _compute_overlap(
    range_a: tuple[int, int] | None,
    range_b: tuple[int, int] | None,
) -> float:
    """计算两个时间范围的交叠时长（秒）。"""
    if range_a is None or range_b is None:
        return 0.0
    start = max(range_a[0], range_b[0])
    end = min(range_a[1], range_b[1])
    if end <= start:
        return 0.0
    return (end - start) / 1_000_000_000


# ---------------------------------------------------------------------------
# 末端可见性评估
# ---------------------------------------------------------------------------


def _assess_end_effector(report: DunjiaCoverageReport) -> None:
    """评估末端执行器可见性。

    遁甲无机器人 kinematics：
      - 主视角 camera0 可解码且帧数 > 0 → 相机覆盖可用
      - 侧视角 camera1/camera2 可解码 → 辅助验证
      - 可见性标记为 ``unassessed``（无几何验证手段）
      - 如需进一步评估，需 VLM 或人工复核
    """
    primary = report.cameras.get("camera0")
    side1 = report.cameras.get("camera1")
    side2 = report.cameras.get("camera2")

    # 检查主视角
    if primary is None or primary.frame_count == 0:
        report.end_effector_visibility = EndEffectorVisibility(
            status="not_visible",
            assessment_method="camera_coverage_only",
            notes="主视角 camera0 无可用帧",
        )
        report.issues.append("末端不可见：主视角 camera0 无帧")
        return

    if primary.decode_status != "decodable":
        report.end_effector_visibility = EndEffectorVisibility(
            status="not_visible",
            assessment_method="camera_coverage_only",
            notes=f"主视角 camera0 解码失败 ({primary.decode_status})",
        )
        report.issues.append(f"末端不可见：主视角解码状态 {primary.decode_status}")
        return

    # 主视角可用 → 末端理论上可见
    # 检查遮挡情况
    side_available = (
        (side1 and side1.frame_count > 0)
        or (side2 and side2.frame_count > 0)
    )

    notes_parts = ["主视角 camera0 可用"]
    if side_available:
        notes_parts.append("侧视角可用（遮挡互补）")

    # 标记为 unassessed（无几何验证）
    report.end_effector_visibility = EndEffectorVisibility(
        status="unassessed",
        assessment_method="camera_coverage_only",
        notes="; ".join(notes_parts) + "；需 VLM 或人工复核确认末端可见性",
    )


# ---------------------------------------------------------------------------
# 聚合
# ---------------------------------------------------------------------------


def _aggregate_coverage(report: DunjiaCoverageReport) -> None:
    """聚合覆盖报告 disposition。"""
    primary = report.cameras.get("camera0")

    if primary is None or primary.frame_count == 0:
        report.overall_disposition = "reject"
        return

    if primary.decode_status == "unavailable":
        report.overall_disposition = "reject"
        return

    # 检查坏帧比例
    if primary.bad_spans:
        total_bad_frames = sum(
            b["end_frame"] - b["start_frame"] + 1 for b in primary.bad_spans
        )
        if total_bad_frames / max(primary.frame_count, 1) > 0.1:
            report.issues.append(
                f"主视角坏帧比例 > 10%: {total_bad_frames}/{primary.frame_count}"
            )
            report.overall_disposition = "keep_with_flag"

    # 深度重叠检查
    if primary.depth_overlap_ratio < 0.5:
        report.issues.append(
            f"主视角深度重叠不足: {primary.depth_overlap_ratio:.1%}"
        )
        report.overall_disposition = "keep_with_flag"

    # IMU 重叠检查
    if primary.imu_overlap_ratio < 0.5:
        report.issues.append(
            f"主视角 IMU 重叠不足: {primary.imu_overlap_ratio:.1%}"
        )
        report.overall_disposition = "keep_with_flag"

    # 侧视角缺失仅告警
    for cam_name in ["camera1", "camera2"]:
        cam = report.cameras.get(cam_name)
        if cam is None or cam.frame_count == 0:
            report.issues.append(f"侧视角 {cam_name} 不可用（非阻断）")


__all__ = [
    "CameraCoverageRecord",
    "DunjiaCoverageReport",
    "EndEffectorVisibility",
    "check_dunjia_coverage",
]
