"""B1: 遁甲来源完整性清单与相机角色。

检查 Dunjia Session 中所有预期流的完整性，建立相机角色映射。

检查项：
  1. MCAP 容器可打开、可解析
  2. 3 路 RGB 相机流（camera0/1/2）每路：帧数 > 0、视频缓存可解码、标定存在
  3. 深度流：帧数 > 0、首帧 PNG 可解码、dtype/分辨率有效
  4. IMU 流：样本数 > 0、列完整
  5. 标定：每路相机有对应 CameraCalibration 消息
  6. 相机角色来自源元数据（CAMERA_IDS），不根据目录名猜测
  7. robot_bc_ready 显式标记为 not_applicable

处置规则：
  - 缺必需资产或容器不可解析 → reject 对应视图
  - 可选流缺失 → keep_with_flag
  - 全部通过 → pass
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# 遁甲常量（从 dunjia_reader 导入以避免硬编码重复）
# ---------------------------------------------------------------------------

from zpds_prepare.readers.dunjia_reader import (
    CALIB_TOPICS,
    CAMERA_IDS,
    CAMERA_TOPICS,
    TOPIC_DEPTH,
    TOPIC_IMU,
    _open_mcap,
)

# 必需流：缺一不可
_REQUIRED_STREAMS = frozenset({"camera0", "depth", "robot0_imu"})
# 可选流：缺失标记但不阻断
_OPTIONAL_STREAMS = frozenset({"camera1", "camera2"})


# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class CameraRole:
    """单路相机的角色声明。

    角色必须来自源元数据或配置，不能根据目录名或单帧画面推测。
    """

    camera_id: str
    role: str  # "primary" | "side" | "depth"
    role_source: str  # 角色来源，如 "dunjia_reader.CAMERA_IDS"
    frame_id: str  # 光学 frame，如 "headcam_center_optical_frame"
    evidence_uri: str | None = None


@dataclass
class StreamCompleteness:
    """单个流的完整性检查结果。"""

    stream_id: str
    stream_type: str  # "video" | "depth" | "imu" | "calibration"
    required: bool
    present: bool
    decodable: bool | None = None  # None = 不适用（如标定流）
    frame_count: int = 0
    sample_count: int = 0
    width: int = 0
    height: int = 0
    dtype: str = ""
    issues: list[str] = field(default_factory=list)
    disposition: str = "pass"  # "pass" | "keep_with_flag" | "reject"


@dataclass
class DunjiaCompletenessReport:
    """遁甲 Session 完整性报告。

    一次性输出所有流的完整性和相机角色，作为 B2–B5 和 QC Stage 0/1 的基础证据。
    """

    session_id: str
    source_path: str
    source_sha256: str
    schema_version: str = "zpds.dunjia_completeness.v1"

    # 每流检查结果
    streams: dict[str, StreamCompleteness] = field(default_factory=dict)

    # 相机角色
    camera_roles: dict[str, CameraRole] = field(default_factory=dict)

    # 质量视图适用性声明
    robot_bc_ready: str = "not_applicable"
    robot_bc_ready_reason: str = "遁甲无机器人 state/action 流"

    # 聚合
    overall_disposition: str = "pass"  # "pass" | "keep_with_flag" | "reject"
    required_present: int = 0
    required_total: int = 0
    optional_present: int = 0
    optional_total: int = 0

    @property
    def all_required_present(self) -> bool:
        return self.required_present == self.required_total


# ---------------------------------------------------------------------------
# 检测逻辑
# ---------------------------------------------------------------------------


def _validate_mcap(mcap_path: str) -> tuple[bool, str]:
    """验证 MCAP 文件可打开且至少有一个可解码消息。"""
    if not Path(mcap_path).is_file():
        return False, f"MCAP 文件不存在: {mcap_path}"
    try:
        reader, fh = _open_mcap(mcap_path)
    except Exception as exc:
        return False, f"MCAP 无法打开: {exc}"
    try:
        message_count = 0
        for _ in reader.iter_decoded_messages():
            message_count += 1
            if message_count >= 1:
                break
        if message_count == 0:
            return False, "MCAP 包含零条可解码消息"
        return True, ""
    except Exception as exc:
        return False, f"MCAP 消息解码失败: {exc}"
    finally:
        fh.close()


def _count_topic_messages(mcap_path: str, topic: str) -> int:
    """快速统计指定 topic 的消息数。"""
    reader, fh = _open_mcap(mcap_path)
    try:
        count = 0
        for _schema, channel, _msg, _decoded in reader.iter_decoded_messages():
            if channel.topic == topic:
                count += 1
        return count
    finally:
        fh.close()


def _validate_depth_first_frame(mcap_path: str) -> tuple[bool, int, int, str, str]:
    """解码首张深度 PNG，返回 (ok, width, height, dtype, error)。"""
    import cv2
    import numpy as np

    reader, fh = _open_mcap(mcap_path)
    try:
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic == TOPIC_DEPTH:
                nparr = np.frombuffer(decoded.data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if img is None:
                    return False, 0, 0, "", "首张深度 PNG 解码为 None"
                if img.ndim != 2:
                    return False, 0, 0, "", f"深度图维度异常: {img.ndim}D（期望 2D）"
                h, w = img.shape
                return True, w, h, str(img.dtype), ""
        return False, 0, 0, "", "深度 topic 无消息"
    finally:
        fh.close()


def _check_imu_columns(imu_df: Any) -> list[str]:
    """检查 IMU DataFrame 是否包含必需的 7 列。"""
    required_cols = {"timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"}
    actual = set(imu_df.columns)
    return sorted(required_cols - actual)


def _sha256_file(path: str) -> str:
    """计算文件 SHA-256 哈希。文件不存在或不可读时返回空字符串。"""
    digest = hashlib.sha256()
    file_path = Path(path)
    if not file_path.is_file():
        return ""
    try:
        with file_path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return ""


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_dunjia_completeness(
    session: Any,
    *,
    require_depth: bool = True,
) -> DunjiaCompletenessReport:
    """检查遁甲 Session 的完整性。

    基于已读取的 Session 对象（来自 ``dunjia_reader.read_session()``），
    逐一核验每个预期流的存在性、可解码性和基本有效性。

    Args:
        session: ``zpds_prepare.readers.session_model.Session`` 对象
        require_depth: 深度流是否为必需（默认 True，对应 config 中 ``dunjia.depth.required: true``）

    Returns:
        DunjiaCompletenessReport 包含每流状态、相机角色、质量视图声明和聚合判断。
    """
    source_path = session.source_path

    report = DunjiaCompletenessReport(
        session_id=session.session_id,
        source_path=str(source_path),
        source_sha256=_sha256_file(str(source_path)),
    )

    # ---- 0. MCAP 容器级 ----
    mcap_ok, mcap_error = _validate_mcap(str(source_path))
    if not mcap_ok:
        for stream_id in _REQUIRED_STREAMS | _OPTIONAL_STREAMS | {"calibration"}:
            report.streams[stream_id] = StreamCompleteness(
                stream_id=stream_id,
                stream_type=_stream_type_for_id(stream_id),
                required=stream_id in _REQUIRED_STREAMS,
                present=False,
                issues=[f"MCAP 容器不可用: {mcap_error}"],
                disposition="reject",
            )
        report.overall_disposition = "reject"
        report.camera_roles = _build_camera_roles()
        return report

    # ---- 1. 视频流（camera0/1/2） ----
    for cam_name in ["camera0", "camera1", "camera2"]:
        video_stream = session.video_streams.get(cam_name)
        required = cam_name in _REQUIRED_STREAMS
        stream = _check_video_stream(
            cam_name, video_stream, required, str(source_path),
        )
        report.streams[cam_name] = stream

    # ---- 2. 深度流 ----
    depth_stream = session.depth_streams.get("ego_depth")
    depth_required = require_depth
    report.streams["depth"] = _check_depth_stream(
        depth_stream, depth_required, str(source_path),
    )

    # ---- 3. IMU 流 ----
    imu_stream = session.imu_streams.get("robot0_imu")
    report.streams["robot0_imu"] = _check_imu_stream(
        imu_stream, str(source_path),
    )

    # ---- 4. 标定 ----
    calib_stream = _check_calibration(source_path, mcap_ok=mcap_ok)
    report.streams["calibration"] = calib_stream

    # ---- 5. 聚合 ----
    required_streams = {
        sid: s for sid, s in report.streams.items() if s.required
    }
    optional_streams = {
        sid: s for sid, s in report.streams.items() if not s.required
    }
    report.required_total = len(required_streams)
    report.required_present = sum(
        1 for s in required_streams.values() if s.present
    )
    report.optional_total = len(optional_streams)
    report.optional_present = sum(
        1 for s in optional_streams.values() if s.present
    )

    dispositions = [s.disposition for s in report.streams.values()]
    if "reject" in dispositions:
        report.overall_disposition = "reject"
    elif "keep_with_flag" in dispositions:
        report.overall_disposition = "keep_with_flag"
    else:
        report.overall_disposition = "pass"

    # ---- 6. 相机角色 ----
    report.camera_roles = _build_camera_roles()

    return report


# ---------------------------------------------------------------------------
# 内部辅助
# ---------------------------------------------------------------------------


def _stream_type_for_id(stream_id: str) -> str:
    if stream_id.startswith("camera"):
        return "video"
    if stream_id == "depth":
        return "depth"
    if stream_id.endswith("imu"):
        return "imu"
    if stream_id == "calibration":
        return "calibration"
    return "unknown"


def _build_camera_roles() -> dict[str, CameraRole]:
    """从源元数据 CAMERA_IDS 构建相机角色映射。

    角色由 dunjia_reader 中的 CAMERA_IDS 字典定义，
    不允许从目录名或画面内容推测。
    """
    roles: dict[str, CameraRole] = {}
    for cam_name, frame_id in CAMERA_IDS.items():
        if cam_name == "depth":
            role = "depth"
        elif cam_name == "camera0":
            role = "primary"
        else:
            role = "side"
        roles[cam_name] = CameraRole(
            camera_id=cam_name,
            role=role,
            role_source="dunjia_reader.CAMERA_IDS",
            frame_id=frame_id,
        )
    return roles


def _check_video_stream(
    cam_name: str,
    video_stream: Any | None,
    required: bool,
    source_path: str,
) -> StreamCompleteness:
    """检查单个视频流。"""
    stream = StreamCompleteness(
        stream_id=cam_name,
        stream_type="video",
        required=required,
        present=video_stream is not None,
    )

    if video_stream is None:
        msg = f"video_streams 中缺少 {cam_name}"
        stream.issues.append(msg)
        stream.disposition = "reject" if required else "keep_with_flag"
        return stream

    stream.frame_count = video_stream.frame_count
    stream.width = video_stream.width
    stream.height = video_stream.height

    if video_stream.frame_count <= 0:
        stream.issues.append(f"{cam_name} 帧数为 0")
        stream.disposition = "reject" if required else "keep_with_flag"
        return stream

    # 检查视频缓存文件是否可解码
    video_path = video_stream.video_path
    if not video_path or not Path(video_path).is_file():
        stream.issues.append(f"{cam_name} 视频缓存缺失或不可用: {video_path}")
        stream.decodable = False
        stream.disposition = "reject" if required else "keep_with_flag"
        return stream

    # 快速验证视频可被 OpenCV 打开
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            stream.issues.append(f"{cam_name} 视频无法解码: {video_path}")
            stream.decodable = False
            stream.disposition = "reject" if required else "keep_with_flag"
        else:
            stream.decodable = True
            if stream.disposition == "pass":
                stream.disposition = "pass"
    finally:
        cap.release()

    # 标定状态
    topic = CAMERA_TOPICS.get(cam_name, "")
    calib_count = (
        _count_topic_messages(source_path, CALIB_TOPICS.get(cam_name, ""))
        if source_path and CALIB_TOPICS.get(cam_name)
        else 0
    )
    if calib_count == 0:
        stream.issues.append(f"{cam_name} 无标定消息")
        # 标定不影响 RGB 可用性，仅记录

    return stream


def _check_depth_stream(
    depth_stream: Any | None,
    required: bool,
    source_path: str,
) -> StreamCompleteness:
    """检查深度流。"""
    stream = StreamCompleteness(
        stream_id="depth",
        stream_type="depth",
        required=required,
        present=depth_stream is not None,
    )

    if depth_stream is None:
        msg = "depth_streams 中缺少 ego_depth"
        stream.issues.append(msg)
        stream.disposition = "reject" if required else "keep_with_flag"
        return stream

    stream.frame_count = depth_stream.frame_count
    stream.width = depth_stream.width
    stream.height = depth_stream.height
    stream.dtype = depth_stream.dtype

    if depth_stream.frame_count <= 0:
        stream.issues.append("深度流帧数为 0")
        stream.disposition = "reject" if required else "keep_with_flag"
        return stream

    # 验证首帧可解码
    if Path(source_path).is_file():
        ok, w, h, dtype, error = _validate_depth_first_frame(source_path)
        if not ok:
            stream.issues.append(f"深度首帧解码失败: {error}")
            stream.decodable = False
            stream.disposition = "reject" if required else "keep_with_flag"
        else:
            stream.decodable = True
            # 交叉验证 Session 中记录的分辨率与实测值
            if w > 0 and stream.width > 0 and w != stream.width:
                stream.issues.append(
                    f"深度宽度不一致: 实测={w}, Session记录={stream.width}"
                )
            if h > 0 and stream.height > 0 and h != stream.height:
                stream.issues.append(
                    f"深度高度不一致: 实测={h}, Session记录={stream.height}"
                )
            if dtype and depth_stream.dtype != dtype:
                stream.issues.append(
                    f"深度 dtype 不一致: 实测={dtype}, Session记录={depth_stream.dtype}"
                )

    # 检查 unit 状态
    if getattr(depth_stream, "unit", "unknown") == "unknown":
        stream.issues.append("深度单位未知（unit=unknown）")

    return stream


def _check_imu_stream(
    imu_stream: Any | None,
    source_path: str,
) -> StreamCompleteness:
    """检查 IMU 流。"""
    stream = StreamCompleteness(
        stream_id="robot0_imu",
        stream_type="imu",
        required=True,  # IMU 对遁甲是必需的
        present=imu_stream is not None,
    )

    if imu_stream is None:
        stream.issues.append("imu_streams 中缺少 robot0_imu")
        stream.disposition = "reject"
        return stream

    stream.sample_count = len(imu_stream.dataframe)

    if stream.sample_count <= 0:
        stream.issues.append("IMU 样本数为 0")
        stream.disposition = "reject"
        return stream

    # 列完整性
    missing_cols = _check_imu_columns(imu_stream.dataframe)
    if missing_cols:
        stream.issues.append(f"IMU 缺少列: {missing_cols}")
        stream.disposition = "keep_with_flag"

    return stream


def _check_calibration(source_path: str, *, mcap_ok: bool = True) -> StreamCompleteness:
    """检查标定消息覆盖。"""
    stream = StreamCompleteness(
        stream_id="calibration",
        stream_type="calibration",
        required=False,  # 标定缺失败不阻断 RGB
        present=False,
        decodable=None,
    )

    if not mcap_ok or not source_path:
        stream.issues.append("MCAP 不可用，跳过标定检查")
        stream.disposition = "keep_with_flag"
        return stream

    calib_present: list[str] = []
    calib_missing: list[str] = []

    for cam_name in ["camera0", "camera1", "camera2"]:
        calib_topic = CALIB_TOPICS.get(cam_name, "")
        if not calib_topic:
            continue
        count = _count_topic_messages(source_path, calib_topic)
        if count > 0:
            calib_present.append(cam_name)
        else:
            calib_missing.append(cam_name)

    stream.present = len(calib_present) > 0
    stream.frame_count = len(calib_present)

    if calib_missing:
        stream.issues.append(f"缺少标定的相机: {calib_missing}")
        stream.disposition = "keep_with_flag"
    else:
        stream.disposition = "pass"

    return stream


__all__ = [
    "CameraRole",
    "DunjiaCompletenessReport",
    "StreamCompleteness",
    "check_dunjia_completeness",
]
