"""
A2D Reader — 读取真机 A2 Episode 的全部数据流。

读取 meta_info.json、相机图像序列、aligned_joints.h5、标定参数，
统一返回 Session（含 VideoStream + TimeSeriesStream）。

Reader 只负责读取、解析字段、建立统一 Stream、提取标定。
不在 Reader 中执行黑屏判断、异常裁剪或 Segment 规划。

用法:
    from zpds_prepare.readers.a2d_reader import read_session

    session = read_session(Path("E:/datasets/真机/A2D/"))
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from zpds_prepare.readers.session_model import (
    Session,
    VideoStream,
    TimeSeriesStream,
)

# 三路相机预期文件
CAMERA_RGB_FILES: dict[str, str] = {
    "head_rgb": "head_color.jpg",
    "hand_left_rgb": "hand_left_color.jpg",
    "hand_right_rgb": "hand_right_color.jpg",
}

# 三路深度文件
CAMERA_DEPTH_FILES: dict[str, str] = {
    "head_depth": "head_depth.png",
    "hand_left_depth": "hand_left_depth.png",
    "hand_right_depth": "hand_right_depth.png",
}

# 相机标称分辨率（基于 RealSense D435/D405 默认配置）
CAMERA_NOMINAL_RESOLUTION: dict[str, tuple[int, int]] = {
    "head_rgb": (640, 480),
    "hand_left_rgb": (640, 480),
    "hand_right_rgb": (640, 480),
    "head_depth": (640, 480),
    "hand_left_depth": (640, 480),
    "hand_right_depth": (640, 480),
}

# aligned_joints.h5 → TimeSeriesStream 字段映射
ROBOT_STATE_FIELDS = ["positions", "velocities", "efforts", "temperatures"]
ROBOT_ACTION_FIELDS = [
    "positions", "velocities", "accelerations",
    "decelerations", "efforts", "torque_rates",
]
GRIPPER_STATE_FIELDS = ["positions"]
GRIPPER_ACTION_FIELDS = [
    "positions", "velocities", "efforts",
    "param1", "param2", "param3",
]


# ======================================================================
# 辅助
# ======================================================================

def _parse_text_field(value: Any) -> dict:
    """安全解析 meta_info.json 中的 text 字段（可能是 dict 或 Python 字面量字符串）。

    text 字段存储为 JSON 字符串，内容是 Python dict 字面量（含 unicode escape）。
    JSON 序列化时可能将原 Python 字符串中的 \\n 转为真实换行符，
    导致 ast.literal_eval 失败——此处做兼容处理。
    """
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        # 尝试①: 直接解析
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, SyntaxError):
            pass

        # 尝试②: 修复真实换行符（JSON 将 \n 展开为实际换行后，
        #         Python 字符串字面量中的换行会导致 SyntaxError）
        try:
            # 将不在三引号内的真实换行符替换为 \\n
            fixed = _escape_newlines_in_python_str(value)
            parsed = ast.literal_eval(fixed)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

        return {"raw_text": value}
    return {"raw_value": value}


def _escape_newlines_in_python_str(s: str) -> str:
    """修复 Python 字面量字符串中因 JSON 展开而变成真实换行的 \\n。

    简单策略：在单引号字符串内部，将真实换行替换为 \\\\n。
    不处理嵌套引号和三引号的复杂情况（A2D text 字段足够简单）。
    """
    result: list[str] = []
    in_string = False
    string_char: str | None = None
    i = 0

    while i < len(s):
        ch = s[i]

        if not in_string:
            if ch in ("'", '"'):
                in_string = True
                string_char = ch
            result.append(ch)
        else:
            if ch == "\\":
                # 转义序列——保留原样
                result.append(ch)
                if i + 1 < len(s):
                    result.append(s[i + 1])
                    i += 1
            elif ch == "\n":
                # 真实换行 → 转义回 \\n
                result.append("\\n")
            elif ch == string_char:
                in_string = False
                string_char = None
                result.append(ch)
            else:
                result.append(ch)
        i += 1

    return "".join(result)


def _load_json(path: Path) -> dict:
    """加载 JSON 文件。"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ======================================================================
# 4.1 读取 meta_info.json
# ======================================================================

def _read_meta(episode_root: Path) -> dict:
    """提取 meta_info.json 中的关键字段。

    Returns:
        扁平化元数据 dict（直接作为 Session.meta）。
    """
    meta_path = episode_root / "meta_info.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"meta_info.json 不存在: {meta_path}")

    raw = _load_json(meta_path)

    # 安全解析 text 字段
    text_parsed = _parse_text_field(raw.get("text", {}))

    return {
        # 标识
        "episode_id": str(raw.get("episode_id", "")),
        "episode_token": raw.get("episode_token", ""),
        "task_id": raw.get("task_id"),
        "job_id": raw.get("job_id"),
        "AID": raw.get("AID", ""),

        # 机器人
        "robot_type": raw.get("robot_type", ""),
        "ee_type": raw.get("ee_type", ""),

        # 时间
        "clip_start_time": raw.get("clip_start_time"),  # float seconds Unix epoch
        "clip_end_time": raw.get("clip_end_time"),
        "duration_s": raw.get("duration"),
        "create_time": raw.get("create_time"),

        # 相机
        "camera_list": raw.get("camera_list", []),
        "camera_type": raw.get("camera_type", []),
        "camera_fps": raw.get("camera_fps", []),
        "sensor_type": raw.get("sensor_type", []),

        # 任务文本
        "task_description": text_parsed.get("description", "").strip(),

        # 状态
        "is_aligned": raw.get("is_aligned", False),
        "integrity": raw.get("integrity", ""),
        "status": raw.get("status"),

        # 原始全量（供下游按需访问）
        "_raw_meta": raw,
    }


# ======================================================================
# 4.2 相机图像序列 → VideoStream
# ======================================================================

def _build_camera_streams(
    episode_root: Path,
    aligned_timestamps_ns: list[int] | None,
    meta: dict,
) -> dict[str, VideoStream]:
    """扫描 camera/ 目录，为每路 RGB 相机建立 VideoStream。

    Args:
        episode_root: Episode 根目录。
        aligned_timestamps_ns: aligned_joints.h5 的 timestamp 数组，
                               用于 frame_index → 纳秒时间戳映射。
        meta: 从 _read_meta() 返回的元数据 dict。

    Returns:
        {"head_rgb": VideoStream, "hand_left_rgb": VideoStream, "hand_right_rgb": VideoStream}
    """
    camera_root = episode_root / "camera"
    video_streams: dict[str, VideoStream] = {}

    # 扫描所有 frame_idx 目录
    frame_dirs: dict[int, Path] = {}
    if camera_root.is_dir():
        for frame_dir in camera_root.iterdir():
            if not frame_dir.is_dir():
                continue
            try:
                idx = int(frame_dir.name)
            except ValueError:
                continue
            frame_dirs[idx] = frame_dir

    sorted_indices = sorted(frame_dirs.keys())

    # 为每路相机收集帧
    for stream_id, filename in CAMERA_RGB_FILES.items():
        index_frames: list[dict] = []
        timestamps_ns: list[int] = []

        for idx in sorted_indices:
            source_path = frame_dirs[idx] / filename
            if not source_path.is_file():
                continue  # 跳过不完整帧

            # 时间戳：优先从 aligned_joints 取，否则 pending
            ts: int | None = None
            ts_method = "pending_alignment"
            ts_error: int | None = None

            if aligned_timestamps_ns is not None and idx < len(aligned_timestamps_ns):
                ts = int(aligned_timestamps_ns[idx])
                ts_method = "aligned_joints_index"
                ts_error = 0  # 精确映射（帧级对齐）

            index_frames.append({
                "seq": len(index_frames),
                "frame_index": idx,
                "source_path": str(source_path),
                "source_timestamp_ns": ts,
                "timestamp_method": ts_method,
                "timestamp_error_ns": ts_error,
            })
            if ts is not None:
                timestamps_ns.append(ts)

        # 相机标称 fps
        camera_index = list(CAMERA_RGB_FILES.keys()).index(stream_id)
        nominal_fps = 30.0
        if camera_index < len(meta.get("camera_fps", [])):
            nominal_fps = float(meta["camera_fps"][camera_index])

        # 标称分辨率
        nominal_w, nominal_h = CAMERA_NOMINAL_RESOLUTION.get(
            stream_id, (640, 480)
        )

        video_streams[stream_id] = VideoStream(
            stream_id=stream_id,
            timestamps_ns=(
                timestamps_ns if timestamps_ns else [0] * len(index_frames)
            ),
            index_frames=index_frames,
            video_path=str(camera_root),  # 目录而非文件
            fps=nominal_fps,
            width=nominal_w,
            height=nominal_h,
            frame_count=len(index_frames),
        )

        # 通过 metadata 传递图像序列特有信息
        # (VideoStream 没有 metadata 字段，但 dataclass 允许运行时赋值)
        video_streams[stream_id].__dict__["_source_kind"] = "image_sequence"
        video_streams[stream_id].__dict__["_frame_indices"] = sorted_indices
        video_streams[stream_id].__dict__["_depth_available"] = (
            (episode_root / "camera" / str(sorted_indices[0]) / CAMERA_DEPTH_FILES.get(
                stream_id.replace("_rgb", "_depth"), ""
            )).exists()
            if sorted_indices else False
        )

    return video_streams


def _scan_frame_dirs(camera_root: Path) -> dict[int, Path]:
    """扫描 camera/ 目录，返回 {frame_index: directory_path}。"""
    frame_dirs: dict[int, Path] = {}
    if camera_root.is_dir():
        for frame_dir in camera_root.iterdir():
            if not frame_dir.is_dir():
                continue
            try:
                idx = int(frame_dir.name)
            except ValueError:
                continue
            frame_dirs[idx] = frame_dir
    return frame_dirs


def _build_depth_streams(
    episode_root: Path,
    frame_dirs: dict[int, Path],
    sorted_indices: list[int],
    aligned_timestamps_ns: list[int] | None,
    meta: dict,
) -> dict[str, VideoStream]:
    """为三路深度相机建立 VideoStream（图像序列，不转码）。

    深度 PNG 为 uint16，需保持原始精度，不转为 MP4。

    Returns:
        {"head_depth": VideoStream, "hand_left_depth": VideoStream, "hand_right_depth": VideoStream}
    """
    depth_streams: dict[str, VideoStream] = {}

    for stream_id, filename in CAMERA_DEPTH_FILES.items():
        index_frames: list[dict] = []
        timestamps_ns: list[int] = []

        for idx in sorted_indices:
            source_path = frame_dirs[idx] / filename
            if not source_path.is_file():
                continue

            ts: int | None = None
            ts_method = "pending_alignment"
            ts_error: int | None = None

            if aligned_timestamps_ns is not None and idx < len(aligned_timestamps_ns):
                ts = int(aligned_timestamps_ns[idx])
                ts_method = "aligned_joints_index"
                ts_error = 0

            index_frames.append({
                "seq": len(index_frames),
                "frame_index": idx,
                "source_path": str(source_path),
                "source_timestamp_ns": ts,
                "timestamp_method": ts_method,
                "timestamp_error_ns": ts_error,
            })
            if ts is not None:
                timestamps_ns.append(ts)

        nominal_w, nominal_h = CAMERA_NOMINAL_RESOLUTION.get(
            stream_id, (640, 480)
        )

        depth_streams[stream_id] = VideoStream(
            stream_id=stream_id,
            timestamps_ns=(
                timestamps_ns if timestamps_ns else [0] * len(index_frames)
            ),
            index_frames=index_frames,
            video_path=str(episode_root / "camera"),
            fps=30.0,
            width=nominal_w,
            height=nominal_h,
            frame_count=len(index_frames),
        )

        # 标记流属性
        vs = depth_streams[stream_id]
        vs.__dict__["_source_kind"] = "image_sequence"
        vs.__dict__["_frame_indices"] = sorted_indices
        vs.__dict__["_modality"] = "depth"
        vs.__dict__["_encoding"] = "png16"
        vs.__dict__["_dtype"] = "uint16"

    return depth_streams


# ======================================================================
# 4.3 aligned_joints.h5 → TimeSeriesStream × 4
# ======================================================================

def _load_joint_names(episode_root: Path) -> list[str]:
    """从 parameters/meshes/joint_map.json 加载关节名列表。

    Returns:
        按索引 0..n-1 排序的关节名列表。
    """
    joint_map_path = episode_root / "parameters" / "meshes" / "joint_map.json"
    if joint_map_path.is_file():
        joint_map = _load_json(joint_map_path)
        # 过滤 mapping >= 0，按值排序
        active = [(k, v) for k, v in joint_map.items() if v >= 0]
        active.sort(key=lambda x: x[1])
        return [name for name, _ in active]

    # 回退：硬编码已知关节顺序
    return [
        "joint31", "joint32", "joint33", "joint34",
        "joint51", "joint52", "joint53", "joint54",
        "joint55", "joint56", "joint57",
        "joint61", "joint62", "joint63", "joint64",
        "joint65", "joint66", "joint67",
    ]


def _build_time_series_streams(
    episode_root: Path,
) -> tuple[dict[str, TimeSeriesStream], list[int] | None]:
    """从 aligned_joints.h5 构建 4 个 TimeSeriesStream。

    Returns:
        (time_series_streams, aligned_timestamps_ns)
        若 aligned_joints.h5 不存在，返回 ({}, None)。
    """
    h5_path = episode_root / "aligned_joints.h5"
    if not h5_path.is_file():
        return {}, None

    joint_names = _load_joint_names(episode_root)
    num_joints = len(joint_names)

    streams: dict[str, TimeSeriesStream] = {}

    with h5py.File(h5_path, "r") as f:
        # --- 共享时间轴 ---
        ts_all = f["timestamp"][:]  # ndarray[int64]
        timestamps_ns = [int(v) for v in ts_all]
        num_samples = len(timestamps_ns)

        source_path = h5_path

        # --- robot_state ---
        robot_state_fields: list[dict[str, Any]] = []
        robot_state_rows = _build_field_columns(
            f, "state/robot", timestamps_ns,
            ROBOT_STATE_FIELDS, joint_names, robot_state_fields,
        )

        streams["robot_state"] = TimeSeriesStream(
            stream_id="robot_state",
            modality="joint_state",
            role="state",
            source_path=source_path,
            timestamps_ns=timestamps_ns,
            rows=robot_state_rows,
            fields=robot_state_fields,
            expected_rate_hz=_estimate_rate_hz(timestamps_ns),
            frame_id="robot_base",
            metadata={
                "joint_names": joint_names,
                "joint_order_source": "parameters/meshes/joint_map.json",
                "num_joints": num_joints,
                "num_samples": num_samples,
                "hdf5_source": str(h5_path),
            },
        )

        # --- robot_action ---
        robot_action_fields: list[dict[str, Any]] = []
        robot_action_rows = _build_field_columns(
            f, "action/robot", timestamps_ns,
            ROBOT_ACTION_FIELDS, joint_names, robot_action_fields,
        )

        streams["robot_action"] = TimeSeriesStream(
            stream_id="robot_action",
            modality="joint_command",
            role="action",
            source_path=source_path,
            timestamps_ns=timestamps_ns,
            rows=robot_action_rows,
            fields=robot_action_fields,
            expected_rate_hz=_estimate_rate_hz(timestamps_ns),
            frame_id="robot_base",
            metadata={
                "joint_names": joint_names,
                "joint_order_source": "parameters/meshes/joint_map.json",
                "num_joints": num_joints,
                "num_samples": num_samples,
                "hdf5_source": str(h5_path),
                "nan_note": (
                    "action 字段含大量 NaN——未执行动作的时间步"
                    "（插值空白）用 NaN 填充。"
                ),
            },
        )

        # --- gripper_state ---
        gripper_names = ["right_joint1", "left_joint1"]
        gripper_state_fields: list[dict[str, Any]] = []
        gripper_state_rows = _build_field_columns(
            f, "state/gripper", timestamps_ns,
            GRIPPER_STATE_FIELDS, gripper_names, gripper_state_fields,
        )

        streams["gripper_state"] = TimeSeriesStream(
            stream_id="gripper_state",
            modality="gripper_state",
            role="state",
            source_path=source_path,
            timestamps_ns=timestamps_ns,
            rows=gripper_state_rows,
            fields=gripper_state_fields,
            expected_rate_hz=_estimate_rate_hz(timestamps_ns),
            frame_id="gripper",
            metadata={
                "gripper_names": gripper_names,
                "num_joints": len(gripper_names),
                "num_samples": num_samples,
                "hdf5_source": str(h5_path),
            },
        )

        # --- gripper_action ---
        gripper_action_fields: list[dict[str, Any]] = []
        gripper_action_rows = _build_field_columns(
            f, "action/gripper", timestamps_ns,
            GRIPPER_ACTION_FIELDS, gripper_names, gripper_action_fields,
        )

        streams["gripper_action"] = TimeSeriesStream(
            stream_id="gripper_action",
            modality="gripper_command",
            role="action",
            source_path=source_path,
            timestamps_ns=timestamps_ns,
            rows=gripper_action_rows,
            fields=gripper_action_fields,
            expected_rate_hz=_estimate_rate_hz(timestamps_ns),
            frame_id="gripper",
            metadata={
                "gripper_names": gripper_names,
                "num_joints": len(gripper_names),
                "num_samples": num_samples,
                "hdf5_source": str(h5_path),
                "nan_note": (
                    "gripper action 字段几乎全为 NaN——本 Episode 未执行"
                    "夹爪动作指令。"
                ),
            },
        )

    return streams, timestamps_ns


def _build_field_columns(
    h5f: h5py.File,
    group_prefix: str,
    timestamps_ns: list[int],
    field_names: list[str],
    joint_names: list[str],
    fields_out: list[dict[str, Any]],
) -> np.ndarray:
    """从 HDF5 group 读取指定字段，展开为宽表列。

    Args:
        h5f: 打开的 HDF5 文件。
        group_prefix: HDF5 group 路径前缀，如 "state/robot"。
        timestamps_ns: 时间戳列表（用于行数校验）。
        field_names: 要读取的字段名列表。
        joint_names: 关节名列表（用于生成列名）。
        fields_out: (输出) 将被填充 [{name, dtype, unit?, ...}, ...]。

    Returns:
        shape=(len(timestamps_ns), total_columns) 的 float64 ndarray。
    """
    columns: list[np.ndarray] = []

    for fname in field_names:
        ds_path = f"{group_prefix}/{fname}"
        if ds_path not in h5f:
            continue

        arr = h5f[ds_path][:]  # shape: (N, DOF)

        # 对齐行数（aligned 文件所有 Dataset 第一维相同，此处防御性截断）
        n_rows = min(len(timestamps_ns), arr.shape[0])
        arr = arr[:n_rows, :]

        dof = arr.shape[1]
        for j in range(dof):
            col = arr[:, j]
            columns.append(col)

            # 列名
            if dof == len(joint_names):
                col_name = f"{joint_names[j]}_{fname}"
            else:
                col_name = f"joint_{j:02d}_{fname}"

            fields_out.append({
                "name": col_name,
                "dtype": "float64",
                "unit": _infer_unit(fname),
                "source_hdf5_path": ds_path,
                "column_index": j,
            })

    if not columns:
        return np.zeros((len(timestamps_ns), 0), dtype=np.float64)

    return np.column_stack(columns)


def _infer_unit(field_name: str) -> str:
    """根据字段名推断物理单位。"""
    unit_map = {
        "positions": "rad",
        "velocities": "rad/s",
        "accelerations": "rad/s²",
        "decelerations": "rad/s²",
        "efforts": "N·m",
        "torque_rates": "N·m/s",
        "temperatures": "°C",
        "param1": "",
        "param2": "",
        "param3": "",
    }
    return unit_map.get(field_name, "")


def _estimate_rate_hz(timestamps_ns: list[int]) -> float:
    """从时间戳估算平均采样率。"""
    if len(timestamps_ns) < 2:
        return 0.0
    diffs = np.diff(np.array(timestamps_ns, dtype=np.float64))
    mean_dt_ns = diffs.mean()
    if mean_dt_ns <= 0:
        return 0.0
    return round(1e9 / mean_dt_ns, 1)


# ======================================================================
# 标定提取
# ======================================================================

def _extract_calibration(episode_root: Path) -> dict:
    """从 parameters/camera/ 提取相机内参，构建标定 dict。

    Returns:
        {
            "calibration_id": str,
            "cameras": [
                {camera_id, model, intrinsics: {fx, fy, cx, cy, distortion_model, ...}},
                ...
            ],
        }
    """
    cameras = []
    calib_dir = episode_root / "parameters" / "camera"

    cam_map = {
        "head": "head_rgb",
        "hand_left": "hand_left_rgb",
        "hand_right": "hand_right_rgb",
    }

    for cam_name, stream_id in cam_map.items():
        calib_path = calib_dir / f"{cam_name}_intrinsic_params.json"
        if not calib_path.is_file():
            continue

        raw = _load_json(calib_path)

        # 归一化字段名
        cameras.append({
            "camera_id": stream_id,
            "model": "pinhole",
            "intrinsics": {
                "fx": raw.get("fx"),
                "fy": raw.get("fy"),
                "cx": raw.get("ppx"),
                "cy": raw.get("ppy"),
                "distortion_model": raw.get("distortion_model", "brown_conrady"),
                "distortion_coeffs": [
                    raw.get("k1", 0.0),
                    raw.get("k2", 0.0),
                    raw.get("k3", 0.0),
                    raw.get("p1", 0.0),
                    raw.get("p2", 0.0),
                ],
            },
            "resolution": {
                "width": raw.get("width"),
                "height": raw.get("height"),
            },
            "source_file": str(calib_path),
        })

    return {
        "calibration_id": "a2d_parameter_camera",
        "cameras": cameras,
    }


# ======================================================================
# 主入口
# ======================================================================

def read_session(
    episode_root: Path,
    config: dict | None = None,
) -> Session:
    """读取 A2D Episode 的全部数据流。

    Args:
        episode_root: Episode 根目录路径（含 meta_info.json / camera/ / aligned_joints.h5）。
        config: 可选配置字典（暂未使用，为扩展预留）。

    Returns:
        Session 对象:
          - video_streams: head_rgb, hand_left_rgb, hand_right_rgb
          - imu_streams: {} (A2D 无独立 IMU 流)
          - annotation_streams: {}
          - time_series_streams: robot_state, robot_action, gripper_state, gripper_action
    """
    if config is None:
        config = {}

    episode_root = Path(episode_root)
    if not episode_root.is_dir():
        raise FileNotFoundError(f"Episode 目录不存在: {episode_root}")

    # ----------------------------------------------------------------
    # 1. meta_info.json
    # ----------------------------------------------------------------
    meta = _read_meta(episode_root)

    # ----------------------------------------------------------------
    # 2. aligned_joints.h5 → time_series_streams + 共享时间轴
    # ----------------------------------------------------------------
    time_series_streams, aligned_timestamps_ns = _build_time_series_streams(
        episode_root,
    )

    # ----------------------------------------------------------------
    # 3. 相机图像序列 → VideoStream (RGB)
    # ----------------------------------------------------------------
    camera_root = episode_root / "camera"
    frame_dirs = _scan_frame_dirs(camera_root)
    sorted_indices = sorted(frame_dirs.keys())

    video_streams = _build_camera_streams(
        episode_root, aligned_timestamps_ns, meta,
    )
    # 改用已扫描的 frame_dirs（避免重复扫描）
    # _build_camera_streams 内部也扫描了，但结果相同，此处仅作记录

    # ----------------------------------------------------------------
    # 3b. 深度图像序列 → VideoStream (Depth)
    # ----------------------------------------------------------------
    depth_streams = _build_depth_streams(
        episode_root, frame_dirs, sorted_indices,
        aligned_timestamps_ns, meta,
    )
    video_streams.update(depth_streams)

    # ----------------------------------------------------------------
    # 4. 标定
    # ----------------------------------------------------------------
    calibration = _extract_calibration(episode_root)

    # ----------------------------------------------------------------
    # 5. 组装 Session
    # ----------------------------------------------------------------
    episode_id = meta.get("episode_id", "unknown")

    # 深度可用性统计（基于已构建的 depth_streams）
    depth_available = {}
    for depth_id in CAMERA_DEPTH_FILES:
        ds = depth_streams.get(depth_id)
        depth_available[depth_id] = (
            ds.frame_count > 0 if ds else False
        )

    # RGB-Depth 配对率
    rgb_depth_pairing = {}
    for rgb_id in CAMERA_RGB_FILES:
        depth_id = rgb_id.replace("_rgb", "_depth")
        rgb_count = video_streams.get(rgb_id, VideoStream).frame_count if rgb_id in video_streams else 0
        depth_count = depth_streams.get(depth_id, VideoStream).frame_count if depth_id in depth_streams else 0
        rgb_depth_pairing[rgb_id] = {
            "rgb_frames": rgb_count,
            "depth_frames": depth_count,
            "pairing_rate": round(depth_count / rgb_count, 4) if rgb_count > 0 else 0.0,
        }

    return Session(
        session_id=f"a2d_{episode_id}",
        source_path=str(episode_root),
        meta={
            # 从 meta_info 提取
            **{k: v for k, v in meta.items() if not k.startswith("_")},

            # 数据源特征
            "device": f"A2D-{meta.get('robot_type', 'unknown')}",
            "profile": "a2d",
            "source_kind": "episode_directory",

            # 流概览
            "num_camera_streams": len(video_streams),
            "num_rgb_streams": len(CAMERA_RGB_FILES),
            "num_depth_streams": len(depth_streams),
            "num_time_series_streams": len(time_series_streams),
            "camera_frame_count": (
                next(iter(video_streams.values())).frame_count
                if video_streams else 0
            ),
            "aligned_samples": (
                next(iter(time_series_streams.values())).num_samples
                if time_series_streams else 0
            ),

            # 深度
            "depth_available": depth_available,
            "rgb_depth_pairing": rgb_depth_pairing,

            # 标定
            "calibration": calibration,
        },
        video_streams=video_streams,
        imu_streams={},
        annotation_streams={},
        time_series_streams=time_series_streams,
    )


__all__ = [
    "read_session",
]
