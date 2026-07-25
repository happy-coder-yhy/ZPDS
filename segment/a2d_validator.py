"""
A2D Prepared Segment 验证器。

对已生成的 Prepared Segment 执行完整性验证：
  - 三路 RGB MP4 可解码、帧数一致、时间单调
  - robot_state Parquet 结构正确、无 NaN/Inf、时间单调
  - robot_action Parquet 维度正确、控制模式合法
  - 相机↔机器人对齐质量

用法:
    from segment.a2d_validator import validate_segment

    report = validate_segment(Path("prepared_segments/seg_000001"))
    # → {"status": "pass_with_warning", "checks": {...}, "statistics": {...}}
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ======================================================================
# 主入口
# ======================================================================

def validate_segment(segment_dir: Path) -> dict:
    """对单个 Prepared Segment 执行全部 A2D 验证。

    Args:
        segment_dir: Prepared Segment 根目录（含 segment.json）。

    Returns:
        validation_report dict:
          {
            "status": "pass" | "pass_with_warning" | "fail",
            "checks": {check_id: "pass" | "warning" | "fail", ...},
            "statistics": {...},
          }
    """
    segment_dir = Path(segment_dir)
    segment = _load_segment_json(segment_dir)

    checks: dict[str, str] = {}
    stats: dict[str, Any] = {}

    # ---- 视频验证 ----
    _validate_three_rgb_streams(segment_dir, segment, checks, stats)

    # ---- Robot State 验证 ----
    _validate_robot_state(segment_dir, segment, checks, stats)

    # ---- Robot Action 验证 ----
    _validate_robot_action(segment_dir, segment, checks, stats)

    # ---- Camera↔Robot Alignment 验证 ----
    _validate_camera_robot_alignment(segment_dir, segment, checks, stats)

    # ---- Depth 验证 ----
    _validate_depth_streams(segment_dir, segment, checks, stats)

    # ---- 汇总 status ----
    status = _compute_status(checks)

    return {
        "status": status,
        "checks": checks,
        "statistics": stats,
    }


# ======================================================================
# 辅助
# ======================================================================

def _load_segment_json(segment_dir: Path) -> dict:
    """加载 segment.json。"""
    seg_path = segment_dir / "segment.json"
    if not seg_path.is_file():
        raise FileNotFoundError(f"segment.json 不存在: {seg_path}")
    with open(seg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _find_stream(segment: dict, stream_id: str) -> dict | None:
    """在 segment["streams"] 中按 stream_id 查找流描述。"""
    for s in segment.get("streams", []):
        if s.get("stream_id") == stream_id:
            return s
    return None


def _compute_status(checks: dict[str, str]) -> str:
    """从所有检查结果计算总体状态。"""
    if any(v == "fail" for v in checks.values()):
        return "fail"
    if any(v == "warning" for v in checks.values()):
        return "pass_with_warning"
    return "pass"


# ======================================================================
# 视频验证
# ======================================================================

def _validate_three_rgb_streams(
    segment_dir: Path,
    segment: dict,
    checks: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """验证三路 RGB MP4 可解码、帧数与 sample_map 一致、时间单调。"""
    import cv2

    camera_ids = ["head_rgb", "hand_left_rgb", "hand_right_rgb"]
    all_pass = True
    any_warn = False

    for cam_id in camera_ids:
        stream = _find_stream(segment, cam_id)
        if stream is None:
            checks[f"{cam_id}_video"] = "fail"
            all_pass = False
            continue

        mp4_path = segment_dir / stream["uri"]
        if not mp4_path.is_file():
            checks[f"{cam_id}_video"] = "fail"
            all_pass = False
            continue

        # 可解码
        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            checks[f"{cam_id}_video"] = "fail"
            all_pass = False
            continue

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        stats[f"{cam_id}_frames"] = frame_count
        stats.setdefault("video_fps", fps)
        stats.setdefault("video_resolution", f"{width}x{height}")

        # 帧数与 sample_map 一致
        sample_map_uri = stream.get("origin", {}).get("sample_map_uri", "")
        sm_path = segment_dir / sample_map_uri
        if sm_path.is_file():
            sm = pd.read_parquet(str(sm_path))
            sm_rows = len(sm)
            if sm_rows != frame_count:
                checks[f"{cam_id}_video"] = "fail"
                all_pass = False
                continue

            # 时间单调
            if not sm["output_timestamp_ns"].is_monotonic_increasing:
                checks[f"{cam_id}_video"] = "fail"
                all_pass = False
                continue

            # 检查源图像是否存在（抽样）
            sample_sources = sm["source_path"].head(10).tolist()
            missing = [p for p in sample_sources if not Path(p).is_file()]
            if missing:
                # 源图像缺失 → warning（可能只是部分帧路径问题）
                any_warn = True
        else:
            any_warn = True

    # 汇总三路结果
    if not all_pass:
        checks["three_rgb_streams_readable"] = "fail"
    elif any_warn:
        checks["three_rgb_streams_readable"] = "warning"
    else:
        checks["three_rgb_streams_readable"] = "pass"


# ======================================================================
# Robot State 验证
# ======================================================================

def _validate_robot_state(
    segment_dir: Path,
    segment: dict,
    checks: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """验证 robot_state Parquet。"""
    stream = _find_stream(segment, "robot_state")
    if stream is None:
        checks["robot_state_valid"] = "fail"
        return

    parquet_path = segment_dir / stream["uri"]
    if not parquet_path.is_file():
        checks["robot_state_valid"] = "fail"
        return

    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception:
        checks["robot_state_valid"] = "fail"
        return

    stats["robot_state_rows"] = len(df)

    duration_ns = segment.get("timeline", {}).get("end_ns", 0)

    # 时间单调
    if not df["timestamp_ns"].is_monotonic_increasing:
        checks["robot_state_valid"] = "fail"
        return

    # 时间位于 Segment 内
    if df["timestamp_ns"].min() < 0 or df["timestamp_ns"].max() > duration_ns:
        checks["robot_state_valid"] = "fail"
        return

    # positions shape = 18
    positions_fields = _get_group_fields(stream, "positions")
    if len(positions_fields) != 18:
        checks["robot_state_valid"] = "fail"
        return
    # 验证 column 存在
    pos_cols = [f"{j}_positions" for j in _load_joint_names(segment_dir)]
    missing_pos = [c for c in pos_cols if c not in df.columns]
    # 回退：用 segment.json 的 fields 推导列名
    if missing_pos:
        # 尝试从 parquet 实际列中匹配
        actual_pos_cols = [c for c in df.columns if c.endswith("_positions")]
        if len(actual_pos_cols) != 18:
            checks["robot_state_valid"] = "fail"
            return

    # 不存在 NaN / Inf
    all_position_cols = [c for c in df.columns if c.endswith("_positions")]
    if not all_position_cols:
        checks["robot_state_valid"] = "fail"
        return

    pos_vals = df[all_position_cols].values
    nan_count = int(pd.isna(pos_vals).sum())
    inf_count = int(np.isinf(pos_vals).sum())
    stats["robot_state_position_nan_count"] = nan_count
    stats["robot_state_position_inf_count"] = inf_count

    if nan_count > 0 or inf_count > 0:
        checks["robot_state_valid"] = "fail"
        return

    # joint_names 数量 = 18
    joint_names = _load_joint_names(segment_dir)
    stats["robot_state_joint_names_count"] = len(joint_names)
    if len(joint_names) != 18:
        checks["robot_state_valid"] = "fail"
        return

    checks["robot_state_valid"] = "pass"


def _get_group_fields(stream: dict, group_name: str) -> list[dict]:
    """从 segment.json 流描述中获取指定 field group。"""
    for f in stream.get("fields", []):
        if f.get("name") == group_name:
            return list(range(f.get("shape", [0])[0]))
    return []


def _load_joint_names(segment_dir: Path) -> list[str]:
    """加载 metadata/joint_names.json。"""
    jn_path = segment_dir / "metadata" / "joint_names.json"
    if jn_path.is_file():
        with open(jn_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ======================================================================
# Robot Action 验证
# ======================================================================

# robot_action 预期的 6 个字段组
_EXPECTED_ACTION_GROUPS = [
    "positions", "velocities", "accelerations",
    "decelerations", "efforts", "torque_rates",
]


def _validate_robot_action(
    segment_dir: Path,
    segment: dict,
    checks: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """验证 robot_action Parquet。"""
    stream = _find_stream(segment, "robot_action")
    if stream is None:
        checks["robot_action_valid"] = "fail"
        return

    parquet_path = segment_dir / stream["uri"]
    if not parquet_path.is_file():
        checks["robot_action_valid"] = "fail"
        return

    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception:
        checks["robot_action_valid"] = "fail"
        return

    stats["robot_action_rows"] = len(df)

    duration_ns = segment.get("timeline", {}).get("end_ns", 0)

    # 时间单调
    if not df["timestamp_ns"].is_monotonic_increasing:
        checks["robot_action_valid"] = "fail"
        return

    # 时间位于 Segment 内
    if df["timestamp_ns"].min() < 0 or df["timestamp_ns"].max() > duration_ns:
        checks["robot_action_valid"] = "fail"
        return

    # 动作维度正确：6 组 × 18 关节
    action_fields = stream.get("fields", [])
    action_group_names = {f["name"] for f in action_fields}
    missing_groups = set(_EXPECTED_ACTION_GROUPS) - action_group_names
    if missing_groups:
        checks["robot_action_valid"] = "fail"
        return

    for fld in action_fields:
        if fld["name"] in _EXPECTED_ACTION_GROUPS:
            if fld.get("shape", []) != [18]:
                checks["robot_action_valid"] = "fail"
                return

    # 控制模式合法：至少 positions 有部分非 NaN（表示有控制指令）
    pos_cols = [c for c in df.columns if c.endswith("_positions")]
    if pos_cols:
        pos_vals = df[pos_cols].values
        non_nan_ratio = 1.0 - pd.isna(pos_vals).sum() / pos_vals.size
        stats["robot_action_positions_valid_ratio"] = round(non_nan_ratio, 4)
        if non_nan_ratio == 0:
            checks["robot_action_valid"] = "fail"
            return

    # 状态与动作有公共时间范围
    state_stream = _find_stream(segment, "robot_state")
    if state_stream:
        state_path = segment_dir / state_stream["uri"]
        if state_path.is_file():
            try:
                rs = pd.read_parquet(str(state_path))
                overlap_start = max(
                    rs["timestamp_ns"].min(), df["timestamp_ns"].min()
                )
                overlap_end = min(
                    rs["timestamp_ns"].max(), df["timestamp_ns"].max()
                )
                stats["state_action_overlap_ns"] = int(
                    max(0, overlap_end - overlap_start)
                )
                if overlap_start >= overlap_end:
                    checks["robot_action_valid"] = "fail"
                    return
            except Exception:
                pass

    checks["robot_action_valid"] = "pass"


# ======================================================================
# Camera↔Robot Alignment 验证
# ======================================================================

def _validate_camera_robot_alignment(
    segment_dir: Path,
    segment: dict,
    checks: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """验证每个视频输出帧可映射回源帧，且相机帧能对应机器人状态。"""
    camera_ids = ["head_rgb", "hand_left_rgb", "hand_right_rgb"]
    all_error_ns: list[float] = []
    total_source_frames = 0
    any_warn = False
    max_error_ns = 0

    for cam_id in camera_ids:
        stream = _find_stream(segment, cam_id)
        if stream is None:
            continue

        sample_map_uri = stream.get("origin", {}).get("sample_map_uri", "")
        sm_path = segment_dir / sample_map_uri
        if not sm_path.is_file():
            any_warn = True
            continue

        sm = pd.read_parquet(str(sm_path))
        stats[f"{cam_id}_sample_map_rows"] = len(sm)

        # 每个 output_frame_index 唯一且无缺失
        expected_indices = set(range(len(sm)))
        actual_indices = set(sm["output_frame_index"].tolist())
        if expected_indices != actual_indices:
            any_warn = True

        # source_frame_index 全部 ≥ 0
        if (sm["source_frame_index"] < 0).any():
            any_warn = True

        # 统计源帧数（去重 source_frame_index）
        distinct_sources = sm["source_frame_index"].nunique()
        total_source_frames += distinct_sources

        # time_error_ns 统计
        errors = sm["time_error_ns"].abs()
        all_error_ns.extend(errors.tolist())
        current_max = int(errors.max())
        if current_max > max_error_ns:
            max_error_ns = current_max

    stats["max_alignment_error_ns"] = max_error_ns
    # A2D 相机时间戳均由 aligned_joints 推断，无原生相机时钟
    stats["timestamp_inferred_frames"] = total_source_frames

    if all_error_ns:
        stats["mean_alignment_error_ns"] = round(float(np.mean(all_error_ns)), 1)
        stats["median_alignment_error_ns"] = round(float(np.median(all_error_ns)), 1)

        # 未跨长缺口映射：检查连续 source_timestamp_ns 差值
        # 如果存在 > 5s 的源时间跳跃，说明跨了长缺口
        long_gap_count = 0
        for cam_id in camera_ids:
            stream = _find_stream(segment, cam_id)
            if stream is None:
                continue
            sm_path = segment_dir / stream.get("origin", {}).get("sample_map_uri", "")
            if not sm_path.is_file():
                continue
            sm = pd.read_parquet(str(sm_path))
            src_ts = sm["source_timestamp_ns"].values
            diffs = np.diff(src_ts.astype(np.int64))
            long_gaps = (np.abs(diffs) > 5_000_000_000).sum()
            long_gap_count += int(long_gaps)
        stats["cross_long_gap_mappings"] = long_gap_count

    # 汇总
    if max_error_ns > 20_000_000:  # > 20ms → warning
        any_warn = True

    if any_warn:
        checks["camera_robot_alignment"] = "warning"
    else:
        checks["camera_robot_alignment"] = "pass"


# ======================================================================
# Depth 验证
# ======================================================================

_DEPTH_STREAM_IDS = ["head_depth", "hand_left_depth", "hand_right_depth"]


def _validate_depth_streams(
    segment_dir: Path,
    segment: dict,
    checks: dict[str, str],
    stats: dict[str, Any],
) -> None:
    """验证深度流：dtype uint16、分辨率一致、零值/无效值比例、RGB-Depth 配对率。"""
    import cv2

    depth_streams_found = 0
    all_pass = True
    any_warn = False

    for depth_id in _DEPTH_STREAM_IDS:
        stream = _find_stream(segment, depth_id)
        if stream is None:
            continue  # 深度流可选

        depth_streams_found += 1
        depth_dir = segment_dir / stream["uri"]

        # 检查目录存在且有 PNG
        if not depth_dir.is_dir():
            checks[f"{depth_id}_depth"] = "fail"
            all_pass = False
            continue

        png_files = sorted(depth_dir.glob("*.png"))
        if not png_files:
            checks[f"{depth_id}_depth"] = "fail"
            all_pass = False
            continue

        stats[f"{depth_id}_depth_frames"] = len(png_files)

        # 抽样检查 dtype、分辨率
        sample_count = min(20, len(png_files))
        step = max(1, len(png_files) // sample_count)
        samples = png_files[::step][:sample_count]

        widths = set()
        heights = set()
        dtypes = set()
        zero_pixels = 0
        total_pixels = 0

        for png_path in samples:
            try:
                with open(str(png_path), "rb") as fh:
                    png_bytes = fh.read()
                img = cv2.imdecode(
                    np.frombuffer(png_bytes, np.uint8), cv2.IMREAD_UNCHANGED
                )
                if img is None:
                    continue
                dtypes.add(str(img.dtype))
                widths.add(img.shape[1])
                heights.add(img.shape[0])
                total_pixels += img.size
                zero_pixels += int((img == 0).sum())
            except Exception:
                pass

        # dtype uint16
        actual_dtype = sorted(dtypes)[0] if len(dtypes) == 1 else str(dtypes)
        stats[f"{depth_id}_depth_dtype"] = actual_dtype
        if actual_dtype != "uint16":
            any_warn = True

        # 分辨率一致
        w_ok = len(widths) == 1
        h_ok = len(heights) == 1
        if w_ok and h_ok:
            stats[f"{depth_id}_depth_resolution"] = f"{widths.pop()}x{heights.pop()}"
        else:
            stats[f"{depth_id}_depth_resolution"] = "inconsistent"
            any_warn = True

        # 零值比例
        if total_pixels > 0:
            zr = round(zero_pixels / total_pixels, 6)
            stats[f"{depth_id}_depth_zero_ratio"] = zr

        # 深度单位确认
        if stream.get("unit") == "unknown":
            any_warn = True

    # RGB-Depth 配对率
    rgb_count = 0
    depth_count = 0
    for depth_id in _DEPTH_STREAM_IDS:
        rgb_id = depth_id.replace("_depth", "_rgb")
        rgb_stream = _find_stream(segment, rgb_id)
        depth_stream = _find_stream(segment, depth_id)
        if rgb_stream is None or depth_stream is None:
            continue

        # 从 sample_map 获取源帧数
        rgb_sm_uri = rgb_stream.get("origin", {}).get("sample_map_uri", "")
        depth_sm_uri = depth_stream.get("origin", {}).get("sample_map_uri", "")

        if (segment_dir / rgb_sm_uri).is_file():
            rgb_sm = pd.read_parquet(str(segment_dir / rgb_sm_uri))
            rgb_count += rgb_sm["source_frame_index"].nunique()

        if (segment_dir / depth_sm_uri).is_file():
            depth_sm = pd.read_parquet(str(segment_dir / depth_sm_uri))
            depth_count += depth_sm["source_frame_index"].nunique()

    if rgb_count > 0:
        pairing_rate = round(depth_count / rgb_count, 4)
        stats["rgb_depth_pairing_rate"] = pairing_rate
        if pairing_rate < 0.95:
            any_warn = True

    # 汇总
    if depth_streams_found == 0:
        # 无深度流 → 不报告此检查（V1 兼容）
        return

    if not all_pass:
        checks["depth_streams_valid"] = "fail"
    elif any_warn:
        checks["depth_streams_valid"] = "warning"
    else:
        checks["depth_streams_valid"] = "pass"


# ======================================================================
# 写出
# ======================================================================

def write_validation_report(report: dict, segment_dir: Path) -> str:
    """写出 validation_report.json。

    Returns:
        输出文件路径。
    """
    out_path = Path(segment_dir) / "validation_report.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(out_path)


__all__ = [
    "validate_segment",
    "write_validation_report",
]
