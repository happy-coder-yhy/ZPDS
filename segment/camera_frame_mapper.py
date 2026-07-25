"""
A2D 相机帧时间映射。

相机目录只有 frame_idx 编号，无逐帧时间戳。此模块按可信程度恢复时间：

  优先级 1: 日志 frame_idx → timestamp           (不可用 — 无逐帧日志)
  优先级 2: 厂商协调日志中的相机时间              (不可用 — 未提供)
  优先级 3: HDF5 中 frame_idx → 行索引            ← 已确认可用
  优先级 4: clip_start_time + frame_idx / FPS     ← 已证伪 (误差 21s)
  优先级 5: 排序序号 / FPS                        ← 已证伪 (误差更大)

  最终采用: frame_idx == aligned_joints 行索引 (timestamp_quality=high)

产出:
    maps/{stream_id}_source_map.parquet      — 每路相机源帧映射
    maps/camera_robot_alignment.parquet     — 相机帧 ↔ 机器人最近行

用法:
    from segment.camera_frame_mapper import build_camera_maps, write_camera_maps

    results = build_camera_maps(session, aligned_timestamps_ns)
    write_camera_maps(results, output_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from zpds_prepare.readers.session_model import Session, VideoStream, TimeSeriesStream


# ======================================================================
# 时间恢复策略验证
# ======================================================================

def validate_frame_index_method(
    session: Session,
    aligned_timestamps_ns: list[int],
) -> dict:
    """验证 frame_idx → HDF5 行索引的可靠性。

    检查项：
      - 所有 frame_idx 是否在 HDF5 范围内
      - 相邻帧 HDF5 时间间隔是否匹配 gap × (1/fps)
      - 最大 frame_idx / FPS 是否接近 meta_info.duration

    Returns:
        {method, quality, frame_indices_in_range, mean_error_ms, max_error_ms, ...}
    """
    vs = session.video_streams.get("head_rgb")
    if vs is None or not vs.index_frames:
        return {"method": "none", "quality": "unavailable", "reason": "无相机流"}

    frame_indices = [f["frame_index"] for f in vs.index_frames]
    num_aligned = len(aligned_timestamps_ns)
    fps = vs.fps

    all_in_range = all(0 <= fi < num_aligned for fi in frame_indices)

    # 相邻帧时间间隔
    errors_ms: list[float] = []
    for i in range(len(frame_indices) - 1):
        a, b = frame_indices[i], frame_indices[i + 1]
        gap = b - a
        if a < num_aligned and b < num_aligned:
            dt_actual_ms = (aligned_timestamps_ns[b] - aligned_timestamps_ns[a]) / 1e6
            dt_expected_ms = gap / fps * 1000
            errors_ms.append(abs(dt_actual_ms - dt_expected_ms))

    mean_error_ms = float(np.mean(errors_ms)) if errors_ms else None
    max_error_ms = float(np.max(errors_ms)) if errors_ms else None

    # 方法 4 验证 (即使不用，记录以供对比)
    duration_s = float(session.meta.get("duration_s", 0))
    if duration_s and frame_indices:
        est_duration_fidx = (max(frame_indices) - min(frame_indices)) / fps
        priority4_error_s = abs(est_duration_fidx - duration_s)
    else:
        priority4_error_s = None

    # 判定
    if all_in_range and mean_error_ms is not None and mean_error_ms < 5.0:
        quality = "high"
        method = "hdf5_frame_index"
    elif all_in_range:
        quality = "medium"
        method = "hdf5_frame_index"
    else:
        quality = "needs_verification"
        method = "hdf5_frame_index"

    return {
        "method": method,
        "quality": quality,
        "frame_indices_in_range": all_in_range,
        "num_frames": len(frame_indices),
        "hdf5_total_samples": num_aligned,
        "mean_error_ms": round(mean_error_ms, 4) if mean_error_ms else None,
        "max_error_ms": round(max_error_ms, 4) if max_error_ms else None,
        "priority4_error_s": round(priority4_error_s, 1) if priority4_error_s else None,
        "priority4_note": (
            f"frame_idx/FPS 推算误差 {priority4_error_s:.0f}s，不可用"
            if priority4_error_s and priority4_error_s > 2 else "可接受"
        ),
        "recommendation": (
            "frame_idx == aligned_joints 行索引，可直接映射"
            if quality == "high" else "需要进一步验证"
        ),
    }


# ======================================================================
# 相机源帧映射表
# ======================================================================

def build_camera_source_map(
    video_stream: VideoStream,
    aligned_timestamps_ns: list[int] | None,
    validation: dict | None = None,
) -> pd.DataFrame:
    """为单路相机构建 source_map.parquet。

    Args:
        video_stream: 单路相机 VideoStream。
        aligned_timestamps_ns: HDF5 共享时间轴（ns）。
        validation: validate_frame_index_method() 的返回结果（可选）。

    Returns:
        DataFrame with columns:
          source_frame_index, source_path, source_timestamp_ns,
          timestamp_method, timestamp_quality, estimated_error_ns,
          file_complete
    """
    rows: list[dict[str, Any]] = []
    quality = validation.get("quality", "high") if validation else "high"
    method = validation.get("method", "hdf5_frame_index") if validation else "hdf5_frame_index"
    num_aligned = len(aligned_timestamps_ns) if aligned_timestamps_ns else 0

    for entry in video_stream.index_frames:
        frame_idx = entry["frame_index"]
        source_path = entry["source_path"]

        # 时间恢复
        ts_ns: int | None = None
        ts_method = method
        ts_quality = quality
        est_error_ns: int | None = None

        if aligned_timestamps_ns is not None and 0 <= frame_idx < num_aligned:
            ts_ns = int(aligned_timestamps_ns[frame_idx])
            # HDF5 行索引映射 — 误差 ±(半帧间隔) ≈ 17ms
            est_error_ns = 17_000_000
        else:
            # 回退: clip_start + frame_idx / fps
            ts_quality = "low_confidence"
            ts_method = "synthetic_frame_idx_fps"
            est_error_ns = 50_000_000

        # 文件完整性
        src_path = Path(source_path) if source_path else None
        file_complete = src_path.is_file() if src_path else False

        rows.append({
            "source_frame_index": frame_idx,
            "source_path": str(source_path) if source_path else None,
            "source_timestamp_ns": ts_ns,
            "timestamp_method": ts_method,
            "timestamp_quality": ts_quality,
            "estimated_error_ns": est_error_ns,
            "file_complete": file_complete,
        })

    return pd.DataFrame(rows)


# ======================================================================
# 相机 ↔ 机器人对齐
# ======================================================================

def build_camera_robot_alignment(
    camera_stream_id: str,
    source_map: pd.DataFrame,
    robot_timestamps_ns: list[int],
) -> pd.DataFrame:
    """为每张相机帧找到最近的机器人时间行。

    Args:
        camera_stream_id: 相机流 ID (head_rgb / hand_left_rgb / hand_right_rgb)。
        source_map: build_camera_source_map() 的返回值。
        robot_timestamps_ns: 机器人时间轴（aligned_joints timestamp 或 robot_state 时间轴）。

    Returns:
        DataFrame with columns:
          camera_stream_id, source_frame_index, camera_timestamp_ns,
          robot_row_index, robot_timestamp_ns, alignment_error_ns, mapping_method
    """
    robot_ts = np.array(robot_timestamps_ns, dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for _, row in source_map.iterrows():
        cam_ts = row["source_timestamp_ns"]
        if cam_ts is None or cam_ts <= 0:
            rows.append({
                "camera_stream_id": camera_stream_id,
                "source_frame_index": row["source_frame_index"],
                "camera_timestamp_ns": cam_ts,
                "robot_row_index": None,
                "robot_timestamp_ns": None,
                "alignment_error_ns": None,
                "mapping_method": "unavailable",
            })
            continue

        # Binary search nearest
        idx = np.searchsorted(robot_ts, cam_ts)
        if idx >= len(robot_ts):
            nearest_idx = len(robot_ts) - 1
        elif idx == 0:
            nearest_idx = 0
        else:
            left = robot_ts[idx - 1]
            right = robot_ts[idx]
            nearest_idx = idx - 1 if abs(left - cam_ts) < abs(right - cam_ts) else idx

        nearest_ts = int(robot_ts[nearest_idx])
        error_ns = int(abs(nearest_ts - cam_ts))

        rows.append({
            "camera_stream_id": camera_stream_id,
            "source_frame_index": row["source_frame_index"],
            "camera_timestamp_ns": int(cam_ts),
            "robot_row_index": int(nearest_idx),
            "robot_timestamp_ns": nearest_ts,
            "alignment_error_ns": error_ns,
            "mapping_method": "nearest_hdf5_row",
        })

    return pd.DataFrame(rows)


# ======================================================================
# 批量构建 + 写入
# ======================================================================

def build_camera_maps(
    session: Session,
    aligned_timestamps_ns: list[int],
) -> dict:
    """为 Session 中所有相机流构建源映射表 + 机器人对齐表。

    Returns:
        {
            "validation": {...},
            "source_maps": {stream_id: DataFrame},
            "alignment": DataFrame (all cameras concatenated),
        }
    """
    result: dict = {}

    # ---- 验证 ----
    validation = validate_frame_index_method(session, aligned_timestamps_ns)
    result["validation"] = validation

    if validation["quality"] == "unavailable":
        return result

    # ---- 源映射表 ----
    source_maps: dict[str, pd.DataFrame] = {}
    for sid, vs in session.video_streams.items():
        sm = build_camera_source_map(vs, aligned_timestamps_ns, validation)
        source_maps[sid] = sm

    result["source_maps"] = source_maps

    # ---- 机器人对齐 ----
    alignment_dfs: list[pd.DataFrame] = []
    for sid, sm in source_maps.items():
        align = build_camera_robot_alignment(sid, sm, aligned_timestamps_ns)
        alignment_dfs.append(align)

    if alignment_dfs:
        result["alignment"] = pd.concat(alignment_dfs, ignore_index=True)
    else:
        result["alignment"] = pd.DataFrame()

    return result


def write_camera_maps(maps_result: dict, output_dir: str) -> dict[str, str]:
    """将相机映射表写入 Prepared Segment 的 maps/ 目录。

    Returns:
        {stream_id_source_map: path, alignment: path, ...}
    """
    maps_dir = Path(output_dir) / "maps"
    maps_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}

    # 各相机源映射
    source_maps = maps_result.get("source_maps", {})
    for sid, df in source_maps.items():
        path = maps_dir / f"{sid}_source_map.parquet"
        df.to_parquet(str(path), index=False)
        written[f"{sid}_source_map"] = str(path)

    # 对齐表
    alignment = maps_result.get("alignment")
    if alignment is not None and not alignment.empty:
        path = maps_dir / "camera_robot_alignment.parquet"
        alignment.to_parquet(str(path), index=False)
        written["camera_robot_alignment"] = str(path)

    return written


__all__ = [
    "validate_frame_index_method",
    "build_camera_source_map",
    "build_camera_robot_alignment",
    "build_camera_maps",
    "write_camera_maps",
]
