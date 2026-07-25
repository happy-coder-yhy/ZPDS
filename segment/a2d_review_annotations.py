"""
A2D Review 动作标注转换器。

将 review_*.json 中的 startFrame/endFrame（aligned_joints.h5 样本索引）
转换为 Prepared Segment 的 output_frame_index，写出 annotations/review_actions.parquet。

帧编号体系确认：
  startFrame/endFrame = aligned_joints.h5 的 0-based 样本索引。
  → 通过 aligned timestamp 桥接 → sample_map → output_frame_index。

用法:
    from segment.a2d_review_annotations import convert_review_actions

    df = convert_review_actions(
        review_path="E:/datasets/真机/A2D/review/review_8032.json",
        aligned_timestamps_ns=[...],
        segment_start_ns=...,
        segment_end_ns=...,
        rgb_sample_map_path="seg_000001/maps/head_rgb_sample_map.parquet",
    )
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

def convert_review_actions(
    review_path: str,
    aligned_timestamps_ns: list[int],
    segment_start_ns: int,
    segment_end_ns: int,
    rgb_sample_map_path: str,
) -> pd.DataFrame:
    """将 review 动作标注转换为 Segment 输出帧索引。

    Args:
        review_path: review_*.json 文件路径。
        aligned_timestamps_ns: aligned_joints.h5 的 timestamp 数组。
        segment_start_ns: Segment 源起始时间。
        segment_end_ns: Segment 源结束时间。
        rgb_sample_map_path: head_rgb 的 sample_map.parquet 路径
                             （用于 output_frame_index 映射）。

    Returns:
        DataFrame，列:
          - start_source_frame_index: aligned 样本索引
          - end_source_frame_index:   aligned 样本索引
          - start_timestamp_ns
          - end_timestamp_ns
          - start_output_frame_index: head_rgb CFR 输出帧索引
          - end_output_frame_index:   head_rgb CFR 输出帧索引
          - action_text
          - skill
          - annotation_source
    """
    # ---- 1. 加载 review actions ----
    actions = _load_review_actions(review_path)
    if not actions:
        return pd.DataFrame()

    # ---- 2. 加载 sample_map（用于 timestamp → output_frame 映射） ----
    # review 使用绝对设备时间戳 → 需转为 Segment 相对时间
    # sample_map.output_timestamp_ns 是相对时间，source_timestamp_ns 是绝对时间
    sm_path = Path(rgb_sample_map_path)
    if sm_path.is_file():
        sample_map = pd.read_parquet(str(sm_path))
        # 用 output_timestamp_ns（相对）+ segment_start_ns 来匹配绝对时间
        sm_rel = sample_map["output_timestamp_ns"].values.astype(np.int64)
        sm_frame_indices = sample_map["output_frame_index"].values
    else:
        sample_map = None
        sm_rel = np.array([], dtype=np.int64)
        sm_frame_indices = np.array([])

    aligned_ts = np.array(aligned_timestamps_ns, dtype=np.int64)

    # ---- 3. 逐 action 转换 ----
    rows = []
    review_filename = Path(review_path).name

    for action in actions:
        sf = action["startFrame"]
        ef = action["endFrame"]

        # 验证帧索引有效
        if sf < 0 or ef >= len(aligned_ts):
            continue
        if sf >= ef:
            continue

        start_ts = int(aligned_ts[sf])
        end_ts = int(aligned_ts[ef])

        # 检查与 Segment 时间范围有交集
        if end_ts < segment_start_ns or start_ts > segment_end_ns:
            continue  # 动作不在当前 Segment 内

        # 映射到 output_frame_index（绝对 → 相对时间）
        start_rel = start_ts - segment_start_ns
        end_rel = end_ts - segment_start_ns
        start_out = _timestamp_to_output_frame(
            start_rel, sm_rel, sm_frame_indices
        )
        end_out = _timestamp_to_output_frame(
            end_rel, sm_rel, sm_frame_indices
        )

        rows.append({
            "start_source_frame_index": sf,
            "end_source_frame_index": ef,
            "start_timestamp_ns": start_ts,
            "end_timestamp_ns": end_ts,
            "start_output_frame_index": start_out,
            "end_output_frame_index": end_out,
            "action_text": action.get("actionText", "").strip(),
            "skill": action.get("skill", ""),
            "annotation_source": review_filename,
        })

    return pd.DataFrame(rows)


def _load_review_actions(review_path: str) -> list[dict]:
    """从 review JSON 中提取 action_config 列表。

    action_config 可能是：
      - 直接 list[dict]
      - JSON 字符串编码的 list[dict]
    """
    review_file = Path(review_path)
    if not review_file.is_file():
        return []

    with open(review_file, "r", encoding="utf-8") as f:
        review = json.load(f)

    action_config = review.get("label_info", {}).get("action_config", [])
    if isinstance(action_config, str):
        try:
            action_config = json.loads(action_config)
        except json.JSONDecodeError:
            return []
    if not isinstance(action_config, list):
        return []

    return action_config


def _timestamp_to_output_frame(
    target_ns: int,
    sm_timestamps: np.ndarray,
    sm_frame_indices: np.ndarray,
) -> int | None:
    """通过最近邻查找将源时间戳映射到 CFR 输出帧索引。

    sm_timestamps 是 output_timestamp_ns（相对 Segment 时间）。
    但 target_ns 是绝对设备时间戳。
    此函数假设调用方已预先将 target_ns 转为相对时间，
    或者 sm_timestamps 包含 source_timestamp_ns（绝对时间）。

    实际使用：传入 source_timestamp_ns 列进行匹配。
    """
    if len(sm_timestamps) == 0:
        return None
    idx = int(np.argmin(np.abs(sm_timestamps.astype(np.int64) - target_ns)))
    return int(sm_frame_indices[idx])


# ======================================================================
# 批量写出（按 Segment）
# ======================================================================

def write_review_annotations(
    df: pd.DataFrame,
    segment_dir: str,
) -> str:
    """写出 annotations/review_actions.parquet。

    Returns:
        输出文件路径。
    """
    ann_dir = Path(segment_dir) / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    out_path = ann_dir / "review_actions.parquet"
    df.to_parquet(str(out_path), index=False)
    return str(out_path)


# ======================================================================
# 标注流 segment.json 描述
# ======================================================================

def build_annotation_stream_entry(
    parquet_uri: str = "annotations/review_actions.parquet",
    sample_map_uri: str = "maps/head_rgb_sample_map.parquet",
) -> dict:
    """构建 review_actions 标注流的 segment.json stream entry。"""
    return {
        "stream_id": "review_actions",
        "role": "annotation",
        "modality": "action_label",
        "uri": parquet_uri,
        "format": "parquet",
        "ground_truth_status": "human_reviewed",
        "time": {
            "clock_id": "segment",
            "sampling": "sparse",
            "timestamp_column": "start_timestamp_ns",
        },
        "fields": [
            {"name": "action_text", "dtype": "str"},
            {"name": "skill", "dtype": "str"},
        ],
        "origin": {
            "kind": "imported_human_annotation",
            "source_asset_id": "review_json",
            "operation": "frame_index_to_output_frame",
            "sample_map_uri": sample_map_uri,
        },
    }


__all__ = [
    "convert_review_actions",
    "write_review_annotations",
    "build_annotation_stream_entry",
]
