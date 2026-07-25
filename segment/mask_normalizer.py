"""
Mask 标注标准化 — Pickle RLE → 紧凑 COCO RLE → Parquet。

将 EPIC-KITCHENS-100 Mask Pickle 中的原始 RLE bytes 解码为二值掩码，
再用 pycocotools 编码为标准 COCO RLE，完成帧映射和 Segment 裁剪。

用法:
    from segment.mask_normalizer import normalize_masks, write_mask_parquet

    df = normalize_masks(
        annotation_stream=session.annotation_streams["masks"],
        video_timestamps_ns=session.primary_video.timestamps_ns,
        sample_map=pd.read_parquet("maps/ego_rgb_sample_map.parquet"),
        source_start_ns=span_start,
        source_end_ns=span_end,
        video_width=456,
        video_height=256,
    )
    write_mask_parquet(df, "prepared_segments/seg_000001", "instance_masks")
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segment.annotation_normalizer import (
    frame_to_timestamp,
    nearest_output_frame,
    clip_bbox,
)


# ---- COCO RLE 编解码 ----

def _get_mask_utils():
    """延迟加载 pycocotools，未安装时给出明确错误。"""
    try:
        from pycocotools import mask as mask_utils
        return mask_utils
    except ImportError:
        raise ImportError(
            "mask_normalizer 需要 pycocotools。安装: pip install pycocotools"
        )


def decode_rle(
    rle_bytes: bytes,
    height: int,
    width: int,
) -> np.ndarray:
    """将 COCO RLE counts bytes 解码为二值掩码 (H×W uint8)。

    Args:
        rle_bytes: 原始 COCO RLE counts (bytes)
        height: 掩码高度
        width: 掩码宽度

    Returns:
        H×W uint8 数组 (0/1)
    """
    mask_utils = _get_mask_utils()
    rle = {"size": [height, width], "counts": rle_bytes}
    return mask_utils.decode(rle).astype(np.uint8)


def encode_binary_mask(mask: np.ndarray) -> dict:
    """将二值掩码编码为 COCO RLE 字典。

    Args:
        mask: H×W uint8 或 bool 数组

    Returns:
        {"size": [height, width], "counts": str}
    """
    mask_utils = _get_mask_utils()
    mask_arr = np.asarray(mask, dtype=np.uint8)
    encoded = mask_utils.encode(np.asfortranarray(mask_arr))

    counts = encoded["counts"]
    if isinstance(counts, bytes):
        counts = counts.decode("ascii")

    return {
        "size": [int(encoded["size"][0]), int(encoded["size"][1])],
        "counts": counts,
    }


# ---- 掩码面积 ----

def rle_area(rle: dict) -> float:
    """计算 COCO RLE 掩码的面积（像素数）。"""
    mask_utils = _get_mask_utils()
    return float(mask_utils.area(rle))


def bbox_area(x1: float, y1: float, x2: float, y2: float) -> float:
    """计算 BBox 面积。"""
    return max(0.0, (x2 - x1) * (y2 - y1))


# ---- 主标准化函数 ----

def normalize_masks(
    annotation_stream,          # AnnotationStream
    video_timestamps_ns: list[int],
    sample_map: pd.DataFrame,
    source_start_ns: int,
    source_end_ns: int,
    video_width: int,
    video_height: int,
) -> pd.DataFrame:
    """将 Mask 标注标准化为 Parquet-ready DataFrame。

    每条 Pickle 记录 → 1 行 DataFrame:
      - 解码原始 RLE bytes → 二值掩码
      - 重新编码为标准 COCO RLE
      - 归一化 bbox 转像素坐标
      - Segment 区间裁剪 + 输出帧映射

    Args:
        annotation_stream: 包含 instance_segmentation 记录的 AnnotationStream
        video_timestamps_ns: 原视频帧时间戳列表 (0-based 索引)
        sample_map: 输出帧 ↔ 源帧映射表
        source_start_ns: Segment 源起始时间 (ns)
        source_end_ns: Segment 源结束时间 (ns)
        video_width: 视频帧宽 (像素)
        video_height: 视频帧高 (像素)

    Returns:
        DataFrame，字段:
          timestamp_ns, output_frame_index,
          source_timestamp_ns, source_frame_index,
          instance_id,
          category_id, category_name, score,
          bbox_x1, bbox_y1, bbox_x2, bbox_y2,
          mask_height, mask_width,
          rle_counts, rle_encoding,
          mask_area_px,
          source_file
    """
    records = annotation_stream.records
    source_file = str(annotation_stream.source_path)

    if not records:
        return pd.DataFrame()

    sm_ts = sample_map["source_timestamp_ns"].values.astype(np.int64)
    mask_utils = _get_mask_utils()

    rows: list[dict] = []

    for rec_idx, rec in enumerate(records):
        frame_idx = rec["frame_index"]

        # ---- 源时间戳 ----
        try:
            src_ts = frame_to_timestamp(frame_idx, video_timestamps_ns)
        except IndexError:
            continue

        # ---- Segment 裁剪 ----
        if src_ts < source_start_ns or src_ts >= source_end_ns:
            continue

        # ---- 输出帧映射 ----
        out_frame_idx = nearest_output_frame(src_ts, sm_ts)
        out_ts = int(sample_map.iloc[out_frame_idx]["output_timestamp_ns"])

        # ---- RLE 解码 ----
        rle_bytes = rec.get("mask_rle_bytes", b"")
        if not rle_bytes:
            continue

        try:
            rle_raw = {"size": [video_height, video_width], "counts": rle_bytes}
            binary_mask = mask_utils.decode(rle_raw)
        except Exception:
            # RLE 解码失败 → 跳过该记录
            continue

        # ---- 编码为标准 COCO RLE ----
        try:
            rle_encoded = mask_utils.encode(np.asfortranarray(binary_mask.astype(np.uint8)))
            rle_counts = rle_encoded["counts"]
            if isinstance(rle_counts, bytes):
                rle_counts = rle_counts.decode("ascii")
        except Exception:
            continue

        mask_h = int(rle_encoded["size"][0])
        mask_w = int(rle_encoded["size"][1])
        area = float(mask_utils.area(rle_encoded))

        # ---- BBox 归一化 → 像素 ----
        bbox_norm = rec["mask_bbox"]
        bx1 = bbox_norm[0] * video_width
        by1 = bbox_norm[1] * video_height
        bx2 = bbox_norm[2] * video_width
        by2 = bbox_norm[3] * video_height

        bx1, by1, bx2, by2, was_clipped = clip_bbox(
            bx1, by1, bx2, by2, video_width, video_height
        )

        # ---- 类别 ----
        category_id = int(rec.get("mask_class", 0))
        category_name = _coco_category_name(category_id)

        rows.append({
            "timestamp_ns": out_ts,
            "output_frame_index": out_frame_idx,
            "source_timestamp_ns": src_ts,
            "source_frame_index": frame_idx,
            "instance_id": f"mask_{rec_idx}",
            "category_id": category_id,
            "category_name": category_name,
            "score": float(rec.get("mask_score", 0.0)),
            "bbox_x1": bx1,
            "bbox_y1": by1,
            "bbox_x2": bx2,
            "bbox_y2": by2,
            "bbox_was_clipped": was_clipped,
            "mask_height": mask_h,
            "mask_width": mask_w,
            "rle_counts": rle_counts,
            "rle_encoding": "coco_rle",
            "mask_area_px": area,
            "source_file": source_file,
        })

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    column_order = [
        "timestamp_ns", "output_frame_index",
        "source_timestamp_ns", "source_frame_index",
        "instance_id",
        "category_id", "category_name", "score",
        "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
        "bbox_was_clipped",
        "mask_height", "mask_width",
        "rle_counts", "rle_encoding",
        "mask_area_px",
        "source_file",
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    extra_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + extra_cols]

    return df


# ---- 写入 ----

def write_mask_parquet(
    df: pd.DataFrame,
    output_dir: str,
    stream_id: str = "instance_masks",
) -> str:
    """将标准化 Mask DataFrame 写出为 Parquet。

    Args:
        df: normalize_masks() 返回的 DataFrame
        output_dir: Prepared Segment 根目录
        stream_id: 流标识（用于文件命名）

    Returns:
        输出文件路径
    """
    annotations_dir = Path(output_dir) / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    output_path = annotations_dir / f"{stream_id}.parquet"
    df.to_parquet(str(output_path), index=False)
    return str(output_path)


# ---- COCO 类别名映射 ----

_COCO_CLASSES: dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane",
    6: "bus", 7: "train", 8: "truck", 9: "boat", 10: "traffic light",
    11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep",
    21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard",
    37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket",
    44: "bottle", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange",
    56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza",
    60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed",
    67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote",
    76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink",
    82: "refrigerator", 84: "book", 85: "clock", 86: "vase",
    87: "scissors", 88: "teddy bear", 89: "hair drier",
    90: "toothbrush",
    # EPIC-KITCHENS 常见补充
    47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
}


def _coco_category_name(category_id: int) -> str:
    return _COCO_CLASSES.get(category_id, f"class_{category_id}")


__all__ = [
    "decode_rle",
    "encode_binary_mask",
    "rle_area",
    "bbox_area",
    "normalize_masks",
    "write_mask_parquet",
]
