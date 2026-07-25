"""
标注预览图生成器：在输出帧上叠加 BBox、置信度、交互连接线。

用法:
    from segment.preview_generator import generate_previews

    generate_previews(
        seg_dir="prepared_segments/epic/P01_01/seg_000001",
        annotation_uri="annotations/hand_objects.parquet",
        video_uri="data/ego_rgb.mp4",
        sample_frames=[100, 250, 500],
        output_dir="reports/previews",
    )
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


# ---- 颜色常量 ----

HAND_COLOR = (0, 255, 0)       # 绿色 — 手
OBJECT_COLOR = (255, 0, 0)     # 蓝色 — 物体
LINK_COLOR = (0, 255, 255)     # 黄色 — 交互连接线
TEXT_COLOR = (255, 255, 255)   # 白色 — 文字
TEXT_BG = (0, 0, 0)            # 黑色 — 文字背景


# ---- 主函数 ----

def generate_previews(
    seg_dir: str,
    annotation_uri: str,
    video_uri: str,
    sample_frames: list[int],
    output_dir: str,
    *,
    hand_color: tuple[int, int, int] = HAND_COLOR,
    object_color: tuple[int, int, int] = OBJECT_COLOR,
    link_color: tuple[int, int, int] = LINK_COLOR,
    line_thickness: int = 2,
    font_scale: float = 0.5,
) -> list[str]:
    """在指定输出帧上叠加标注可视化，生成预览 JPG。

    Args:
        seg_dir: Prepared Segment 根目录
        annotation_uri: 标注 Parquet 相对路径 (如 "annotations/hand_objects.parquet")
        video_uri: 视频相对路径 (如 "data/ego_rgb.mp4")
        sample_frames: 要生成预览的输出帧索引列表
        output_dir: 输出目录
        hand_color: 手 BBox 颜色 (BGR)
        object_color: 物体 BBox 颜色 (BGR)
        link_color: 交互连接线颜色 (BGR)
        line_thickness: 线条粗细
        font_scale: 文字比例

    Returns:
        生成的 JPG 文件路径列表
    """
    seg_path = Path(seg_dir)
    ann_path = seg_path / annotation_uri
    video_path = seg_path / video_uri
    preview_dir = Path(output_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    # ---- 加载标注 ----
    if not ann_path.exists():
        raise FileNotFoundError(f"标注文件不存在: {ann_path}")

    df = pd.read_parquet(str(ann_path))

    # 按 output_frame_index 分组
    groups = df.groupby("output_frame_index")

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"无法打开视频: {video_path}")

    output_paths: list[str] = []

    try:
        for out_frame_idx in sample_frames:
            # 定位到指定帧
            cap.set(cv2.CAP_PROP_POS_FRAMES, out_frame_idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                print(f"  ⚠ 帧 {out_frame_idx} 不可解码，跳过")
                continue

            # 获取该帧的标注
            if out_frame_idx not in groups.groups:
                print(f"  ⚠ 帧 {out_frame_idx} 无标注数据，跳过")
                continue

            frame_anns = groups.get_group(out_frame_idx)
            frame = _draw_annotations(
                frame, frame_anns, out_frame_idx,
                hand_color, object_color, link_color,
                line_thickness, font_scale,
            )

            # 写出
            out_name = f"frame_{out_frame_idx:06d}.jpg"
            out_path = preview_dir / out_name
            cv2.imwrite(str(out_path), frame)
            output_paths.append(str(out_path))

    finally:
        cap.release()

    return output_paths


# ---- 绘制逻辑 ----

def _draw_annotations(
    frame: np.ndarray,
    frame_anns: pd.DataFrame,
    output_frame_idx: int,
    hand_color: tuple[int, int, int],
    object_color: tuple[int, int, int],
    link_color: tuple[int, int, int],
    line_thickness: int,
    font_scale: float,
) -> np.ndarray:
    """在单帧上绘制所有标注。"""
    h, w = frame.shape[:2]

    # 分离 hand 和 object
    hands = frame_anns[frame_anns["entity_type"] == "hand"]
    objects = frame_anns[frame_anns["entity_type"] == "object"]

    # 画物体 BBox（先画，在底层）
    for _, obj in objects.iterrows():
        _draw_bbox(frame, obj, object_color, line_thickness, font_scale, "obj")

    # 画手 BBox
    for _, hand in hands.iterrows():
        _draw_bbox(frame, hand, hand_color, line_thickness, font_scale, "hand")

    # 画交互连接线 (hand ↔ object 视线)
    obj_by_id = {}
    for _, obj in objects.iterrows():
        obj_by_id[obj.get("entity_id", "")] = obj

    for _, hand in hands.iterrows():
        linked = hand.get("linked_entity_id")
        if linked and linked in obj_by_id:
            obj = obj_by_id[linked]
            hx, hy = _bbox_center(hand)
            ox, oy = _bbox_center(obj)
            cv2.line(
                frame,
                (int(hx), int(hy)),
                (int(ox), int(oy)),
                link_color,
                thickness=1,
                lineType=cv2.LINE_AA,
            )

    # 帧信息面板
    _draw_info_panel(frame, frame_anns, output_frame_idx, font_scale, w)

    return frame


def _draw_bbox(
    frame: np.ndarray,
    row: pd.Series,
    color: tuple[int, int, int],
    thickness: int,
    font_scale: float,
    label_prefix: str,
) -> None:
    """绘制单个 BBox + 标签。"""
    x1 = int(row.get("bbox_x1", 0))
    y1 = int(row.get("bbox_y1", 0))
    x2 = int(row.get("bbox_x2", 0))
    y2 = int(row.get("bbox_y2", 0))

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # 标签文本
    conf = row.get("confidence", 0.0)
    entity_id = row.get("entity_id", "?")
    label = f"{label_prefix}:{entity_id.split('_')[-1]} {conf:.2f}"

    _put_text_with_bg(frame, label, (x1, y1 - 6), color, font_scale)


def _bbox_center(row: pd.Series) -> tuple[float, float]:
    """计算 BBox 中心点。"""
    x1 = row.get("bbox_x1", 0)
    y1 = row.get("bbox_y1", 0)
    x2 = row.get("bbox_x2", 0)
    y2 = row.get("bbox_y2", 0)
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _put_text_with_bg(
    frame: np.ndarray,
    text: str,
    position: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
) -> None:
    """绘制带背景的文本。"""
    x, y = position
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
    )
    # 确保不超出画面上边界
    if y - th - 4 < 0:
        y = y + th + 16
        bg_y1 = y - th - 4
        bg_y2 = y + baseline + 2
    else:
        bg_y1 = y - th - 4
        bg_y2 = y + baseline + 2

    cv2.rectangle(
        frame,
        (x, bg_y1),
        (x + tw + 4, bg_y2),
        TEXT_BG,
        -1,
    )
    cv2.putText(
        frame, text,
        (x + 2, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1, cv2.LINE_AA,
    )


def _draw_info_panel(
    frame: np.ndarray,
    frame_anns: pd.DataFrame,
    output_frame_idx: int,
    font_scale: float,
    frame_width: int,
) -> None:
    """绘制左上角信息面板。"""
    first = frame_anns.iloc[0]
    src_frame = first.get("source_frame_index", "?")
    ts_ns = first.get("timestamp_ns", 0)
    ts_s = ts_ns / 1e9 if ts_ns else 0
    n_hands = int((frame_anns["entity_type"] == "hand").sum())
    n_objs = int((frame_anns["entity_type"] == "object").sum())

    lines = [
        f"output_frame: {output_frame_idx}",
        f"source_frame: {src_frame}",
        f"timestamp: {ts_s:.3f}s",
        f"hands: {n_hands}  objects: {n_objs}",
    ]

    panel_height = len(lines) * 18 + 10
    panel_width = 320

    # 半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (5, 5), (panel_width, panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    for i, line in enumerate(lines):
        y = 22 + i * 18
        cv2.putText(
            frame, line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, TEXT_COLOR, 1, cv2.LINE_AA,
        )


__all__ = ["generate_previews"]
