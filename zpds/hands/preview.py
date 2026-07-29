"""
Hands Preview Video Generator。

读取 Prepared Segment MP4 + hands_2d.parquet，
在每帧上叠加手部 BBox、21 个关键点、关键点连线和左右手标签，
输出可视化预览视频。

叠加内容:
  - 手部 BBox 矩形
  - 21 个关键点圆点
  - 关键点连线 (MediaPipe Hand Landmarks 拓扑)
  - Left / Right 标签 + 置信度
  - 输出帧号 + 时间戳
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zpds.hands.writer import HAND_CONNECTIONS

# ---- 颜色定义 ----
COLOR_LEFT = (0, 120, 255)       # 橙色 (BGR) — 左手
COLOR_RIGHT = (255, 60, 60)      # 蓝色 (BGR) — 右手
COLOR_BBOX = (0, 255, 0)         # 绿色 — BBox
COLOR_TEXT = (255, 255, 255)     # 白色 — 文字
COLOR_PANEL = (40, 40, 40)       # 深灰 — 信息栏背景
COLOR_KP_CENTER = (0, 0, 255)    # 红色 — 掌心/手腕
COLOR_KP_TIP = (0, 255, 255)     # 黄色 — 指尖


# 指尖 landmark 索引
FINGERTIP_IDS = {4, 8, 12, 16, 20}
WRIST_ID = 0


def _get_hand_color(handedness: str) -> tuple[int, int, int]:
    if str(handedness).lower() == "left":
        return COLOR_LEFT
    return COLOR_RIGHT


def _is_normalized_points(kp: np.ndarray) -> bool:
    """Return True when keypoints look like MediaPipe normalized xy coords."""
    if kp.size == 0 or not np.isfinite(kp).all():
        return False
    xs = kp[:, 0]
    ys = kp[:, 1]
    return (
        float(np.nanmin(xs)) >= -0.05
        and float(np.nanmax(xs)) <= 1.05
        and float(np.nanmin(ys)) >= -0.05
        and float(np.nanmax(ys)) <= 1.05
    )


def _is_normalized_bbox(bbox: tuple[float, float, float, float]) -> bool:
    """Return True when bbox looks like normalized xyxy coords."""
    values = np.array(bbox, dtype=np.float32)
    if not np.isfinite(values).all():
        return False
    return float(values.min()) >= -0.05 and float(values.max()) <= 1.05


def _to_pixel_keypoints(kp_raw, frame_w: int, frame_h: int) -> np.ndarray:
    """Convert serialized keypoints to a 21x2 pixel-coordinate array.

    The V1 schema says `keypoints_2d` should be pixels, but early A/B integration
    builds sometimes wrote MediaPipe normalized coordinates. Preview accepts both
    so the visualization remains useful while Validator reports suspicious data.
    """
    if isinstance(kp_raw, np.ndarray):
        kp_array = np.array(kp_raw.tolist(), dtype=np.float32)
    else:
        kp_array = np.array(list(kp_raw), dtype=np.float32)

    if kp_array.shape != (21, 2):
        raise ValueError(f"keypoints_2d must be 21x2, got {kp_array.shape}")

    if _is_normalized_points(kp_array):
        kp_array = kp_array.copy()
        kp_array[:, 0] *= frame_w
        kp_array[:, 1] *= frame_h

    return kp_array


def _to_pixel_bbox(
    bbox: tuple[float, float, float, float],
    frame_w: int,
    frame_h: int,
) -> tuple[float, float, float, float]:
    """Convert a serialized bbox to pixel xyxy coordinates."""
    if _is_normalized_bbox(bbox):
        x1, y1, x2, y2 = bbox
        return (x1 * frame_w, y1 * frame_h, x2 * frame_w, y2 * frame_h)
    return bbox


def _draw_single_hand(
    frame: np.ndarray,
    kp: np.ndarray,       # (21, 2) pixel coords
    bbox: tuple[float, float, float, float],
    handedness: str,
    score: float,
    frame_h: int,
    frame_w: int,
):
    """在帧上绘制单只手的覆层。"""
    color = _get_hand_color(handedness)

    # --- BBox ---
    bx1, by1, bx2, by2 = [
        int(max(0, min(v, s - 1)))
        for v, s in zip(bbox, [frame_w, frame_h, frame_w, frame_h])
    ]
    cv2.rectangle(frame, (bx1, by1), (bx2, by2), COLOR_BBOX, 2)

    # --- 关键点连线 ---
    for i, j in HAND_CONNECTIONS:
        if i < len(kp) and j < len(kp):
            pt1 = (int(kp[i][0]), int(kp[i][1]))
            pt2 = (int(kp[j][0]), int(kp[j][1]))
            if (0 <= pt1[0] < frame_w and 0 <= pt1[1] < frame_h and
                    0 <= pt2[0] < frame_w and 0 <= pt2[1] < frame_h):
                cv2.line(frame, pt1, pt2, color, 1, cv2.LINE_AA)

    # --- 关键点圆点 ---
    for idx, (x, y) in enumerate(kp):
        px, py = int(x), int(y)
        if 0 <= px < frame_w and 0 <= py < frame_h:
            if idx == WRIST_ID:
                cv2.circle(frame, (px, py), 4, COLOR_KP_CENTER, -1, cv2.LINE_AA)
            elif idx in FINGERTIP_IDS:
                cv2.circle(frame, (px, py), 3, COLOR_KP_TIP, -1, cv2.LINE_AA)
            else:
                cv2.circle(frame, (px, py), 2, color, -1, cv2.LINE_AA)

    # --- 标签 ---
    label = f"{handedness}  {score:.2f}"
    tx, ty = bx1, max(by1 - 6, 18)
    cv2.putText(frame, label, (tx, ty),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)


def generate_hands_preview(
    segment_dir: str,
    hands_parquet_path: str,
    output_path: str | None = None,
    video_stream_id: str | None = None,
    target_fps: float | None = None,
) -> str:
    """生成手部检测预览视频。

    Args:
        segment_dir: Prepared Segment 目录 (包含 segment.json)
        hands_parquet_path: hands_2d.parquet 文件路径
        output_path: 输出 MP4 路径，默认 output/hand_preview/<segment>_hands_preview.mp4
        video_stream_id: 视频流 ID，默认自动选择第一个 rgb 流
        target_fps: 输出帧率，默认使用源视频帧率

    Returns:
        输出视频文件路径
    """
    seg_dir = Path(segment_dir)

    # ---- 读取 segment.json ----
    seg_path = seg_dir / "segment.json"
    if not seg_path.exists():
        raise FileNotFoundError(f"segment.json not found: {seg_path}")
    with open(seg_path) as f:
        segment = json.load(f)

    # ---- 找到视频流 ----
    video_streams = [s for s in segment["streams"] if s.get("format") == "mp4"]
    if not video_streams:
        raise ValueError("No MP4 streams in segment.json")

    if video_stream_id is not None:
        vs = next((s for s in video_streams if s["stream_id"] == video_stream_id), None)
        if vs is None:
            raise ValueError(f"Stream {video_stream_id} not found")
    else:
        vs = video_streams[0]

    selected_stream_id = vs["stream_id"]
    video_path = str(seg_dir / vs["uri"])
    if not Path(video_path).exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    # ---- 读取 Parquet ----
    if not Path(hands_parquet_path).exists():
        raise FileNotFoundError(f"Parquet not found: {hands_parquet_path}")
    df = pd.read_parquet(hands_parquet_path)

    # ---- 打开视频 ----
    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 如果存在多流，必须按实际预览的视频流过滤。否则别的相机/视频流检测会叠到当前画面。
    if "video_stream_id" in df.columns:
        df = df[df["video_stream_id"] == selected_stream_id]

    # 按帧分组
    by_frame = {}
    for _, row in df.iterrows():
        fi = int(row["output_frame_index"])
        if fi not in by_frame:
            by_frame[fi] = []
        kp_array = _to_pixel_keypoints(row["keypoints_2d"], frame_w, frame_h)
        raw_bbox = (
            float(row["bbox_x1"]),
            float(row["bbox_y1"]),
            float(row["bbox_x2"]),
            float(row["bbox_y2"]),
        )
        bbox = _to_pixel_bbox(raw_bbox, frame_w, frame_h)
        by_frame[fi].append({
            "handedness": row["handedness"],
            "score": float(row["handedness_score"]),
            "keypoints": kp_array,
            "bbox": bbox,
        })

    if target_fps is None:
        target_fps = video_fps if video_fps > 0 else 30.0

    # ---- 输出路径 ----
    if output_path is None:
        preview_dir = Path("output") / "hand_preview"
        preview_dir.mkdir(parents=True, exist_ok=True)
        seg_name = seg_dir.name
        output_path = str(preview_dir / f"{seg_name}_hands_preview.mp4")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (frame_w, frame_h))
    if not writer.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (frame_w, frame_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Cannot create output video writer")

    # ---- 逐帧处理 ----
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 叠加信息栏
        overlay = frame.copy()

        # 底部信息栏
        info_h = 28
        cv2.rectangle(overlay, (0, frame_h - info_h,),
                      (frame_w, frame_h), COLOR_PANEL, -1)
        info_text = (
            f"Frame: {frame_idx}/{total_frames}  |  "
            f"Hands: {len(by_frame.get(frame_idx, []))}  |  "
            f"Stream: {selected_stream_id}"
        )
        cv2.putText(overlay, info_text, (6, frame_h - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, COLOR_TEXT, 1, cv2.LINE_AA)

        # 绘制手部覆层
        if frame_idx in by_frame:
            for hand_info in by_frame[frame_idx]:
                _draw_single_hand(
                    overlay, hand_info["keypoints"], hand_info["bbox"],
                    hand_info["handedness"], hand_info["score"],
                    frame_h, frame_w,
                )

        # 半透明混合
        alpha = 0.85
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    return str(Path(output_path).resolve())
