"""合成 fixture 工具：OpenCV 画人脸 + 文字，用于无真实数据的测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_test_frame(
    width: int = 640,
    height: int = 480,
    *,
    with_face: bool = True,
    with_text: bool = True,
    face_bbox: tuple[float, float, float, float] = (0.3, 0.2, 0.7, 0.8),
    text_bbox: tuple[float, float, float, float] = (0.1, 0.05, 0.9, 0.15),
    text_content: str = "姓名: 张三  电话: 13800138000",
    text_private: bool = True,
) -> np.ndarray:
    """合成一帧包含人脸和/或文本的测试图像。

    Args:
        width, height: 图像尺寸。
        with_face: 是否画模拟人脸。
        with_text: 是否画模拟文本。
        face_bbox: 归一化人脸位置。
        text_bbox: 归一化文本位置。
        text_content: 文本内容。
        text_private: True = 写隐私信息（可用于测试 LLM 分类），False = 写无害文本。

    Returns:
        BGR uint8 图像数组。
    """
    frame = np.random.randint(200, 240, (height, width, 3), dtype=np.uint8)

    if with_face:
        x1 = int(face_bbox[0] * width)
        y1 = int(face_bbox[1] * height)
        x2 = int(face_bbox[2] * width)
        y2 = int(face_bbox[3] * height)
        # 画椭圆模拟人脸
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        axes = ((x2 - x1) // 2, (y2 - y1) // 2)
        cv2.ellipse(frame, center, axes, 0, 0, 360, (180, 140, 120), -1)
        # 眼睛
        eye_y = center[1] - axes[1] // 4
        cv2.circle(frame, (center[0] - axes[0] // 3, eye_y), 5, (50, 50, 50), -1)
        cv2.circle(frame, (center[0] + axes[0] // 3, eye_y), 5, (50, 50, 50), -1)
        # 嘴
        mouth_y = center[1] + axes[1] // 3
        cv2.ellipse(frame, (center[0], mouth_y), (axes[0] // 3, axes[1] // 6),
                    0, 0, 180, (80, 60, 60), 2)

    if with_text:
        if not text_private:
            text_content = "产品: 机械臂关节模块 型号: ZPDS-A1"
        x1 = int(text_bbox[0] * width)
        y1 = int(text_bbox[1] * height)
        x2 = int(text_bbox[2] * width)
        y2 = int(text_bbox[3] * height)
        # 白色背景
        cv2.rectangle(frame, (x1, y1), (x2, y2), (240, 240, 240), -1)
        # 文字
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = (x2 - x1) / 350
        cv2.putText(frame, text_content, (x1 + 10, y1 + int((y2 - y1) * 0.7)),
                    font, font_scale, (0, 0, 0), 1)

    return frame


def create_test_video(
    output_path: str | Path,
    num_frames: int = 30,
    fps: float = 10.0,
    width: int = 640,
    height: int = 480,
    *,
    with_face: bool = True,
    with_text: bool = True,
    text_private: bool = True,
) -> Path:
    """合成一段测试视频（MP4）。

    Returns:
        视频文件路径。
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # 人脸轻微移动
        dx = np.sin(i * 0.3) * 0.02
        face_bbox = (0.3 + dx, 0.2, 0.7 + dx, 0.8)
        frame = create_test_frame(
            width, height,
            with_face=with_face,
            with_text=with_text,
            face_bbox=face_bbox,
            text_private=text_private,
        )
        writer.write(frame)

    writer.release()
    return output_path


__all__ = ["create_test_frame", "create_test_video"]
