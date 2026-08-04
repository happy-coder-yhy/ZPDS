"""人脸检测与模糊(基于 YOLOv11n-face,同事推荐,来自 akanametov/yolo-face)。

与原项目(Visual-Privacy-protection)的 Caffe SSD 方案相比:
- 精度更高、接口与文本检测统一(都是 Ultralytics YOLO)
- 不依赖 OpenCV 的 Caffe 加载器(该功能在 OpenCV 5 中已移除)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

from . import config

_face_model = None


@dataclass
class FaceBox:
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float


def get_face_model():
    """懒加载 YOLOv11n-face 模型。"""
    global _face_model
    if _face_model is None:
        if not config.FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"人脸模型不存在: {config.FACE_MODEL_PATH}\n"
                "可通过环境变量 PRIVACY_FACE_MODEL 指定其它路径。\n"
                "下载地址: https://github.com/akanametov/yolo-face/releases"
            )
        from ultralytics import YOLO

        _face_model = YOLO(str(config.FACE_MODEL_PATH))
    return _face_model


def detect_faces(
    image: np.ndarray, confidence_threshold: float = config.FACE_CONFIDENCE
) -> list[FaceBox]:
    """检测图像中的人脸,返回置信度超过阈值的检测框。"""
    model = get_face_model()
    h, w = image.shape[:2]

    results = model(image, conf=confidence_threshold, verbose=False)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()

    faces: list[FaceBox] = []
    for box, conf in zip(boxes, confs):
        x1, y1, x2, y2 = box.astype(int)
        # clamp 到图像边界,防止越界
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 > x1 and y2 > y1:
            faces.append(FaceBox(bbox=(x1, y1, x2, y2), confidence=float(conf)))
    return faces


def blur_regions(
    image: np.ndarray,
    boxes: Sequence[tuple[int, int, int, int]],
    strength: float = config.BLUR_STRENGTH,
) -> np.ndarray:
    """对给定区域做自适应高斯模糊:核大小 = 区域短边 * strength,强制为奇数。

    :param image: BGR 图像数组
    :param boxes: (x1, y1, x2, y2) 坐标序列
    """
    result = image.copy()
    for x1, y1, x2, y2 in boxes:
        region_h, region_w = y2 - y1, x2 - x1
        ksize = max(3, int(min(region_h, region_w) * strength) | 1)
        region = result[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return result


def blur_faces(
    image: np.ndarray, confidence_threshold: float = config.FACE_CONFIDENCE
) -> np.ndarray:
    """检测并模糊图像中的全部人脸,返回处理后的图像。"""
    faces = detect_faces(image, confidence_threshold)
    return blur_regions(image, [f.bbox for f in faces])
