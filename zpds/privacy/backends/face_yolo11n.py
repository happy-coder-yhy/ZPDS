"""人脸检测后端 — YOLOv11n-face（ultralytics 栈）。

实现 ``FaceDetector`` Protocol，返回归一化 ``FaceDetection`` schema。
"""

from __future__ import annotations

import cv2
import numpy as np

from zpds.privacy import config as _cfg
from zpds.privacy.schemas import FaceDetection

_face_model = None


def get_face_model():
    """懒加载 YOLOv11n-face 模型（单例）。"""
    global _face_model
    if _face_model is None:
        if not _cfg.FACE_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"人脸模型不存在: {_cfg.FACE_MODEL_PATH}\n"
                "可通过环境变量 PRIVACY_FACE_MODEL 指定其它路径。"
            )
        from ultralytics import YOLO

        _face_model = YOLO(str(_cfg.FACE_MODEL_PATH))
    return _face_model


class YOLOFaceDetector:
    """YOLOv11n-face 人脸检测器，实现 ``FaceDetector`` Protocol。"""

    def __init__(self, confidence_threshold: float = _cfg.FACE_CONFIDENCE) -> None:
        self._confidence = confidence_threshold

    # ---- FaceDetector Protocol ----

    def detect(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        timestamp_ns: int,
    ) -> list[FaceDetection]:
        """检测单帧中的所有人脸。

        Returns:
            FaceDetection 列表（bbox 已归一化到 [0, 1]），无检测时为空。
        """
        model = get_face_model()
        h, w = frame_rgb.shape[:2]

        results = model(frame_rgb, conf=self._confidence, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        detections: list[FaceDetection] = []
        for box, conf in zip(boxes, confs):
            x1, y1, x2, y2 = box.astype(int)
            # clamp 到图像边界
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 <= x1 or y2 <= y1:
                continue
            # 归一化
            detections.append(FaceDetection(
                frame_index=frame_index,
                timestamp_ns=timestamp_ns,
                bbox_xyxy=(
                    float(x1) / w,
                    float(y1) / h,
                    float(x2) / w,
                    float(y2) / h,
                ),
                confidence=float(conf),
                backend="yolo11n_face",
            ))
        return detections

    def close(self) -> None:
        """释放模型资源。YOLO 模型无显式 close，这里做 no-op。"""

    # ---- 向后兼容：像素级 API（给旧的 redactor.py 使用） ----

    def detect_pixel_boxes(
        self,
        image: np.ndarray,
        confidence_threshold: float | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """返回像素级 (x1, y1, x2, y2) 列表。旧 API，新代码请用 detect()。"""
        conf = confidence_threshold if confidence_threshold is not None else self._confidence
        model = get_face_model()
        h, w = image.shape[:2]
        results = model(image, conf=conf, verbose=False)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        pixel_boxes: list[tuple[int, int, int, int]] = []
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 > x1 and y2 > y1:
                pixel_boxes.append((x1, y1, x2, y2))
        return pixel_boxes


# ---- 向后兼容函数（旧 face_detector.py API，新代码请用 YOLOFaceDetector） ----

_legacy_detector: YOLOFaceDetector | None = None


def _get_legacy() -> YOLOFaceDetector:
    global _legacy_detector
    if _legacy_detector is None:
        _legacy_detector = YOLOFaceDetector()
    return _legacy_detector


def blur_faces(image: np.ndarray, confidence_threshold: float = _cfg.FACE_CONFIDENCE) -> np.ndarray:
    """检测并模糊图像中的全部人脸（向后兼容旧 redactor.py）。"""
    detector = _get_legacy()
    # 需要旧 pixel box API 做模糊
    boxes = detector.detect_pixel_boxes(image, confidence_threshold)
    return _blur_regions(image, boxes)


def blur_regions(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strength: float = _cfg.BLUR_STRENGTH,
) -> np.ndarray:
    """向后兼容旧 API。"""
    return _blur_regions(image, boxes, strength)


def _blur_regions(
    image: np.ndarray,
    boxes: list[tuple[int, int, int, int]],
    strength: float = _cfg.BLUR_STRENGTH,
) -> np.ndarray:
    """对给定区域做自适应高斯模糊。"""
    result = image.copy()
    for x1, y1, x2, y2 in boxes:
        region_h, region_w = y2 - y1, x2 - x1
        ksize = max(3, int(min(region_h, region_w) * strength) | 1)
        region = result[y1:y2, x1:x2]
        result[y1:y2, x1:x2] = cv2.GaussianBlur(region, (ksize, ksize), 0)
    return result


__all__ = [
    "YOLOFaceDetector",
    "blur_faces",
    "blur_regions",
    "get_face_model",
]
