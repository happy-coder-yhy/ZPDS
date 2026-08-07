"""文本检测与 OCR 后端 — EasyOCR + 可选 YOLO 区域提议。

实现 ``TextDetector`` Protocol，返回归一化 ``TextDetection`` schema。
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from zpds.privacy import config as _cfg
from zpds.privacy.schemas import TextDetection

# 模型/OCR 单例
_yolo_model: Optional[object] = None  # ultralytics YOLO
_ocr_reader = None                    # easyocr.Reader


def get_yolo():
    """懒加载 YOLO 文本检测模型（可选）。"""
    global _yolo_model
    if _yolo_model is None:
        yolo_path = Path(str(_cfg.YOLO_MODEL_PATH))
        if not yolo_path.exists():
            raise FileNotFoundError(
                f"YOLO 文本模型不存在: {yolo_path}\n"
                "可通过环境变量 PRIVACY_YOLO_PATH 指定其它路径。"
            )
        from ultralytics import YOLO
        _yolo_model = YOLO(str(yolo_path))
    return _yolo_model


def get_ocr_reader():
    """懒加载 EasyOCR reader。"""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(_cfg.OCR_LANGS)
    return _ocr_reader


def _preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """灰度 + 锐化，提升 OCR 识别率。"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    sharpen_kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(gray, -1, sharpen_kernel)


def _zoomed_crop(
    image: np.ndarray,
    x1: float, y1: float, x2: float, y2: float,
    zoom: float = 1.5,
) -> tuple[int, int, int, int]:
    """以框中心放大裁剪，并 clamp 到图像边界。"""
    h, w = image.shape[:2]
    cx, cy = int((x1 + x2) // 2), int((y1 + y2) // 2)
    nw, nh = int((x2 - x1) * zoom), int((y2 - y1) * zoom)
    nx1 = max(0, cx - nw // 2)
    ny1 = max(0, cy - nh // 2)
    nx2 = min(w, cx + nw // 2)
    ny2 = min(h, cy + nh // 2)
    return nx1, ny1, nx2, ny2


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """两个矩形框的 IoU。"""
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


class EasyOCRTextDetector:
    """EasyOCR + 可选 YOLO 文本检测器，实现 ``TextDetector`` Protocol。"""

    def __init__(
        self,
        *,
        yolo_conf: float = _cfg.YOLO_CONF,
        min_ocr_conf: float = _cfg.OCR_MIN_CONFIDENCE,
        zoom: float = 1.5,
        full_image_ocr: bool = True,
        fallback_ocr_conf: float = 0.3,
        ocr_upscale: float = 2.0,
    ) -> None:
        self._yolo_conf = yolo_conf
        self._min_ocr_conf = min_ocr_conf
        self._zoom = zoom
        self._full_image_ocr = full_image_ocr
        self._fallback_ocr_conf = fallback_ocr_conf
        self._ocr_upscale = ocr_upscale

    # ---- TextDetector Protocol ----

    def detect(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        timestamp_ns: int,
    ) -> list[TextDetection]:
        """检测并识别单帧中的文本。

        Returns:
            TextDetection 列表（bbox 已归一化），无文本时为空。
        """
        h, w = frame_rgb.shape[:2]
        reader = get_ocr_reader()
        detections: list[TextDetection] = []

        # ---- 路径 1: YOLO 定位 + 裁剪 OCR ----
        try:
            yolo = get_yolo()
            results = yolo(frame_rgb, conf=self._yolo_conf, verbose=False)
            for box in results[0].boxes.xyxy.cpu().numpy():
                nx1, ny1, nx2, ny2 = _zoomed_crop(frame_rgb, *box, self._zoom)
                crop = frame_rgb[ny1:ny2, nx1:nx2]
                if crop.size == 0:
                    continue

                ocr_results = reader.readtext(
                    _preprocess_crop(crop),
                    text_threshold=_cfg.OCR_TEXT_THRESHOLD,
                    low_text=_cfg.OCR_LOW_TEXT,
                )
                texts_list, confs = [], []
                for _, text, conf in ocr_results:
                    if conf >= self._min_ocr_conf and text.strip():
                        texts_list.append(text.strip())
                        confs.append(float(conf))
                if not texts_list:
                    continue

                detections.append(TextDetection(
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    bbox_xyxy=(
                        float(nx1) / w, float(ny1) / h,
                        float(nx2) / w, float(ny2) / h,
                    ),
                    text=" ".join(texts_list),
                    confidence=max(confs),
                    detector="easyocr+yolo",
                ))
        except (FileNotFoundError, ImportError):
            pass  # YOLO 文本模型不可用，走全图 OCR 兜底

        # ---- 路径 2: 全图 OCR 兜底 ----
        if self._full_image_ocr:
            up = frame_rgb
            if self._ocr_upscale > 1:
                up = cv2.resize(
                    frame_rgb, None,
                    fx=self._ocr_upscale, fy=self._ocr_upscale,
                    interpolation=cv2.INTER_CUBIC,
                )
            for bbox4, text, conf in reader.readtext(
                up,
                text_threshold=_cfg.OCR_TEXT_THRESHOLD,
                low_text=_cfg.OCR_LOW_TEXT,
            ):
                if conf < self._fallback_ocr_conf or not text.strip():
                    continue
                if len(text.strip()) < 2:
                    continue
                xs = [p[0] / self._ocr_upscale for p in bbox4]
                ys = [p[1] / self._ocr_upscale for p in bbox4]
                pixel_box = (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))
                # 与已检出区域去重
                if any(
                    _iou(pixel_box, (
                        int(d.bbox_xyxy[0] * w), int(d.bbox_xyxy[1] * h),
                        int(d.bbox_xyxy[2] * w), int(d.bbox_xyxy[3] * h),
                    )) > 0.5
                    for d in detections
                ):
                    continue
                # easyocr 的框可能略微超出图像边缘，clamp 后再归一化
                cx1 = max(0, min(w, int(min(xs))))
                cy1 = max(0, min(h, int(min(ys))))
                cx2 = max(0, min(w, int(max(xs))))
                cy2 = max(0, min(h, int(max(ys))))
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                detections.append(TextDetection(
                    frame_index=frame_index,
                    timestamp_ns=timestamp_ns,
                    bbox_xyxy=(
                        float(cx1) / w, float(cy1) / h,
                        float(cx2) / w, float(cy2) / h,
                    ),
                    text=text.strip(),
                    confidence=float(conf),
                    detector="easyocr",
                ))

        return detections

    def close(self) -> None:
        """释放 OCR/YOLO 资源。"""


__all__ = [
    "EasyOCRTextDetector",
    "get_ocr_reader",
    "get_yolo",
]
