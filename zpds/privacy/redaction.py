"""遮挡算子：blur / pixelate / black_rect + 跨帧 IoU 平滑。

实现 ``Redactor`` Protocol。输入归一化 bbox，自动映射到像素坐标。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from zpds.privacy.schemas import RedactionMethod, RedactionRegion, VALID_METHODS

# ---------------------------------------------------------------------------
# 单帧遮挡
# ---------------------------------------------------------------------------


def _denorm_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """归一化 bbox → 像素坐标。"""
    x1 = max(0, int(bbox_xyxy[0] * width))
    y1 = max(0, int(bbox_xyxy[1] * height))
    x2 = min(width, int(bbox_xyxy[2] * width))
    y2 = min(height, int(bbox_xyxy[3] * height))
    return x1, y1, x2, y2


def apply_blur(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    ksize: int = 41,
    sigma: int = 15,
) -> None:
    """原地高斯模糊一个区域。"""
    if x2 <= x1 or y2 <= y1:
        return
    ksize = max(3, ksize | 1)  # 强制奇数
    roi = image[y1:y2, x1:x2]
    image[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (ksize, ksize), sigma)


def apply_pixelate(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
    blocks: int = 10,
) -> None:
    """原地像素化一个区域。"""
    if x2 <= x1 or y2 <= y1:
        return
    roi = image[y1:y2, x1:x2]
    rh, rw = roi.shape[:2]
    bw = max(1, rw // blocks)
    bh = max(1, rh // blocks)
    small = cv2.resize(roi, (bw, bh), interpolation=cv2.INTER_LINEAR)
    image[y1:y2, x1:x2] = cv2.resize(small, (rw, rh), interpolation=cv2.INTER_NEAREST)


def apply_black_rect(
    image: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> None:
    """原地用黑色矩形遮挡一个区域。"""
    if x2 <= x1 or y2 <= y1:
        return
    image[y1:y2, x1:x2] = 0


_METHOD_FN = {
    "blur": apply_blur,
    "pixelate": apply_pixelate,
    "black_rect": apply_black_rect,
}


class FrameRedactor:
    """单帧遮挡器，实现 ``Redactor`` Protocol。"""

    def __init__(
        self,
        face_method: RedactionMethod = "blur",
        text_method: RedactionMethod = "black_rect",
        blur_ksize: int = 41,
        blur_sigma: int = 15,
        pixelate_blocks: int = 10,
    ) -> None:
        if face_method not in VALID_METHODS:
            raise ValueError(f"face_method 必须是 {sorted(VALID_METHODS)}")
        if text_method not in VALID_METHODS:
            raise ValueError(f"text_method 必须是 {sorted(VALID_METHODS)}")
        self._face_method = face_method
        self._text_method = text_method
        self._blur_ksize = blur_ksize
        self._blur_sigma = blur_sigma
        self._pixelate_blocks = pixelate_blocks

    # ---- Redactor Protocol ----

    def apply(
        self,
        frame_rgb: np.ndarray,
        regions: list[RedactionRegion],
    ) -> np.ndarray:
        """对单帧应用所有遮挡，返回新数组（不修改原图）。"""
        result = frame_rgb.copy()
        h, w = result.shape[:2]

        for region in regions:
            x1, y1, x2, y2 = _denorm_bbox(region.bbox_xyxy, w, h)
            if x2 <= x1 or y2 <= y1:
                continue

            method = region.method
            if method == "blur":
                apply_blur(result, x1, y1, x2, y2, self._blur_ksize, self._blur_sigma)
            elif method == "pixelate":
                apply_pixelate(result, x1, y1, x2, y2, self._pixelate_blocks)
            elif method == "black_rect":
                apply_black_rect(result, x1, y1, x2, y2)

        return result

    @property
    def face_method(self) -> str:
        return self._face_method

    @property
    def text_method(self) -> str:
        return self._text_method


# ---------------------------------------------------------------------------
# 跨帧 IoU 平滑 — 减少相邻帧遮挡框抖动
# ---------------------------------------------------------------------------


@dataclass
class _SmoothedRegion:
    bbox_xyxy: tuple[float, float, float, float]
    count: int = 1


class TemporalSmoother:
    """跨帧 IoU 平滑器：同一目标在相邻帧的检测框因微小抖动产生不同坐标时，
    用滑动窗口内的历史框做加权平均，减少闪烁。"""

    def __init__(
        self,
        window_frames: int = 5,
        iou_threshold: float = 0.3,
    ) -> None:
        if window_frames < 1:
            raise ValueError("window_frames 必须 >= 1")
        if not 0 <= iou_threshold <= 1:
            raise ValueError("iou_threshold 必须在 [0, 1]")
        self._window = window_frames
        self._iou_threshold = iou_threshold
        self._history: deque[list[_SmoothedRegion]] = deque(maxlen=window_frames)

    def smooth(self, regions: list[RedactionRegion]) -> list[RedactionRegion]:
        """对一帧的遮挡区域做时序平滑。"""
        if not self._history:
            self._history.append(
                [_SmoothedRegion(bbox_xyxy=r.bbox_xyxy) for r in regions]
            )
            return list(regions)

        prev = self._history[-1]
        smoothed: list[_SmoothedRegion] = []

        for region in regions:
            best_iou = 0.0
            best_idx = -1
            for i, pr in enumerate(prev):
                iou = _iou_norm(region.bbox_xyxy, pr.bbox_xyxy)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

            if best_iou >= self._iou_threshold and best_idx >= 0:
                matched = prev[best_idx]
                weight = 1.0 / (matched.count + 1)
                new_bbox = tuple(
                    matched.bbox_xyxy[j] * (1 - weight) + region.bbox_xyxy[j] * weight
                    for j in range(4)
                )
                matched.bbox_xyxy = new_bbox
                matched.count += 1
                smoothed.append(matched)
            else:
                smoothed.append(_SmoothedRegion(bbox_xyxy=region.bbox_xyxy))

        self._history.append(smoothed)

        return [
            RedactionRegion(
                kind=region.kind,
                bbox_xyxy=s.bbox_xyxy,
                method=region.method,
                category=region.category,
                confidence=region.confidence,
            )
            for region, s in zip(regions, smoothed)
        ]


def _iou_norm(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    """两个归一化 bbox 的 IoU。"""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-9)


__all__ = [
    "FrameRedactor",
    "TemporalSmoother",
    "apply_blur",
    "apply_black_rect",
    "apply_pixelate",
]
