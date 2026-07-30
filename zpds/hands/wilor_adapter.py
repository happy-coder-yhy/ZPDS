"""WiLoR 适配层。

位于 WiLoRBackend（原始推理）和 Pipeline（统一 HandObservation）之间，
负责：

- 输入校验
- 图像变换链记录
- 检测器 BBox 逆变换（letterbox → resize → 原图）
- BBox 合法性检查
- handedness 规范化
- 输出 :class:`WiLoRDetection`

不在此层做 21 点映射或 RawHandResult 转换。
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from zpds.hands.backends.wilor import WiLoRBackend
from zpds.hands.wilor_schema import (
    InvalidDetectionError,
    WiLoRDetection,
    WiLoRImageTransform,
)


# ════════════════════════════════════════════════════════════════════
# 适配器
# ════════════════════════════════════════════════════════════════════


class WiLoRAdapter:
    """WiLoR 适配器。

    封装 Backend → 输入校验 → 坐标逆变换 → BBox 校验 → WiLoRDetection。

    用法::

        backend = WiLoRBackend(config)
        adapter = WiLoRAdapter(backend)
        detection = adapter.detect(frame_rgb, timestamp_ms=0)
    """

    def __init__(self, backend: WiLoRBackend) -> None:
        self._backend = backend

    @property
    def backend(self) -> WiLoRBackend:
        return self._backend

    def detect(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[WiLoRDetection]:
        """对单帧执行检测，返回原图像素 BBox。

        完整流程::

            输入校验
                ↓
            backend.infer_raw(frame_rgb)
                ↓
            提取检测结果
                ↓
            BBox 逆变换（detector 坐标 → 原图坐标）
                ↓
            BBox 合法性检查
                ↓
            handedness 规范化
                ↓
            WiLoRDetection[]

        Args:
            frame_rgb: RGB uint8 图像 ``(H, W, 3)``。
            timestamp_ms: 帧时间戳（毫秒）。

        Returns:
            WiLoRDetection 列表，无检测时为空列表。

        Raises:
            TypeError: 输入类型不合法。
            ValueError: 输入形状/范围不合法。
            InvalidDetectionError: BBox NaN/Inf、坐标顺序错误、超出原图。
        """
        # ---- 1. 输入校验 ----
        _validate_input(frame_rgb, timestamp_ms)
        h, w = frame_rgb.shape[:2]

        # ---- 2. 后端原始推理 ----
        raw = self._backend.infer_raw(frame_rgb)

        # ---- 3. 提取检测 + 逆变换 + 校验 ----
        detections = _extract_detections(raw, w, h)

        return detections

    def close(self) -> None:
        self._backend.close()


# ════════════════════════════════════════════════════════════════════
# 输入校验
# ════════════════════════════════════════════════════════════════════


def _validate_input(frame_rgb: np.ndarray, timestamp_ms: int) -> None:
    """校验单帧输入。与 MediaPipe adapter 保持一致。"""
    if not isinstance(frame_rgb, np.ndarray):
        raise TypeError("frame_rgb 必须是 np.ndarray")

    if frame_rgb.size == 0:
        raise ValueError("frame_rgb 不能为空")

    if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
        raise ValueError(
            f"frame_rgb 必须是形状为 (H, W, 3) 的 RGB 图像，"
            f"实际形状 {frame_rgb.shape}"
        )

    if frame_rgb.dtype != np.uint8:
        raise TypeError(f"frame_rgb 必须是 uint8，实际 {frame_rgb.dtype}")

    if timestamp_ms < 0:
        raise ValueError(f"timestamp_ms 不能为负数，实际 {timestamp_ms}")


# ════════════════════════════════════════════════════════════════════
# BBox 逆变换
# ════════════════════════════════════════════════════════════════════


def inverse_project_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    transform: WiLoRImageTransform,
) -> tuple[float, float, float, float]:
    """将检测器坐标系 BBox 逆变换回原图像素坐标。

    变换链（逆向）::

        detector BBox (letterbox 后图像坐标)
            ↓ 去除 letterbox padding
        resize 后坐标
            ↓ 逆 resize
        原图像素坐标
            ↓ clip 到 [0, w-1] × [0, h-1]
        最终 BBox

    Args:
        bbox_xyxy: 检测器输出的 ``(x1, y1, x2, y2)``，位于 letterbox 后坐标系。
        transform: 从原图到检测器输入的变换参数。

    Returns:
        原图像素 ``(x1, y1, x2, y2)``。
    """
    x1, y1, x2, y2 = bbox_xyxy

    # 1. 去除 letterbox padding
    x1_unpadded = x1 - transform.letterbox_left
    y1_unpadded = y1 - transform.letterbox_top
    x2_unpadded = x2 - transform.letterbox_left
    y2_unpadded = y2 - transform.letterbox_top

    # 2. 逆 resize
    if transform.resize_scale_x > 0:
        orig_x1 = x1_unpadded / transform.resize_scale_x
        orig_x2 = x2_unpadded / transform.resize_scale_x
    else:
        orig_x1 = x1_unpadded
        orig_x2 = x2_unpadded

    if transform.resize_scale_y > 0:
        orig_y1 = y1_unpadded / transform.resize_scale_y
        orig_y2 = y2_unpadded / transform.resize_scale_y
    else:
        orig_y1 = y1_unpadded
        orig_y2 = y2_unpadded

    # 3. clip 到原图范围
    w = float(transform.original_width)
    h = float(transform.original_height)

    x1_clipped = float(np.clip(orig_x1, 0.0, w - 1.0))
    y1_clipped = float(np.clip(orig_y1, 0.0, h - 1.0))
    x2_clipped = float(np.clip(orig_x2, 0.0, w - 1.0))
    y2_clipped = float(np.clip(orig_y2, 0.0, h - 1.0))

    return (x1_clipped, y1_clipped, x2_clipped, y2_clipped)


# ════════════════════════════════════════════════════════════════════
# BBox 合法性检查
# ════════════════════════════════════════════════════════════════════


def validate_bbox(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> None:
    """校验 BBox 合法性。

    检查项：
    - 无 NaN / Inf
    - x1 <= x2 且 y1 <= y2
    - 坐标在原图范围内
    - 面积不小于 2px

    Raises:
        InvalidDetectionError: 任一检查未通过。
    """
    x1, y1, x2, y2 = bbox_xyxy

    if not all(np.isfinite([x1, y1, x2, y2])):
        raise InvalidDetectionError(
            f"BBox 包含 NaN 或 Inf: {bbox_xyxy}"
        )

    if x1 > x2 or y1 > y2:
        raise InvalidDetectionError(
            f"BBox 坐标顺序错误 (x1 > x2 或 y1 > y2): {bbox_xyxy}"
        )

    if not (
        0.0 <= x1 <= image_width - 1
        and 0.0 <= x2 <= image_width - 1
        and 0.0 <= y1 <= image_height - 1
        and 0.0 <= y2 <= image_height - 1
    ):
        raise InvalidDetectionError(
            f"BBox 超出原图范围 ({image_width}×{image_height}): {bbox_xyxy}"
        )

    box_width = x2 - x1
    box_height = y2 - y1

    if box_width < 2.0 or box_height < 2.0:
        raise InvalidDetectionError(
            f"BBox 面积过小 ({box_width:.1f}×{box_height:.1f}): {bbox_xyxy}"
        )


def check_bbox_clipped(
    bbox_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> bool:
    """检查 BBox 是否在逆变换后被裁剪到图像边界。

    Returns:
        True 如果有坐标被 clip 到边界。
    """
    x1, y1, x2, y2 = bbox_xyxy
    return (
        x1 <= 0.0
        or y1 <= 0.0
        or x2 >= image_width - 1.0
        or y2 >= image_height - 1.0
    )


# ════════════════════════════════════════════════════════════════════
# handedness 规范化
# ════════════════════════════════════════════════════════════════════


def normalize_handedness(raw_label: str | None) -> str:
    """将 WiLoR 上游 handedness 统一为大写首字母格式。

    不允许根据 BBox 位置推断左右手。

    Args:
        raw_label: WiLoR 上游原始标签（可能为 None）。

    Returns:
        ``"Left"``、``"Right"`` 或 ``"Unknown"``。
    """
    if raw_label is None:
        return "Unknown"

    normalized = raw_label.strip().lower()

    if normalized == "left":
        return "Left"
    if normalized == "right":
        return "Right"

    return "Unknown"


# ════════════════════════════════════════════════════════════════════
# 检测提取
# ════════════════════════════════════════════════════════════════════


def _extract_detections(
    raw_result: dict[str, Any],
    image_width: int,
    image_height: int,
) -> list[WiLoRDetection]:
    """从 WiLoR 原始输出提取检测结果。

    使用 YOLO 检测器产生的 BBox + 左右手标签构建 WiLoRDetection。
    不在此层做 21 点坐标投影（由 joint_mapping 负责）。

    Args:
        raw_result: ``WiLoRBackend.infer_raw()`` 返回的字典。
        image_width: 原图宽度（像素）。
        image_height: 原图高度（像素）。

    Returns:
        WiLoRDetection 列表。
    """
    detections: list[WiLoRDetection] = []

    bboxes = raw_result.get("boxes", [])
    is_right = raw_result.get("is_right", [])
    all_kp3d = raw_result.get("pred_keypoints_3d") or []
    all_kp2d = raw_result.get("pred_keypoints_2d") or []
    all_cam_t = raw_result.get("pred_cam_t") or []
    all_focal = raw_result.get("focal_length") or []

    for i in range(len(bboxes)):
        x1, y1, x2, y2 = bboxes[i]
        right_flag = bool(is_right[i]) if i < len(is_right) else False

        # WiLoR 原始关节（保留为原始格式，供后续 mapping 使用）
        joints_3d = all_kp3d[i] if i < len(all_kp3d) else None
        joints_2d_raw = all_kp2d[i] if i < len(all_kp2d) else None

        # BBox 校验
        try:
            validate_bbox((float(x1), float(y1), float(x2), float(y2)),
                          image_width, image_height)
            clipped = check_bbox_clipped(
                (float(x1), float(y1), float(x2), float(y2)),
                image_width, image_height,
            )
        except InvalidDetectionError:
            continue  # 跳过不合法 BBox

        handedness = "Right" if right_flag else "Left"
        transform = WiLoRImageTransform.from_resize(
            original_width=image_width,
            original_height=image_height,
            detector_width=image_width,
            detector_height=image_height,
        )

        detection = WiLoRDetection(
            handedness=handedness,
            handedness_score=0.8,  # WiLoR 不直接提供，由 YOLO cls 给出
            detection_score=0.8,
            bbox_xyxy_px=(float(x1), float(y1), float(x2), float(y2)),
            raw_keypoints_2d=joints_2d_raw,
            raw_keypoint_format="wilor_model_crop",
            raw_keypoints_3d=joints_3d,
            cam_t=all_cam_t[i] if i < len(all_cam_t) else None,
            focal=all_focal[i] if i < len(all_focal) else None,
            clipped=clipped,
            transform=transform,
        )
        detections.append(detection)

    return detections


__all__ = [
    "WiLoRAdapter",
    "check_bbox_clipped",
    "inverse_project_bbox",
    "normalize_handedness",
    "validate_bbox",
]
