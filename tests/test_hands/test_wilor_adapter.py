"""WiLoR 适配层阶段 2 测试。

验证：
- 输入校验
- BBox 逆变换（letterbox → resize → 原图）
- BBox 合法性检查
- handedness 规范化
- BBox 裁剪检测
"""

from __future__ import annotations

import numpy as np
import pytest

from zpds.hands.wilor_adapter import (
    check_bbox_clipped,
    inverse_project_bbox,
    normalize_handedness,
    validate_bbox,
    _validate_input,
)
from zpds.hands.wilor_schema import (
    InvalidDetectionError,
    WiLoRImageTransform,
)


# ════════════════════════════════════════════════════════════════════
# 输入校验
# ════════════════════════════════════════════════════════════════════


def test_validate_input_accepts_valid_frame() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    _validate_input(frame, 0)  # 不抛异常


def test_validate_input_rejects_none() -> None:
    with pytest.raises(TypeError, match="np.ndarray"):
        _validate_input(None, 0)  # type: ignore[arg-type]


def test_validate_input_rejects_empty() -> None:
    with pytest.raises(ValueError, match="不能为空"):
        _validate_input(np.array([], dtype=np.uint8), 0)


def test_validate_input_rejects_wrong_ndim() -> None:
    frame = np.zeros((480, 640), dtype=np.uint8)  # 2D
    with pytest.raises(ValueError, match="H, W, 3"):
        _validate_input(frame, 0)


def test_validate_input_rejects_wrong_channels() -> None:
    frame = np.zeros((480, 640, 1), dtype=np.uint8)  # 灰度
    with pytest.raises(ValueError, match="H, W, 3"):
        _validate_input(frame, 0)


def test_validate_input_rejects_wrong_dtype() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.float32)
    with pytest.raises(TypeError, match="uint8"):
        _validate_input(frame, 0)


def test_validate_input_rejects_negative_timestamp() -> None:
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="timestamp_ms"):
        _validate_input(frame, -1)


# ════════════════════════════════════════════════════════════════════
# BBox 逆变换
# ════════════════════════════════════════════════════════════════════


_MOCK_TRANSFORM = WiLoRImageTransform.from_resize(
    original_width=1920,
    original_height=1080,
    detector_width=640,
    detector_height=384,
    letterbox_left=0.0,
    letterbox_top=0.0,
)


def test_inverse_project_bbox_no_padding() -> None:
    """无 letterbox 时，逆变换 = 逆 resize。"""
    transform = WiLoRImageTransform.from_resize(
        original_width=1920,
        original_height=1080,
        detector_width=640,
        detector_height=384,
    )
    # detector 坐标 (320, 192) 对应原图中心 (960, 540)
    bbox = inverse_project_bbox((300.0, 180.0, 340.0, 204.0), transform)

    assert bbox[0] == pytest.approx(900.0)   # 300 / (640/1920) = 900
    assert bbox[1] == pytest.approx(506.25)  # 180 / (384/1080) ≈ 506.25
    assert bbox[2] == pytest.approx(1020.0)  # 340 * 3 = 1020
    assert bbox[3] == pytest.approx(573.75)  # 204 * 2.8125 ≈ 573.75


def test_inverse_project_bbox_with_letterbox() -> None:
    """有 letterbox padding 时先减去 padding 再逆 resize。"""
    transform = WiLoRImageTransform.from_resize(
        original_width=1920,
        original_height=1080,
        detector_width=600,
        detector_height=400,
        letterbox_left=20.0,
        letterbox_top=10.0,
    )
    # detector 坐标 (320, 210) — 去 padding 后 (300, 200)
    # 逆 resize: 300 / (600/1920) = 960, 200 / (400/1080) = 540
    bbox = inverse_project_bbox((320.0, 210.0, 380.0, 250.0), transform)

    scale_x = 1920 / 600  # 3.2
    scale_y = 1080 / 400  # 2.7

    assert bbox[0] == pytest.approx((320.0 - 20.0) * scale_x)  # 960
    assert bbox[1] == pytest.approx((210.0 - 10.0) * scale_y)  # 540
    assert bbox[2] == pytest.approx((380.0 - 20.0) * scale_x)  # 1152
    assert bbox[3] == pytest.approx((250.0 - 10.0) * scale_y)  # 648


def test_inverse_project_bbox_clips_to_image_bounds() -> None:
    """超出原图范围的坐标应被裁剪。"""
    transform = WiLoRImageTransform.from_resize(
        original_width=640,
        original_height=480,
        detector_width=640,
        detector_height=480,
    )
    # 逆 resize 后超出范围
    bbox = inverse_project_bbox((-10.0, -10.0, 700.0, 500.0), transform)

    assert bbox[0] == 0.0  # clipped
    assert bbox[1] == 0.0  # clipped
    assert bbox[2] == 639.0  # clipped to w-1
    assert bbox[3] == 479.0  # clipped to h-1


def test_inverse_project_bbox_different_aspect_ratio() -> None:
    """非等比例 resize — x 和 y 方向缩放因子不同。"""
    transform = WiLoRImageTransform.from_resize(
        original_width=1280,
        original_height=720,
        detector_width=512,
        detector_height=256,
    )
    bbox = inverse_project_bbox((100.0, 50.0, 400.0, 200.0), transform)

    scale_x = 1280 / 512  # 2.5
    scale_y = 720 / 256   # 2.8125

    assert bbox[0] == pytest.approx(100.0 * scale_x)
    assert bbox[1] == pytest.approx(50.0 * scale_y)
    assert bbox[2] == pytest.approx(400.0 * scale_x)
    assert bbox[3] == pytest.approx(200.0 * scale_y)


# ════════════════════════════════════════════════════════════════════
# BBox 合法性检查
# ════════════════════════════════════════════════════════════════════


def test_validate_bbox_accepts_valid() -> None:
    validate_bbox((10.0, 20.0, 100.0, 120.0), 640, 480)  # 不抛异常


def test_validate_bbox_rejects_nan() -> None:
    with pytest.raises(InvalidDetectionError, match="NaN"):
        validate_bbox((float("nan"), 20.0, 100.0, 120.0), 640, 480)


def test_validate_bbox_rejects_inf() -> None:
    with pytest.raises(InvalidDetectionError, match="Inf"):
        validate_bbox((10.0, float("inf"), 100.0, 120.0), 640, 480)


def test_validate_bbox_rejects_inverted() -> None:
    with pytest.raises(InvalidDetectionError, match="坐标顺序错误"):
        validate_bbox((100.0, 120.0, 10.0, 20.0), 640, 480)


def test_validate_bbox_rejects_out_of_bounds() -> None:
    with pytest.raises(InvalidDetectionError, match="超出原图范围"):
        validate_bbox((10.0, 20.0, 700.0, 500.0), 640, 480)


def test_validate_bbox_rejects_out_of_bounds_negative() -> None:
    with pytest.raises(InvalidDetectionError, match="超出原图范围"):
        validate_bbox((-5.0, 20.0, 100.0, 120.0), 640, 480)


def test_validate_bbox_rejects_too_small() -> None:
    with pytest.raises(InvalidDetectionError, match="面积过小"):
        validate_bbox((10.0, 20.0, 11.0, 21.0), 640, 480)  # 1px × 1px


def test_validate_bbox_accepts_minimum_size() -> None:
    validate_bbox((10.0, 20.0, 12.0, 22.0), 640, 480)  # 2px × 2px


# ════════════════════════════════════════════════════════════════════
# BBox 裁剪检测
# ════════════════════════════════════════════════════════════════════


def test_check_bbox_clipped_false() -> None:
    assert not check_bbox_clipped((10.0, 20.0, 100.0, 120.0), 640, 480)


def test_check_bbox_clipped_true_at_edge() -> None:
    assert check_bbox_clipped((0.0, 20.0, 100.0, 120.0), 640, 480)


def test_check_bbox_clipped_true_at_bottom_right() -> None:
    assert check_bbox_clipped((10.0, 20.0, 639.0, 479.0), 640, 480)


# ════════════════════════════════════════════════════════════════════
# handedness 规范化
# ════════════════════════════════════════════════════════════════════


def test_normalize_handedness_left() -> None:
    assert normalize_handedness("left") == "Left"
    assert normalize_handedness("Left") == "Left"
    assert normalize_handedness("LEFT") == "Left"
    assert normalize_handedness(" left ") == "Left"


def test_normalize_handedness_right() -> None:
    assert normalize_handedness("right") == "Right"
    assert normalize_handedness("Right") == "Right"


def test_normalize_handedness_unknown() -> None:
    assert normalize_handedness("both") == "Unknown"
    assert normalize_handedness("") == "Unknown"
    assert normalize_handedness("unknown") == "Unknown"


def test_normalize_handedness_none() -> None:
    assert normalize_handedness(None) == "Unknown"


def test_normalize_handedness_does_not_use_position() -> None:
    """handedness 不得根据位置推断 — 不明就是 Unknown。"""
    assert normalize_handedness("first_hand") == "Unknown"
    assert normalize_handedness("0") == "Unknown"
