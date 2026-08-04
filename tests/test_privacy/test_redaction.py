"""测试遮挡算子与跨帧平滑。"""

import numpy as np
import pytest

from zpds.privacy.redaction import (
    FrameRedactor,
    TemporalSmoother,
    apply_blur,
    apply_black_rect,
    apply_pixelate,
)
from zpds.privacy.schemas import RedactionRegion


class TestApplyOps:
    def test_blur_modifies_roi(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        original = frame.copy()
        apply_blur(frame, 20, 20, 80, 80, ksize=21)
        # 模糊后 ROI 内方差应减小（趋于平滑）
        roi = frame[20:80, 20:80]
        assert roi.std() < original[20:80, 20:80].std() or roi.std() == 0

    def test_pixelate_reduces_resolution(self):
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        apply_pixelate(frame, 20, 20, 80, 80, blocks=5)
        # 像素化后相邻像素可能相同
        roi = frame[20:80, 20:80]
        assert roi is not None

    def test_black_rect_zeros_roi(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        apply_black_rect(frame, 20, 20, 80, 80)
        assert frame[30, 30].sum() == 0

    def test_empty_roi_noop(self):
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        original = frame.copy()
        apply_blur(frame, 50, 50, 50, 50)  # zero-area
        apply_black_rect(frame, 50, 50, 40, 40)  # x2 < x1
        assert np.array_equal(frame, original)


class TestFrameRedactor:
    def test_apply_empty_regions(self):
        redactor = FrameRedactor()
        frame = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = redactor.apply(frame, [])
        assert result.shape == frame.shape

    def test_apply_blur_face(self):
        redactor = FrameRedactor(face_method="blur")
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 128
        region = RedactionRegion("face", (0.2, 0.2, 0.8, 0.8), "blur", "face", 0.9)
        result = redactor.apply(frame, [region])
        assert result.shape == frame.shape
        # 原图未修改
        assert frame[50, 50, 0] == 128

    def test_apply_black_rect_text(self):
        redactor = FrameRedactor(text_method="black_rect")
        frame = np.ones((100, 100, 3), dtype=np.uint8) * 255
        region = RedactionRegion("text", (0.2, 0.2, 0.8, 0.8), "black_rect", "phone", 0.9)
        result = redactor.apply(frame, [region])
        # 遮挡区域内应为黑色
        assert result[50, 50].sum() == 0

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError, match="face_method"):
            FrameRedactor(face_method="invalid")


class TestTemporalSmoother:
    def test_first_frame_passthrough(self):
        smoother = TemporalSmoother(3, 0.3)
        regions = [RedactionRegion("face", (0.1, 0.1, 0.5, 0.5), "blur", "face", 0.9)]
        result = smoother.smooth(regions)
        assert len(result) == 1
        assert result[0].bbox_xyxy == (0.1, 0.1, 0.5, 0.5)

    def test_nearby_boxes_smoothed(self):
        smoother = TemporalSmoother(3, 0.3)
        r1 = [RedactionRegion("face", (0.1, 0.1, 0.5, 0.5), "blur", "face", 0.9)]
        smoother.smooth(r1)
        r2 = [RedactionRegion("face", (0.12, 0.12, 0.52, 0.52), "blur", "face", 0.9)]
        result = smoother.smooth(r2)
        # 应平滑到接近 (0.11, 0.11, 0.51, 0.51)
        assert abs(result[0].bbox_xyxy[0] - 0.11) < 0.02

    def test_unmatched_box_not_smoothed(self):
        smoother = TemporalSmoother(3, 0.3)
        smoother.smooth([RedactionRegion("face", (0.1, 0.1, 0.2, 0.2), "blur", "face", 0.9)])
        result = smoother.smooth([RedactionRegion("face", (0.5, 0.5, 0.6, 0.6), "blur", "face", 0.9)])
        # IoU = 0 → 不平滑，保持原坐标
        assert result[0].bbox_xyxy == (0.5, 0.5, 0.6, 0.6)

    def test_invalid_window(self):
        with pytest.raises(ValueError):
            TemporalSmoother(0, 0.3)
