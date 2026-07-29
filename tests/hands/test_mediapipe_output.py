"""test_mediapipe_output — 验证 RawHandResult 构造和工厂方法。"""

import pytest

from zpds.hands.base import HandBBox, HandKeypoints, RawHandResult


class TestRawHandResult:
    def test_valid_hand(self):
        kp = HandKeypoints(
            normalized=[(0.5, 0.5, 0.0)] * 21,
            pixel=[(100.0, 200.0)] * 21,
        )
        bbox = HandBBox(x1=80, y1=180, x2=320, y2=380, confidence=0.95)
        hand = RawHandResult(
            handedness="Left", handedness_score=0.93,
            keypoints=kp, bbox=bbox, detection_score=0.91,
            label="hand_0",
        )
        assert hand.handedness == "Left"
        assert hand.handedness_score == 0.93
        assert len(hand.keypoints.pixel) == 21
        assert hand.bbox.area > 0

    def test_keypoints_wrong_count_raises(self):
        with pytest.raises(ValueError, match="21"):
            HandKeypoints(
                normalized=[(0.5, 0.5, 0.0)] * 10,
                pixel=[(100.0, 200.0)] * 10,
            )

    def test_keypoints_pixel_mismatch_raises(self):
        with pytest.raises(ValueError):
            HandKeypoints(
                normalized=[(0.5, 0.5, 0.0)] * 21,
                pixel=[(100.0, 200.0)] * 10,
            )

    def test_bbox_area(self):
        bbox = HandBBox(x1=0, y1=0, x2=100, y2=100)
        assert bbox.area == 10000
        assert bbox.is_valid

    def test_bbox_invalid(self):
        bbox = HandBBox(x1=100, y1=100, x2=0, y2=0)
        assert not bbox.is_valid

    def test_bbox_with_padding(self):
        bbox = HandBBox(
            x1=100, y1=100, x2=200, y2=200,
            is_padded=True, padding_ratio=0.10,
        )
        assert bbox.is_padded
        assert bbox.width == 100
