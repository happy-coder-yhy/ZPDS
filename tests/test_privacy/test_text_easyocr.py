"""EasyOCRTextDetector 全图 OCR 兜底路径的 bbox 边界回归测试。

历史事故：easyocr 的框可能略微超出图像边缘（如 x2=1.015），
未 clamp 会被 ``_require_normalized_bbox`` 拒绝抛错，导致整条
脱敏流程崩溃（制造-鼠标套袋2 处理时实测）。此处锁定 clamp 行为。
"""

import numpy as np
import pytest

from zpds.privacy.backends import text_easyocr as te


class _FakeReader:
    """极简 fake：readtext 直接返回预设的 (bbox4, text, conf) 列表。"""

    def __init__(self, ocr_result):
        self._ocr = ocr_result

    def readtext(self, image, **kwargs):  # noqa: ARG002
        return self._ocr


@pytest.fixture
def detector(monkeypatch):
    """构造走全图 OCR 兜底路径的检测器（YOLO 不可用）。"""

    def _no_yolo(*args, **kwargs):  # noqa: ARG001
        raise FileNotFoundError("YOLO 文本模型不存在（测试）")

    def make(ocr_result):
        monkeypatch.setattr(te, "get_yolo", _no_yolo)
        monkeypatch.setattr(te, "get_ocr_reader", lambda: _FakeReader(ocr_result))
        return te.EasyOCRTextDetector(
            full_image_ocr=True,
            ocr_upscale=2.0,
            fallback_ocr_conf=0.3,
        )

    return make


def _frame() -> np.ndarray:
    return np.zeros((720, 1280, 3), dtype=np.uint8)


class TestFullImageOcrBboxClamp:
    def test_bbox_oob_positive_clamped(self, detector):
        """bbox 超出右/下边缘：clamp 到 [0, 1]，不得抛错。"""
        # upscale 2x 后坐标：x2=2580 → 原图 1290 > 1280 越界
        oob = [(0, 10), (10, 10), (2580, 710), (2560, 710)]
        det = detector([(oob, "测试文本", 0.9)])
        results = det.detect(_frame(), frame_index=0, timestamp_ns=0)

        assert len(results) == 1
        bx1, by1, bx2, by2 = results[0].bbox_xyxy
        assert bx2 <= 1.0 and by2 <= 1.0
        assert bx1 >= 0.0 and by1 >= 0.0
        assert bx2 == 1.0  # 越界方向被 clamp 到图像边缘
        assert results[0].detector == "easyocr"

    def test_bbox_oob_negative_clamped(self, detector):
        """bbox 超出左/上边缘：clamp 到 0，不得抛错。"""
        oob = [(0, -10), (10, -10), (510, 710), (500, 710)]
        det = detector([(oob, "测试文本", 0.9)])
        results = det.detect(_frame(), frame_index=0, timestamp_ns=0)

        assert len(results) == 1
        bx1, by1, bx2, by2 = results[0].bbox_xyxy
        assert bx1 >= 0.0 and by1 >= 0.0
        assert by1 == 0.0
        assert bx2 <= 1.0 and by2 <= 1.0

    def test_fully_outside_bbox_skipped(self, detector):
        """完全越界导致 clamp 后退化（cx2<=cx1）：跳过该检测而非抛错。"""
        # x 全在右边缘外（upscale 空间 > 2560）：clamp 后 cx1 == cx2 == 1280
        oob = [(2570, 10), (2580, 10), (5000, 710), (4990, 710)]
        det = detector([(oob, "测试文本", 0.9)])
        results = det.detect(_frame(), frame_index=0, timestamp_ns=0)

        assert results == []

    def test_inbounds_bbox_passthrough(self, detector):
        """界内 bbox 正常返回，值域合法。"""
        inside = [(100, 100), (200, 100), (300, 300), (200, 300)]
        det = detector([(inside, "正常文本", 0.95)])
        results = det.detect(_frame(), frame_index=0, timestamp_ns=0)

        assert len(results) == 1
        for value in results[0].bbox_xyxy:
            assert 0.0 <= value <= 1.0
        assert results[0].text == "正常文本"
