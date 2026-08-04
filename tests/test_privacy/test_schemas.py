"""测试 Schema 校验与构造。"""

import pytest

from zpds.privacy.schemas import (
    FaceDetection,
    PIIClassification,
    PrivacyFrameResult,
    PrivacyRunManifest,
    RedactionRegion,
    TextDetection,
)


class TestFaceDetection:
    def test_valid(self):
        fd = FaceDetection(0, 0, (0.1, 0.1, 0.5, 0.5), 0.9)
        assert fd.frame_index == 0
        assert fd.confidence == 0.9
        assert fd.backend == "yolo11n_face"

    def test_negative_frame_index(self):
        with pytest.raises(ValueError, match="frame_index"):
            FaceDetection(-1, 0, (0.1, 0.1, 0.5, 0.5), 0.9)

    def test_bbox_out_of_range(self):
        with pytest.raises(ValueError, match="xyxy"):
            FaceDetection(0, 0, (1.5, 0.1, 0.5, 0.5), 0.9)

    def test_confidence_out_of_range(self):
        with pytest.raises(ValueError, match="confidence"):
            FaceDetection(0, 0, (0.1, 0.1, 0.5, 0.5), 1.5)

    def test_x1_greater_than_x2(self):
        with pytest.raises(ValueError, match="xyxy"):
            FaceDetection(0, 0, (0.5, 0.1, 0.1, 0.5), 0.9)


class TestTextDetection:
    def test_valid(self):
        td = TextDetection(0, 0, (0.1, 0.1, 0.5, 0.5), "hello", 0.8)
        assert td.text == "hello"
        assert td.detector == "easyocr"

    def test_empty_backend(self):
        with pytest.raises(ValueError, match="detector"):
            TextDetection(0, 0, (0.1, 0.1, 0.5, 0.5), "test", 0.8, detector="")


class TestPIIClassification:
    def test_valid(self):
        td = TextDetection(0, 0, (0.1, 0.1, 0.5, 0.5), "13800138000", 0.9)
        pii = PIIClassification(td, "phone", "mask", 0.95)
        assert pii.category == "phone"
        assert pii.decision == "mask"

    def test_invalid_category(self):
        td = TextDetection(0, 0, (0.1, 0.1, 0.5, 0.5), "test", 0.9)
        with pytest.raises(ValueError, match="category"):
            PIIClassification(td, "not_a_category", "mask", 0.9)

    def test_invalid_decision(self):
        td = TextDetection(0, 0, (0.1, 0.1, 0.5, 0.5), "test", 0.9)
        with pytest.raises(ValueError, match="decision"):
            PIIClassification(td, "phone", "delete", 0.9)


class TestRedactionRegion:
    def test_valid_face(self):
        rr = RedactionRegion("face", (0.1, 0.1, 0.5, 0.5), "blur", "face", 0.9)
        assert rr.kind == "face"
        assert rr.method == "blur"

    def test_valid_text(self):
        rr = RedactionRegion("text", (0.1, 0.1, 0.5, 0.5), "black_rect", "phone", 0.9)
        assert rr.kind == "text"

    def test_invalid_kind(self):
        with pytest.raises(ValueError, match="kind"):
            RedactionRegion("hand", (0.1, 0.1, 0.5, 0.5), "blur", "hand", 0.9)

    def test_invalid_method(self):
        with pytest.raises(ValueError, match="method"):
            RedactionRegion("face", (0.1, 0.1, 0.5, 0.5), "invert", "face", 0.9)


class TestPrivacyFrameResult:
    def test_valid(self):
        fd = FaceDetection(0, 0, (0.1, 0.1, 0.5, 0.5), 0.9)
        pfr = PrivacyFrameResult(0, 0, faces=(fd,))
        assert len(pfr.faces) == 1

    def test_invalid_face_in_faces(self):
        with pytest.raises(TypeError):
            PrivacyFrameResult(0, 0, faces=("not_a_face",))


class TestPrivacyRunManifest:
    def test_valid(self):
        m = PrivacyRunManifest("s1", "/tmp/v.mp4", "guida_ego")
        assert m.session_id == "s1"
        assert m.producer == "zpds.privacy"

    def test_empty_session_id(self):
        with pytest.raises(ValueError, match="session_id"):
            PrivacyRunManifest("", "/tmp/v.mp4", "guida_ego")
