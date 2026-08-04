"""Privacy 模块跨模块共享的数据契约。

本模块只定义数据结构和与后端无关的校验规则，禁止导入 torch、ultralytics、easyocr。

坐标约定：
- ``bbox_xyxy`` 为归一化坐标 [0, 1]，左上角原点。
- ``timestamp_ns`` 为整数纳秒。
- ``frame_index`` 为 0-based 帧序号。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Literal

PrivacyCategory = Literal[
    "person_name",
    "id_card",
    "phone",
    "email",
    "date",
    "address",
    "place",
    "bank_card",
    "medical_record",
    "social_account",
    "license_plate",
    "unknown",
]

PrivacyDecision = Literal["mask", "keep", "review"]

RedactionMethod = Literal["blur", "pixelate", "black_rect"]

VALID_CATEGORIES = frozenset(
    {
        "person_name", "id_card", "phone", "email", "date",
        "address", "place", "bank_card", "medical_record",
        "social_account", "license_plate", "unknown",
    }
)
VALID_DECISIONS = frozenset({"mask", "keep", "review"})
VALID_METHODS = frozenset({"blur", "pixelate", "black_rect"})


def _require_finite(values: tuple[float, ...], field_name: str) -> None:
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError(f"{field_name} 必须全部为有限数值")


def _require_normalized_bbox(bbox: tuple[float, float, float, float], field_name: str) -> None:
    if len(bbox) != 4:
        raise ValueError(f"{field_name} 必须包含 4 个值")
    _require_finite(bbox, field_name)
    x1, y1, x2, y2 = bbox
    if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
        raise ValueError(
            f"{field_name} 必须是合法的归一化 xyxy 坐标 (0~1): {bbox}"
        )


# ---------------------------------------------------------------------------
# 检测结果
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceDetection:
    """单帧中的一个人脸检测结果。"""

    frame_index: int
    timestamp_ns: int
    bbox_xyxy: tuple[float, float, float, float]  # 归一化 0~1
    confidence: float
    backend: str = "yolo11n_face"

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index 不能为负数")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负数")
        _require_normalized_bbox(self.bbox_xyxy, "bbox_xyxy")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 范围内")
        if not self.backend.strip():
            raise ValueError("backend 不能为空")


@dataclass(frozen=True)
class TextDetection:
    """单帧中的一个文本区域检测结果。"""

    frame_index: int
    timestamp_ns: int
    bbox_xyxy: tuple[float, float, float, float]  # 归一化 0~1
    text: str
    confidence: float
    detector: str = "easyocr"

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index 不能为负数")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负数")
        _require_normalized_bbox(self.bbox_xyxy, "bbox_xyxy")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 范围内")
        if not self.detector.strip():
            raise ValueError("detector 不能为空")


@dataclass(frozen=True)
class PIIClassification:
    """一个文本块的 PII 分类结果。"""

    text: TextDetection
    category: PrivacyCategory
    decision: PrivacyDecision
    confidence: float
    classifier: str = "llm"

    def __post_init__(self) -> None:
        if self.category not in VALID_CATEGORIES:
            raise ValueError(f"非法 category: {self.category!r}")
        if self.decision not in VALID_DECISIONS:
            raise ValueError(f"非法 decision: {self.decision!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 范围内")
        if not self.classifier.strip():
            raise ValueError("classifier 不能为空")


@dataclass(frozen=True)
class RedactionRegion:
    """一个需要遮挡的区域（人脸或文本）。"""

    kind: Literal["face", "text"]
    bbox_xyxy: tuple[float, float, float, float]  # 归一化 0~1
    method: RedactionMethod
    category: str
    confidence: float

    def __post_init__(self) -> None:
        if self.kind not in ("face", "text"):
            raise ValueError(f"kind 必须是 face 或 text，实际: {self.kind!r}")
        _require_normalized_bbox(self.bbox_xyxy, "bbox_xyxy")
        if self.method not in VALID_METHODS:
            raise ValueError(f"非法 method: {self.method!r}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence 必须在 [0, 1] 范围内")


# ---------------------------------------------------------------------------
# 逐帧结果与运行产物
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrivacyFrameResult:
    """一帧的完整隐私检测与遮挡结果。"""

    frame_index: int
    timestamp_ns: int
    faces: tuple[FaceDetection, ...] = ()
    texts: tuple[TextDetection, ...] = ()
    pii: tuple[PIIClassification, ...] = ()
    regions: tuple[RedactionRegion, ...] = ()

    def __post_init__(self) -> None:
        if self.frame_index < 0:
            raise ValueError("frame_index 不能为负数")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负数")
        for obj in self.faces:
            if not isinstance(obj, FaceDetection):
                raise TypeError("faces 必须全部是 FaceDetection")
        for obj in self.texts:
            if not isinstance(obj, TextDetection):
                raise TypeError("texts 必须全部是 TextDetection")
        for obj in self.pii:
            if not isinstance(obj, PIIClassification):
                raise TypeError("pii 必须全部是 PIIClassification")
        for obj in self.regions:
            if not isinstance(obj, RedactionRegion):
                raise TypeError("regions 必须全部是 RedactionRegion")


@dataclass(frozen=True)
class PrivacyRunManifest:
    """一次脱敏运行的完整 manifest。"""

    session_id: str
    source_uri: str
    profile: str
    producer: str = "zpds.privacy"
    version: str = "v1"

    # 配置追溯
    config_hash: str = ""
    face_model_hash: str = ""
    llm_endpoint: str = ""

    # 统计
    total_frames: int = 0
    frames_with_faces: int = 0
    frames_with_text: int = 0
    total_face_regions: int = 0
    total_text_regions: int = 0
    pii_categories_found: tuple[str, ...] = ()
    llm_available: bool = False

    # 运行信息
    elapsed_seconds: float = 0.0
    error: str = ""

    def __post_init__(self) -> None:
        if not self.session_id.strip():
            raise ValueError("session_id 不能为空")
        if not self.source_uri.strip():
            raise ValueError("source_uri 不能为空")
        if not self.profile.strip():
            raise ValueError("profile 不能为空")
        if not self.producer.strip():
            raise ValueError("producer 不能为空")
        if not self.version.strip():
            raise ValueError("version 不能为空")
        if self.total_frames < 0:
            raise ValueError("total_frames 不能为负数")
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds 不能为负数")


__all__ = [
    "VALID_CATEGORIES",
    "VALID_DECISIONS",
    "VALID_METHODS",
    "FaceDetection",
    "PIIClassification",
    "PrivacyCategory",
    "PrivacyDecision",
    "PrivacyFrameResult",
    "PrivacyRunManifest",
    "RedactionMethod",
    "RedactionRegion",
    "TextDetection",
]
