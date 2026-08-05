"""Privacy 后端与流水线之间的轻量公共契约。

本模块不能导入 torch、ultralytics、easyocr 或任何模型后端。
只定义 Protocol 和数据记录类型。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from zpds.privacy.schemas import (
    FaceDetection,
    PIIClassification,
    PrivacyFrameResult,
    RedactionRegion,
    TextDetection,
)


# ---------------------------------------------------------------------------
# 后端 Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class FaceDetector(Protocol):
    """人脸检测后端最小接口。"""

    def detect(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        timestamp_ns: int,
    ) -> list[FaceDetection]:
        """检测单帧中的人脸；无检测时返回空列表。"""

    def close(self) -> None:
        """释放模型和 GPU 资源。"""


@runtime_checkable
class TextDetector(Protocol):
    """文本检测与 OCR 后端最小接口。"""

    def detect(
        self,
        frame_rgb: np.ndarray,
        frame_index: int,
        timestamp_ns: int,
    ) -> list[TextDetection]:
        """检测并识别单帧中的文本；无文本时返回空列表。"""

    def close(self) -> None:
        """释放模型和 GPU 资源。"""


@runtime_checkable
class PIIClassifier(Protocol):
    """PII 分类后端最小接口。"""

    def classify(
        self,
        texts: list[TextDetection],
    ) -> list[PIIClassification]:
        """对文本块做隐私分类；返回与输入一一对应的分类结果。"""

    def close(self) -> None:
        """释放 LLM 连接和缓存资源。"""


@runtime_checkable
class Redactor(Protocol):
    """遮挡执行器最小接口。"""

    def apply(
        self,
        frame_rgb: np.ndarray,
        regions: list[RedactionRegion],
    ) -> np.ndarray:
        """对单帧应用遮挡，返回处理后的 BGR/RGB 图像数组。"""


# ---------------------------------------------------------------------------
# 流水线数据记录（对标 hands/contracts.py 的 FrameInferenceRecord）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameRedactionRecord:
    """流水线为每个输出帧生成的一条脱敏记录。"""

    frame_index: int
    timestamp_ns: int

    # 检测结果
    faces: tuple[FaceDetection, ...] = ()
    texts: tuple[TextDetection, ...] = ()
    pii_classifications: tuple[PIIClassification, ...] = ()

    # 遮挡结果
    regions: tuple[RedactionRegion, ...] = ()
    redacted_frame: np.ndarray | None = None

    # 状态
    face_detector_used: str = ""
    text_detector_used: str = ""
    pii_classifier_used: str = ""
    llm_available: bool = False

    # 耗时
    face_inference_ms: float = 0.0
    text_inference_ms: float = 0.0
    pii_classification_ms: float = 0.0
    redaction_ms: float = 0.0

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
        for obj in self.pii_classifications:
            if not isinstance(obj, PIIClassification):
                raise TypeError("pii_classifications 必须全部是 PIIClassification")
        for obj in self.regions:
            if not isinstance(obj, RedactionRegion):
                raise TypeError("regions 必须全部是 RedactionRegion")
        if self.face_inference_ms < 0:
            raise ValueError("face_inference_ms 不能为负数")
        if self.text_inference_ms < 0:
            raise ValueError("text_inference_ms 不能为负数")
        if self.pii_classification_ms < 0:
            raise ValueError("pii_classification_ms 不能为负数")
        if self.redaction_ms < 0:
            raise ValueError("redaction_ms 不能为负数")


@dataclass
class RedactionRunStatistics:
    """全帧计数器，可直接写入 run_summary。"""

    frames_processed: int = 0
    frames_with_faces: int = 0
    frames_with_text: int = 0
    total_face_regions: int = 0
    total_text_regions: int = 0
    total_pii_masked: int = 0
    pii_categories_found: set[str] = field(default_factory=set)
    llm_available: bool = False
    elapsed_seconds: float = 0.0

    def add(self, record: FrameRedactionRecord) -> None:
        self.frames_processed += 1
        if record.faces:
            self.frames_with_faces += 1
            self.total_face_regions += len(record.faces)
        if record.texts:
            self.frames_with_text += 1
            self.total_text_regions += len(record.texts)
        self.total_pii_masked += sum(
            1 for p in record.pii_classifications if p.decision == "mask"
        )
        for p in record.pii_classifications:
            if p.decision == "mask":
                self.pii_categories_found.add(p.category)
        self.llm_available = record.llm_available

    @property
    def average_fps(self) -> float:
        if self.elapsed_seconds <= 0:
            return 0.0
        return self.frames_processed / self.elapsed_seconds


__all__ = [
    "FaceDetector",
    "FrameRedactionRecord",
    "PIIClassifier",
    "RedactionRunStatistics",
    "Redactor",
    "TextDetector",
]
