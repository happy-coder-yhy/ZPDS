"""Hands V1 跨模块共享的数据契约。

本模块只定义数据结构和与后端无关的校验规则：

- 人员 B 的模型适配器输出 :class:`RawHandResult`
- 人员 A 的流水线输出 :class:`HandObservation`
- 人员 C 的 Writer 消费 :class:`HandObservation`

坐标和时间约定：

- ``bbox_xyxy`` 和 ``keypoints_2d`` 均为输出 RGB 帧上的绝对像素坐标。
- ``keypoints_z_relative`` 保留模型提供的相对深度尺度。
- 所有持久化时间戳均为整数纳秒；模型适配器可在调用边界转换为毫秒。
- ``HandObservation.handedness`` 固定为小写 ``left/right/unknown``。
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np

Handedness = Literal["left", "right", "unknown"]
HAND_KEYPOINT_COUNT = 21
VALID_HANDEDNESS = frozenset({"left", "right", "unknown"})


def _require_finite(values: list[float] | tuple[float, ...], field_name: str) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError(f"{field_name} 必须全部为有限数值")


@dataclass
class HandKeypoints:
    """单只手的 21 个关键点（MediaPipe Hand Landmarks 拓扑）。

    ``normalized`` 保存模型归一化 ``(x, y, z)``，``pixel`` 保存输出帧上的
    绝对像素 ``(x, y)``。该类型属于模型层原始结果，不直接作为 Parquet 行。
    """

    normalized: list[tuple[float, float, float]]
    pixel: list[tuple[float, float]]
    has_visibility: bool = False
    visibility: list[float] = field(default_factory=list)
    any_clipped: bool = False
    clipped_count: int = 0

    def __post_init__(self) -> None:
        if len(self.normalized) != HAND_KEYPOINT_COUNT:
            raise ValueError(f"关键点数量应为 {HAND_KEYPOINT_COUNT}，实际 {len(self.normalized)}")
        if len(self.pixel) != HAND_KEYPOINT_COUNT:
            raise ValueError(f"像素关键点数量应为 {HAND_KEYPOINT_COUNT}，实际 {len(self.pixel)}")
        if self.has_visibility and len(self.visibility) != HAND_KEYPOINT_COUNT:
            raise ValueError(
                f"visibility 数量应为 {HAND_KEYPOINT_COUNT}，实际 {len(self.visibility)}"
            )


@dataclass
class HandBBox:
    """模型层手部边界框，采用输出帧绝对像素 ``xyxy`` 坐标。"""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 0.0
    is_padded: bool = False
    padding_ratio: float = 0.0

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def is_valid(self) -> bool:
        return self.area > 0 and self.x1 >= 0 and self.y1 >= 0


@dataclass
class RawHandResult:
    """模型适配器的一次单手检测结果。

    这是人员 B 向人员 A 提供的接口。``handedness`` 保留模型原始标签；
    Pipeline 转换为 :class:`HandObservation` 时再规范为小写枚举。
    """

    handedness: str
    handedness_score: float
    keypoints: HandKeypoints
    bbox: HandBBox
    detection_score: float = 0.0
    label: str = ""

    @classmethod
    def from_mediapipe(
        cls,
        hand_landmarks,
        handedness,
        image_width: int,
        image_height: int,
        bbox_padding_ratio: float = 0.10,
        hand_index: int = 0,
    ) -> RawHandResult:
        """从 MediaPipe HandLandmarker 单帧输出构造统一原始结果。"""
        normalized: list[tuple[float, float, float]] = []
        pixel: list[tuple[float, float]] = []
        visibility: list[float] = []
        has_visibility = False
        xs: list[float] = []
        ys: list[float] = []

        for landmark in hand_landmarks:
            nx = float(landmark.x)
            ny = float(landmark.y)
            nz = float(landmark.z)
            normalized.append((nx, ny, nz))

            px = nx * image_width
            py = ny * image_height
            px_clipped = max(0.0, min(float(image_width - 1), px))
            py_clipped = max(0.0, min(float(image_height - 1), py))
            pixel.append((px_clipped, py_clipped))
            xs.append(px_clipped)
            ys.append(py_clipped)

            if hasattr(landmark, "visibility") and landmark.visibility is not None:
                has_visibility = True
                visibility.append(float(landmark.visibility))
            elif hasattr(landmark, "visibility"):
                visibility.append(0.0)
            else:
                visibility.append(1.0)

        px1, py1 = min(xs), min(ys)
        px2, py2 = max(xs), max(ys)

        if bbox_padding_ratio > 0:
            box_width = max(px2 - px1, 1.0)
            box_height = max(py2 - py1, 1.0)
            pad_width = box_width * bbox_padding_ratio
            pad_height = box_height * bbox_padding_ratio
            px1 = max(0.0, px1 - pad_width)
            py1 = max(0.0, py1 - pad_height)
            px2 = min(float(image_width), px2 + pad_width)
            py2 = min(float(image_height), py2 + pad_height)

        hand_score = float(handedness.score) if handedness.score else 0.0
        hand_label = handedness.category_name if handedness.category_name else "Unknown"
        clipped_count = sum(
            px <= 0
            or px >= image_width - 1
            or py <= 0
            or py >= image_height - 1
            for px, py in pixel
        )

        return cls(
            handedness=hand_label,
            handedness_score=hand_score,
            keypoints=HandKeypoints(
                normalized=normalized,
                pixel=pixel,
                has_visibility=has_visibility,
                visibility=visibility,
                any_clipped=clipped_count > 0,
                clipped_count=clipped_count,
            ),
            bbox=HandBBox(
                x1=px1,
                y1=py1,
                x2=px2,
                y2=py2,
                confidence=hand_score,
                is_padded=bbox_padding_ratio > 0,
                padding_ratio=bbox_padding_ratio,
            ),
            detection_score=hand_score,
            label=f"hand_{hand_index}",
        )


@dataclass
class BackendInfo:
    """实际启用的 MediaPipe 后端及 fallback 信息。"""

    requested_backend: str = ""
    active_backend: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    delegate: str = ""


@dataclass
class ModelInfo:
    """Tasks 模型文件的路径、哈希和大小信息。"""

    path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    download_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    exists: bool = False

    @classmethod
    def from_file(cls, model_path: str | Path) -> ModelInfo:
        path = Path(model_path)
        if not path.is_file():
            return cls(path=str(path.resolve()), exists=False)
        return cls(
            path=str(path.resolve()),
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            size_bytes=path.stat().st_size,
            exists=True,
        )


@dataclass
class SessionStats:
    """单次 MediaPipe 推理会话统计。"""

    total_frames: int = 0
    empty_frames: int = 0
    no_hand_frames: int = 0
    hand_frames: int = 0
    exception_frames: int = 0
    init_time_ms: float = 0.0
    total_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    model_info: ModelInfo | None = None
    backend_info: BackendInfo | None = None


@dataclass(frozen=True)
class PreparedFrame:
    """Prepared Segment Reader 产生的一帧模型输入。

    ``frame_rgb`` 固定为 ``uint8`` 的 ``H×W×3`` RGB 图像。输出帧时间使用
    Segment 相对纳秒时间；源帧信息在上游无法提供时允许为 ``None``。
    """

    frame_rgb: np.ndarray
    output_frame_index: int
    timestamp_ns: int
    source_frame_index: int | None
    source_timestamp_ns: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.frame_rgb, np.ndarray):
            raise TypeError("frame_rgb 必须是 numpy.ndarray")
        if self.frame_rgb.dtype != np.uint8:
            raise ValueError("frame_rgb dtype 必须是 uint8")
        if self.frame_rgb.ndim != 3 or self.frame_rgb.shape[2] != 3:
            raise ValueError("frame_rgb 必须是 H×W×3 图像")
        if self.frame_rgb.shape[0] <= 0 or self.frame_rgb.shape[1] <= 0:
            raise ValueError("frame_rgb 不能为空")
        if self.output_frame_index < 0:
            raise ValueError("output_frame_index 不能为负数")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负数")
        if self.source_frame_index is not None and self.source_frame_index < 0:
            raise ValueError("source_frame_index 不能为负数")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("source_timestamp_ns 不能为负数")


@dataclass(frozen=True)
class HandObservation:
    """Pipeline 产生、Writer 持久化的一条 Hands V1 观测。

    一条记录表示某个输出视频帧中检测到的一只手。无手帧不创建记录。
    """

    segment_id: str
    video_stream_id: str
    output_frame_index: int
    timestamp_ns: int
    source_frame_index: int | None
    source_timestamp_ns: int | None
    detection_id: int
    handedness: Handedness
    handedness_score: float
    bbox_xyxy: tuple[float, float, float, float]
    keypoints_2d: list[tuple[float, float]]
    keypoints_z_relative: list[float]
    model_name: str
    model_version: str

    def __post_init__(self) -> None:
        if not self.segment_id:
            raise ValueError("segment_id 不能为空")
        if not self.video_stream_id:
            raise ValueError("video_stream_id 不能为空")
        if self.output_frame_index < 0:
            raise ValueError("output_frame_index 不能为负数")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns 不能为负数")
        if self.source_frame_index is not None and self.source_frame_index < 0:
            raise ValueError("source_frame_index 不能为负数")
        if self.source_timestamp_ns is not None and self.source_timestamp_ns < 0:
            raise ValueError("source_timestamp_ns 不能为负数")
        if self.detection_id < 0:
            raise ValueError("detection_id 不能为负数")
        if self.handedness not in VALID_HANDEDNESS:
            raise ValueError(
                f"handedness 必须是 left、right 或 unknown，实际为 {self.handedness!r}"
            )
        if not 0.0 <= self.handedness_score <= 1.0:
            raise ValueError("handedness_score 必须在 [0, 1] 范围内")
        if len(self.bbox_xyxy) != 4:
            raise ValueError("bbox_xyxy 必须包含 4 个值")
        _require_finite(self.bbox_xyxy, "bbox_xyxy")
        x1, y1, x2, y2 = self.bbox_xyxy
        if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1:
            raise ValueError("bbox_xyxy 必须是合法的非负绝对像素 xyxy 坐标")
        if len(self.keypoints_2d) != HAND_KEYPOINT_COUNT:
            raise ValueError(f"keypoints_2d 必须包含 {HAND_KEYPOINT_COUNT} 个点")
        if len(self.keypoints_z_relative) != HAND_KEYPOINT_COUNT:
            raise ValueError(f"keypoints_z_relative 必须包含 {HAND_KEYPOINT_COUNT} 个值")
        for point in self.keypoints_2d:
            if len(point) != 2:
                raise ValueError("keypoints_2d 中的每个点必须包含 x、y 两个值")
            _require_finite(point, "keypoints_2d")
        _require_finite(self.keypoints_z_relative, "keypoints_z_relative")
        if not self.model_name:
            raise ValueError("model_name 不能为空")
        if not self.model_version:
            raise ValueError("model_version 不能为空")


__all__ = [
    "HAND_KEYPOINT_COUNT",
    "VALID_HANDEDNESS",
    "BackendInfo",
    "HandBBox",
    "HandKeypoints",
    "HandObservation",
    "Handedness",
    "ModelInfo",
    "PreparedFrame",
    "RawHandResult",
    "SessionStats",
]
