"""
手部检测统一数据结构。

定义 MediaPipe / WiLoR / HaWoR 等不同后端共用的输出类型，
使得下游代码不依赖具体检测器实现。
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class HandKeypoints:
    """单只手的 21 个关键点（MediaPipe Hand Landmarks 拓扑）。

    Attributes:
        normalized: 归一化坐标 (x, y, z)，各 21 个值。
            x/y ∈ [0, 1]，z 以手腕为原点，正值朝向相机。
        pixel: 像素坐标 (x, y)，21 个值。x ∈ [0, width)，y ∈ [0, height)。
        has_visibility: 是否存在 MediaPipe visibility 信息（归一化坐标预测置信度）。
        visibility: 21 个 visibility 值，仅在 has_visibility 时有效。
    """

    normalized: list[tuple[float, float, float]]   # 21× (x, y, z)
    pixel: list[tuple[float, float]]                # 21× (px_x, px_y)
    has_visibility: bool = False
    visibility: list[float] = field(default_factory=list)  # 21×
    any_clipped: bool = False                       # 是否有像素关键点被裁剪到图像边界
    clipped_count: int = 0                          # 被裁剪的关键点数量

    def __post_init__(self):
        if len(self.normalized) != 21:
            raise ValueError(f"关键点数量应为 21，实际 {len(self.normalized)}")
        if len(self.pixel) != 21:
            raise ValueError(f"像素关键点数量应为 21，实际 {len(self.pixel)}")
        if self.has_visibility and len(self.visibility) != 21:
            raise ValueError(f"visibility 数量应为 21，实际 {len(self.visibility)}")


@dataclass
class HandBBox:
    """手部边界框（像素坐标）。

    Attributes:
        x1, y1, x2, y2: 左上 / 右下像素坐标。
        confidence: 检测置信度（来自检测模型，非关键点回归）。
        is_padded: 是否已添加边距。
        padding_ratio: 使用的边距比例。
    """

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
        """BBox 面积 > 0 且坐标非负。"""
        return self.area > 0 and self.x1 >= 0 and self.y1 >= 0


@dataclass
class RawHandResult:
    """单只手的一次检测结果。

    Attributes:
        handedness: "Left" 或 "Right"（MediaPipe 原生标签）。
        handedness_score: 左右手分类置信度 [0, 1]。
        keypoints: 21 个关键点（归一化 + 像素）。
        bbox: 手部边界框。
        detection_score: 手掌检测模型置信度 [0, 1]。
        label: 实例标识（如 "hand_0", "hand_1"）。
    """

    handedness: str                     # "Left" | "Right"
    handedness_score: float             # [0, 1]
    keypoints: HandKeypoints
    bbox: HandBBox
    detection_score: float = 0.0        # [0, 1]
    label: str = ""

    # ---- 工厂方法 ----

    @classmethod
    def from_mediapipe(
        cls,
        hand_landmarks,
        handedness,
        image_width: int,
        image_height: int,
        bbox_padding_ratio: float = 0.10,
        hand_index: int = 0,
    ) -> "RawHandResult":
        """从 MediaPipe HandLandmarker 单帧输出构造 RawHandResult。

        Args:
            hand_landmarks: MediaPipe NormalizedLandmark 列表（21 个）。
            handedness: MediaPipe Category 对象。
            image_width: 图像宽度（像素）。
            image_height: 图像高度（像素）。
            bbox_padding_ratio: BBox 扩展边距比例。
            hand_index: 手在帧内的序号（用于 label）。

        Returns:
            RawHandResult 实例。
        """
        # ---- 归一化关键点 ----
        normalized: list[tuple[float, float, float]] = []
        pixel: list[tuple[float, float]] = []
        visibility: list[float] = []
        has_visibility = False

        xs, ys = [], []

        for lm in hand_landmarks:
            nx = float(lm.x)
            ny = float(lm.y)
            nz = float(lm.z)
            normalized.append((nx, ny, nz))

            px = nx * image_width
            py = ny * image_height

            # 裁剪到图像边界（保留原始预测 + 记录 clipped flag）
            px_clipped = max(0.0, min(float(image_width - 1), px))
            py_clipped = max(0.0, min(float(image_height - 1), py))

            pixel.append((px_clipped, py_clipped))
            xs.append(px_clipped)
            ys.append(py_clipped)

            # MediaPipe NormalizedLandmark 有 visibility 字段
            if hasattr(lm, "visibility") and lm.visibility is not None:
                has_visibility = True
                visibility.append(float(lm.visibility))
            elif hasattr(lm, "visibility"):
                visibility.append(0.0)
            else:
                visibility.append(1.0)

        # ---- BBox（从关键点计算 + 边距） ----
        px1, py1 = min(xs), min(ys)
        px2, py2 = max(xs), max(ys)

        if bbox_padding_ratio > 0:
            bw = max(px2 - px1, 1.0)
            bh = max(py2 - py1, 1.0)
            pad_w = bw * bbox_padding_ratio
            pad_h = bh * bbox_padding_ratio
            px1 = max(0.0, px1 - pad_w)
            py1 = max(0.0, py1 - pad_h)
            px2 = min(float(image_width), px2 + pad_w)
            py2 = min(float(image_height), py2 + pad_h)

        bbox = HandBBox(
            x1=px1, y1=py1, x2=px2, y2=py2,
            confidence=float(handedness.score) if handedness.score else 0.0,
            is_padded=bbox_padding_ratio > 0,
            padding_ratio=bbox_padding_ratio,
        )

        # ---- 左右手 ----
        hand_label = handedness.category_name if handedness.category_name else "Unknown"
        hand_score = float(handedness.score) if handedness.score else 0.0

        # ---- 检测关键点裁剪 ----
        clipped_count = 0
        for px, py in pixel:
            if px <= 0 or px >= image_width - 1 or py <= 0 or py >= image_height - 1:
                clipped_count += 1

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
            bbox=bbox,
            detection_score=hand_score,
            label=f"hand_{hand_index}",
        )


@dataclass
class BackendInfo:
    """后端运行时信息。

    记录本次推理实际使用的后端及 fallback 情况，
    便于事后分析不同后端的输出差异。

    Attributes:
        requested_backend: 配置请求的后端名称。
        active_backend: 实际激活的后端名称。
        fallback_used: 是否触发了 fallback。
        fallback_reason: fallback 原因（未触发时为空）。
        delegate: Tasks 后端使用的 delegate 类型。
    """

    requested_backend: str = ""
    active_backend: str = ""
    fallback_used: bool = False
    fallback_reason: str = ""
    delegate: str = ""


@dataclass
class ModelInfo:
    """模型文件元信息。

    Attributes:
        path: 模型文件路径。
        sha256: SHA-256 校验和。
        size_bytes: 文件大小。
        download_url: 官方下载地址。
        exists: 文件是否存在。
    """

    path: str = ""
    sha256: str = ""
    size_bytes: int = 0
    download_url: str = (
        "https://storage.googleapis.com/mediapipe-models/"
        "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    )
    exists: bool = False

    @classmethod
    def from_file(cls, model_path: str | Path) -> "ModelInfo":
        """从模型文件路径构造，自动计算 SHA-256 和文件大小。

        Args:
            model_path: .task 模型文件路径。

        Returns:
            ModelInfo，exists=False 时 sha256 和 size_bytes 为 0。
        """
        path = Path(model_path)
        if not path.is_file():
            return cls(
                path=str(path.resolve()),
                exists=False,
            )

        size = path.stat().st_size
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            path=str(path.resolve()),
            sha256=sha,
            size_bytes=size,
            exists=True,
        )


@dataclass
class SessionStats:
    """单次推理会话统计。

    Attributes:
        total_frames: 总推理帧数。
        empty_frames: 输入为空帧次数。
        no_hand_frames: 无手检测帧数。
        hand_frames: 检测到手帧数。
        exception_frames: 异常帧数。
        init_time_ms: 模型初始化耗时（毫秒）。
        total_inference_ms: 累计推理耗时（毫秒）。
        avg_inference_ms: 平均每帧推理耗时（毫秒）。
        model_info: 模型文件信息。
        backend_info: 后端运行信息。
    """

    total_frames: int = 0
    empty_frames: int = 0
    no_hand_frames: int = 0
    hand_frames: int = 0
    exception_frames: int = 0
    init_time_ms: float = 0.0
    total_inference_ms: float = 0.0
    avg_inference_ms: float = 0.0
    model_info: Optional[ModelInfo] = None
    backend_info: Optional[BackendInfo] = None
