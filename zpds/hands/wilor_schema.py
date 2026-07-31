"""WiLoR 专属数据结构。

本模块只定义 WiLoR 模型特有的原始检测、坐标变换、配置、模型元信息
和 3D/MANO 结构。模型无关的公共类型（InferenceStatus, ModelAttemptResult,
HandFrameResult）请使用 :mod:`zpds.hands.schemas`。

延迟导入原则：
    本模块和 :mod:`zpds.hands.backends.wilor` 均不在模块顶层导入
    ``torch``、``wilor`` 或任何 WiLoR 上游依赖。所有可选依赖均在
    ``WiLoRBackend.__init__`` 内部动态导入，确保仅 ``import zpds``
    不会因缺少 PyTorch/WiLoR 而失败。
"""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# ════════════════════════════════════════════════════════════════════
# 异常
# ════════════════════════════════════════════════════════════════════


# ════════════════════════════════════════════════════════════════════
# WiLoR 异常层级
# ════════════════════════════════════════════════════════════════════


class WiLoRError(RuntimeError):
    """WiLoR 模块所有异常的基类。

    子类按粒度分为：
    - 运行级（应终止本次 WiLoR run）
    - 单帧级（可记录后继续下一帧）
    """


class WiLoRUnavailableError(WiLoRError):
    """当前环境未安装 WiLoR 所需依赖（PyTorch / wiLoR 包等）。"""


class CheckpointIntegrityError(WiLoRError):
    """WiLoR checkpoint SHA-256 校验失败，拒绝运行。"""


class WiLoRInitializationError(WiLoRError):
    """WiLoR 模型初始化失败（运行级，应终止本次 run）。"""


class WiLoRInferenceError(WiLoRError):
    """WiLoR 单帧推理失败（通常为单帧级，可继续下一帧）。"""


class WiLoROutputFormatError(WiLoRError):
    """WiLoR 原始输出格式不符合预期（可能为单帧级或运行级）。"""


class CoordinateTransformError(WiLoRError):
    """坐标逆变换失败（BBox 或关键点包含 NaN/Inf、越界等）。"""


class JointMappingError(WiLoRError):
    """21 点映射失败（映射版本不兼容或关节数不匹配）。"""


class InvalidDetectionError(ValueError):
    """WiLoR 检测结果不合法（BBox NaN/Inf、坐标顺序错误、超出原图范围等）。"""


# 单帧级 vs 运行级分类
FRAME_LEVEL_ERRORS = (
    WiLoRInferenceError,
    WiLoROutputFormatError,
    CoordinateTransformError,
    JointMappingError,
    InvalidDetectionError,
)
"""单帧级异常：记录后可继续下一帧。"""


RUN_LEVEL_ERRORS = (
    WiLoRUnavailableError,
    CheckpointIntegrityError,
    WiLoRInitializationError,
)
"""运行级异常：应终止本次 WiLoR run，不得逐帧 fallback。"""


# ════════════════════════════════════════════════════════════════════
# 配置
# ════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class WiLoRConfig:
    """WiLoR 后端初始化配置。

    所有路径在构造前应由调用方解析为绝对路径。
    """

    checkpoint_path: str = ""
    expected_sha256: str = ""

    wilor_source_path: str = ""  # WiLoR 源码目录（绝对路径）
    detector_path: str = ""      # YOLO detector.pt 路径
    model_config_path: str = ""  # model_config.yaml 路径

    device: str = "cpu"  # "cpu" | "cuda" | "cuda:0"
    precision: str = "float32"  # "float32" | "float16"

    model_version: str = ""

    # 上游溯源（固定后不可随意更改）
    upstream_repository: str = ""
    upstream_git_commit: str = ""
    upstream_license_checked: bool = False

    def __post_init__(self) -> None:
        if self.device not in {"cpu", "cuda"} and not self.device.startswith("cuda:"):
            raise ValueError(
                f"device 必须是 cpu、cuda 或 cuda:N，实际 {self.device!r}"
            )
        if self.precision not in {"float32", "float16"}:
            raise ValueError(
                f"precision 必须是 float32 或 float16，实际 {self.precision!r}"
            )
        if not self.model_version.strip():
            raise ValueError("model_version 不能为空")


# ════════════════════════════════════════════════════════════════════
# 模型运行时元信息
# ════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class WiLoRModelInfo:
    """WiLoR 初始化后采集的完整运行时元信息。

    应在 ``WiLoRBackend.__init__`` 成功完成后生成，
    后续写入 run report 和 Experience manifest。
    """

    model_name: str = "wilor"
    model_version: str = ""

    upstream_git_commit: str = ""
    checkpoint_path: str = ""
    checkpoint_sha256: str = ""
    checkpoint_size_bytes: int = 0

    device: str = "cpu"
    precision: str = "float32"

    python_version: str = ""
    torch_version: str = ""
    cuda_version: str | None = None
    gpu_name: str | None = None

    init_time_ms: float = 0.0

    @classmethod
    def from_config(
        cls,
        config: WiLoRConfig,
        *,
        torch_version: str = "",
        cuda_version: str | None = None,
        gpu_name: str | None = None,
        init_time_ms: float = 0.0,
    ) -> "WiLoRModelInfo":
        """从配置和运行时环境采集元信息，自动计算 checkpoint SHA-256。

        Raises:
            FileNotFoundError: checkpoint 文件不存在。
            CheckpointIntegrityError: SHA-256 不匹配。
        """
        path = Path(config.checkpoint_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"WiLoR checkpoint 不存在: {path.resolve()}"
            )

        size = path.stat().st_size
        actual_sha256 = _sha256_file(path)

        if config.expected_sha256:
            if actual_sha256 != config.expected_sha256:
                raise CheckpointIntegrityError(
                    f"WiLoR checkpoint SHA-256 不匹配。\n"
                    f"  期望: {config.expected_sha256}\n"
                    f"  实际: {actual_sha256}\n"
                    f"  路径: {path.resolve()}"
                )

        return cls(
            model_name="wilor",
            model_version=config.model_version,
            upstream_git_commit=config.upstream_git_commit,
            checkpoint_path=str(path.resolve()),
            checkpoint_sha256=actual_sha256,
            checkpoint_size_bytes=size,
            device=config.device,
            precision=config.precision,
            python_version=sys.version.split()[0],
            torch_version=torch_version,
            cuda_version=cuda_version,
            gpu_name=gpu_name,
            init_time_ms=init_time_ms,
        )


def _sha256_file(path: Path) -> str:
    """流式计算文件 SHA-256。"""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(slots=True)
class WiLoRImageTransform:
    """WiLoR 预处理阶段对输入图像执行的完整变换链。

    记录从 Prepared 原图到模型输入每一步的缩放和填充参数，
    用于将模型输出逆变换回原图像素坐标。

    变换链（正向）：:

        Prepared 原图 (W_orig × H_orig)
            ↓ resize
        detector 输入 (detector_width × detector_height)
            ↓ letterbox padding
        letterbox 后图像
            ↓ 检测 BBox + BBox padding
            ↓ WiLoR crop + crop resize / normalize
        模型输入 (model_input_size × model_input_size)

    逆变换将检测器输出的 BBox 映射回原图。
    """

    original_width: int
    original_height: int

    detector_width: int  # 检测器 resize 目标宽度
    detector_height: int  # 检测器 resize 目标高度

    resize_scale_x: float  # detector_width / original_width
    resize_scale_y: float  # detector_height / original_height

    letterbox_left: float  # 水平 padding（添加到 resize 后图像左侧）
    letterbox_top: float   # 垂直 padding（添加到 resize 后图像顶部）

    # WiLoR crop 参数（用于关键点逆变换，阶段 4+）
    crop_x1: float = 0.0  # crop 区域在原图的左上角 X
    crop_y1: float = 0.0  # crop 区域在原图的左上角 Y
    crop_width: float = 0.0   # crop 区域在原图的宽度
    crop_height: float = 0.0  # crop 区域在原图的高度
    crop_input_width: float = 256.0   # crop 被 resize 到的模型输入宽度
    crop_input_height: float = 256.0  # crop 被 resize 到的模型输入高度

    is_padded: bool = False
    maintain_aspect: bool = True  # 是否保持宽高比

    @classmethod
    def from_resize(
        cls,
        *,
        original_width: int,
        original_height: int,
        detector_width: int,
        detector_height: int,
        letterbox_left: float = 0.0,
        letterbox_top: float = 0.0,
        maintain_aspect: bool = True,
    ) -> "WiLoRImageTransform":
        """从 resize 参数构造变换记录。

        Args:
            original_width: 原图宽度（像素）。
            original_height: 原图高度（像素）。
            detector_width: 检测器 resize 目标宽度。
            detector_height: 检测器 resize 目标高度。
            letterbox_left: 水平 letterbox padding。
            letterbox_top: 垂直 letterbox padding。
            maintain_aspect: 是否保持宽高比。

        Returns:
            WiLoRImageTransform 实例。
        """
        return cls(
            original_width=original_width,
            original_height=original_height,
            detector_width=detector_width,
            detector_height=detector_height,
            resize_scale_x=detector_width / original_width,
            resize_scale_y=detector_height / original_height,
            letterbox_left=letterbox_left,
            letterbox_top=letterbox_top,
            is_padded=letterbox_left > 0 or letterbox_top > 0,
            maintain_aspect=maintain_aspect,
        )


@dataclass(slots=True)
class WiLoRDetection:
    """WiLoR 原始 2D 检测结果（未映射到 21 点公共约定前）。

    所有 BBox 坐标为原图像素坐标（已做完逆变换）。
    在 ``WILOR_TO_HANDS_V1`` 关节映射未验收前，
    不转换为 :class:`RawHandResult`。
    """

    handedness: str  # "Left" | "Right" | "Unknown"
    handedness_score: float  # [0, 1]
    detection_score: float  # [0, 1]

    bbox_xyxy_px: tuple[float, float, float, float]  # 原图像素坐标

    raw_keypoints_2d: np.ndarray | None = None  # (N, 2) WiLoR crop 空间关节
    raw_keypoint_format: str | None = None  # "wilor_model_crop" 等

    # 3D 关键点 + 相机参数（WiLoR 完整推理输出，可选）
    raw_keypoints_3d: np.ndarray | None = None  # (N, 3) MANO 顺序
    cam_t: np.ndarray | None = None  # (3,) 相机平移（pred_cam_t，crop 空间）
    focal: np.ndarray | None = None  # (2,) 焦距

    # batch context — cam_crop_to_full 所需参数
    pred_cam: np.ndarray | None = None  # (3,) weak-perspective camera (s, tx, ty)
    box_center: np.ndarray | None = None  # (2,) BBox 中心在原图的坐标
    box_size: float | None = None  # BBox 尺寸（含 rescale_factor）
    scaled_focal_length: float | None = None  # FOCAL_LENGTH * img_size / IMAGE_SIZE

    clipped: bool = False
    transform: WiLoRImageTransform | None = None


# ════════════════════════════════════════════════════════════════════
# 回退策略配置
# ════════════════════════════════════════════════════════════════════


@dataclass(slots=True)
class WiLoRFallbackPolicy:
    """WiLoR → MediaPipe 的回退和对照策略。

    每项独立控制，不会互相覆盖。
    """

    on_wilor_init_failure: bool = True
    on_wilor_frame_failure: bool = True
    on_wilor_no_hand: bool = False
    on_invalid_input: bool = False

    compare_with_mediapipe: bool = False

    def __post_init__(self) -> None:
        if self.compare_with_mediapipe and (
            self.on_wilor_frame_failure
            or self.on_wilor_no_hand
            or self.on_invalid_input
        ):
            raise ValueError(
                "compare_with_mediapipe=True 时不应同时启用回退开关。"
                "对照模式是同时运行两个模型以比较结果，"
                "回退模式是 WiLoR 失败后才调用 MediaPipe。"
                "两者语义互斥。"
            )


@dataclass(slots=True)
class WiLoRRunThresholds:
    """运行级失败阈值。

    当连续失败或失败比例超过阈值时，终止本次 WiLoR run。
    """

    max_consecutive_frame_failures: int = 5
    max_failure_ratio: float = 0.02  # allowed_failed / total_frames

    def __post_init__(self) -> None:
        if self.max_consecutive_frame_failures < 1:
            raise ValueError("max_consecutive_frame_failures 必须 >= 1")
        if not 0.0 < self.max_failure_ratio <= 1.0:
            raise ValueError("max_failure_ratio 必须在 (0, 1] 范围内")


@dataclass
class WiLoRRunReport:
    """单次 WiLoR run 的完整报告。

    包含运行覆盖、计时、回退、质量和完成状态。
    """

    requested_model: str = "wilor"
    ego_bbox_every_frame: bool = False

    model: WiLoRModelInfo | None = None

    coverage: dict = field(default_factory=dict)
    fallback: dict = field(default_factory=dict)
    effective_output: dict = field(default_factory=dict)
    timing: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    completion: dict = field(default_factory=dict)
    errors: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """转换为可 JSON 序列化的字典。"""
        return {
            "requested_model": self.requested_model,
            "ego_bbox_every_frame": self.ego_bbox_every_frame,
            "model": {
                "name": self.model.model_name if self.model else "wilor",
                "version": self.model.model_version if self.model else "",
                "checkpoint_sha256": self.model.checkpoint_sha256 if self.model else "",
                "device": self.model.device if self.model else "",
                "precision": self.model.precision if self.model else "",
                "torch_version": self.model.torch_version if self.model else "",
                "cuda_version": self.model.cuda_version if self.model else "",
                "gpu_name": self.model.gpu_name if self.model else "",
            },
            "coverage": self.coverage,
            "fallback": self.fallback,
            "effective_output": self.effective_output,
            "timing": self.timing,
            "quality": self.quality,
            "completion": self.completion,
            "errors": self.errors,
        }


__all__ = [
    "CheckpointIntegrityError",
    "CoordinateTransformError",
    "FRAME_LEVEL_ERRORS",
    "InvalidDetectionError",
    "JointMappingError",
    "RUN_LEVEL_ERRORS",
    "WiLoRConfig",
    "WiLoRDetection",
    "WiLoRError",
    "WiLoRFallbackPolicy",
    "WiLoRImageTransform",
    "WiLoRInitializationError",
    "WiLoRInferenceError",
    "WiLoRModelInfo",
    "WiLoROutputFormatError",
    "WiLoRRunReport",
    "WiLoRRunThresholds",
    "WiLoRUnavailableError",
]
