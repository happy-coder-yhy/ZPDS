"""Hands 双模型编排使用的轻量公共契约。

本模块不能导入 MediaPipe、PyTorch 或 WiLoR。人员 A 的 Pipeline、人员 B 的
模型适配器以及人员 C 的 Writer 仅通过这些类型交换逐帧结果。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import numpy as np

from zpds.hands.schemas import HandFrameResult, PreparedFrame, RawHandResult

InferenceStatus = Literal[
    "detected",
    "no_hand",
    "failed",
    "skipped_invalid_input",
]
VALID_INFERENCE_STATUSES = frozenset(
    {"detected", "no_hand", "failed", "skipped_invalid_input"}
)


@runtime_checkable
class HandEstimator(Protocol):
    """人员 B 向人员 A 提供的最小模型接口。"""

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        """执行单帧推理；无检测时返回空列表。"""

    def close(self) -> None:
        """释放模型、GPU 和第三方运行时资源。"""


@runtime_checkable
class FrameStatusHandEstimator(HandEstimator, Protocol):
    """可向人员 A 保留模型原始逐帧状态的扩展估计器接口。"""

    def estimate_frame(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> HandFrameResult:
        """返回 detected/no-hand/failed 等结构化模型状态。"""


@dataclass(frozen=True)
class FrameInferenceRecord:
    """人员 A 为每个 Prepared 输出帧生成的一条推理记录。"""

    frame: PreparedFrame
    inference_status: InferenceStatus
    raw_hands: tuple[RawHandResult, ...] = ()
    effective_hands: tuple[RawHandResult, ...] = ()
    frame_result: HandFrameResult | None = None
    failure_reason: str | None = None
    active_backend: str = ""
    inference_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.inference_status not in VALID_INFERENCE_STATUSES:
            raise ValueError(
                f"非法 inference_status: {self.inference_status!r}"
            )
        if not self.active_backend.strip():
            raise ValueError("active_backend 不能为空")
        if not math.isfinite(self.inference_ms) or self.inference_ms < 0:
            raise ValueError("inference_ms 必须是非负有限数值")
        if any(not isinstance(hand, RawHandResult) for hand in self.raw_hands):
            raise TypeError("raw_hands 必须全部是 RawHandResult")
        if any(
            not isinstance(hand, RawHandResult)
            for hand in self.effective_hands
        ):
            raise TypeError("effective_hands 必须全部是 RawHandResult")

        if self.inference_status == "detected" and not self.raw_hands:
            raise ValueError("detected 状态必须至少包含一只手")
        if self.inference_status != "detected" and self.raw_hands:
            raise ValueError(
                f"{self.inference_status} 状态不能包含手部结果"
            )
        if self.inference_status in {"failed", "skipped_invalid_input"}:
            if not (self.failure_reason or "").strip():
                raise ValueError(
                    f"{self.inference_status} 状态必须提供 failure_reason"
                )
        elif self.failure_reason is not None:
            raise ValueError(
                f"{self.inference_status} 状态不应提供 failure_reason"
            )


@dataclass
class RunFrameStatistics:
    """可直接写入 run manifest 的全帧计数器。"""

    requested: int = 0
    detected: int = 0
    no_hand: int = 0
    failed: int = 0
    skipped_invalid_input: int = 0

    def add(self, record: FrameInferenceRecord) -> None:
        self.requested += 1
        setattr(
            self,
            record.inference_status,
            getattr(self, record.inference_status) + 1,
        )

    @property
    def accounted(self) -> int:
        return (
            self.detected
            + self.no_hand
            + self.failed
            + self.skipped_invalid_input
        )

    @property
    def is_complete(self) -> bool:
        return self.requested == self.accounted

    def to_manifest(self) -> dict[str, int]:
        return {
            "requested": self.requested,
            "detected": self.detected,
            "no_hand": self.no_hand,
            "failed": self.failed,
            "skipped_invalid_input": self.skipped_invalid_input,
        }


@runtime_checkable
class FrameStatusWriter(Protocol):
    """人员 C 的逐帧状态 Writer 最小接口。"""

    def write(self, record: FrameInferenceRecord) -> None: ...

    def close(self) -> None: ...


@runtime_checkable
class BBoxWriter(Protocol):
    """人员 C 的 WiLoR BBox Writer 最小接口。"""

    def write(self, record: FrameInferenceRecord) -> None: ...

    def close(self) -> None: ...


__all__ = [
    "VALID_INFERENCE_STATUSES",
    "BBoxWriter",
    "FrameInferenceRecord",
    "FrameStatusHandEstimator",
    "FrameStatusWriter",
    "HandEstimator",
    "InferenceStatus",
    "RunFrameStatistics",
]
