"""Hands 编排层的轻量测试桩。

这些测试桩不依赖 MediaPipe、PyTorch 或 WiLoR，可用于人员 A 在真实模型和
正式 Writer 就绪前开发 Pipeline、CLI 与批处理。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Self

import numpy as np

from zpds.hands.contracts import FrameInferenceRecord
from zpds.hands.schemas import RawHandResult


class FakeHandEstimator:
    """按预设响应顺序返回结果或抛出异常的模型测试桩。"""

    def __init__(self, responses: Sequence[list[RawHandResult] | Exception]) -> None:
        self._responses = list(responses)
        self.frames: list[np.ndarray] = []
        self.timestamps_ms: list[int] = []
        self.closed = False

    def estimate(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: int,
    ) -> list[RawHandResult]:
        if self.closed:
            raise RuntimeError("FakeHandEstimator 已关闭")
        response_index = len(self.timestamps_ms)
        if response_index >= len(self._responses):
            raise RuntimeError("FakeHandEstimator 没有更多预设响应")
        self.frames.append(frame_rgb)
        self.timestamps_ms.append(timestamp_ms)
        response = self._responses[response_index]
        if isinstance(response, Exception):
            raise response
        return response

    def close(self) -> None:
        self.closed = True

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class FakeFrameStatusWriter:
    """在内存中收集所有逐帧状态。"""

    def __init__(self) -> None:
        self.records: list[FrameInferenceRecord] = []
        self.closed = False

    def write(self, record: FrameInferenceRecord) -> None:
        if self.closed:
            raise RuntimeError("FakeFrameStatusWriter 已关闭")
        self.records.append(record)

    def close(self) -> None:
        self.closed = True


class FakeBBoxWriter:
    """在内存中收集 detected 记录中的 BBox 结果。"""

    def __init__(self) -> None:
        self.records: list[FrameInferenceRecord] = []
        self.closed = False

    @property
    def bbox_count(self) -> int:
        return sum(len(record.raw_hands) for record in self.records)

    def write(self, record: FrameInferenceRecord) -> None:
        if self.closed:
            raise RuntimeError("FakeBBoxWriter 已关闭")
        if record.inference_status == "detected":
            self.records.append(record)

    def close(self) -> None:
        self.closed = True


__all__ = [
    "FakeBBoxWriter",
    "FakeFrameStatusWriter",
    "FakeHandEstimator",
]
