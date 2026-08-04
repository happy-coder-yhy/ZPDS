"""场景分割流水线的轻量 Protocol；不导入任何模型运行时。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np

from zpds.scene.schemas import (
    BoundaryScore,
    DetectorFrameScores,
    SceneProposal,
    TransitionProposal,
    VLMReviewResult,
)


@runtime_checkable
class TransitionDetector(Protocol):
    """Stage A 单检测器的公共接口。输入帧使用 OpenCV BGR 排列。"""

    @property
    def source(self) -> str: ...

    def score_frames(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
    ) -> DetectorFrameScores: ...

    def detect(
        self,
        frames: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
    ) -> list[TransitionProposal]: ...


@runtime_checkable
class SemanticEmbedder(Protocol):
    """Stage B 的单一轻量语义 embedding 接口。"""

    @property
    def embedding_dimension(self) -> int: ...

    def embed(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray: ...

    def score_boundaries(
        self,
        frames_rgb: Sequence[np.ndarray],
        *,
        frame_indices: Sequence[int],
        timestamps_ns: Sequence[int],
    ) -> list[BoundaryScore]: ...


@runtime_checkable
class BoundaryFusion(Protocol):
    """Stage A/B 候选定稿接口。"""

    def fuse(
        self,
        transitions: Sequence[TransitionProposal],
        semantic_boundaries: Sequence[BoundaryScore],
        *,
        start_ns: int,
        end_ns: int,
        fps: float,
    ) -> list[SceneProposal]: ...


@runtime_checkable
class VLMReviewer(Protocol):
    """人员 B 实现的 VLM 场景-动作一致性复核接口。"""

    def review(
        self,
        scene: SceneProposal,
        representative_frames_rgb: Sequence[np.ndarray],
    ) -> VLMReviewResult: ...


__all__ = [
    "BoundaryFusion",
    "SemanticEmbedder",
    "TransitionDetector",
    "VLMReviewer",
]
