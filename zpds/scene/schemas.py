"""场景分割与 VLM 复核共享的数据契约。"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal

TRANSITION_SOURCES = frozenset(
    {"histogram", "ssim", "optical_flow", "brightness"}
)
BOUNDARY_SOURCES = TRANSITION_SOURCES | {"dino"}
VLMDecision = Literal["consistent", "inconsistent", "unknown"]


def _require_non_negative(value: int, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field_name} 必须是非负整数")


def _require_unit_interval(value: float, field_name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} 必须是 [0, 1] 范围内的有限数值")


def _require_sources(
    sources: tuple[str, ...],
    *,
    allowed: frozenset[str],
    field_name: str,
    allow_empty: bool = False,
) -> None:
    if not sources and not allow_empty:
        raise ValueError(f"{field_name} 不能为空")
    unknown = set(sources) - allowed
    if unknown:
        raise ValueError(f"{field_name} 包含未知来源: {sorted(unknown)}")
    if len(set(sources)) != len(sources):
        raise ValueError(f"{field_name} 不能包含重复来源")


def _normalise_uris(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    normalised = tuple(str(value).strip() for value in values)
    if any(not value for value in normalised):
        raise ValueError(f"{field_name} 不能包含空 URI")
    return normalised


@dataclass(frozen=True)
class TransitionProposal:
    """Stage A 检测器输出的转场、黑帧或冻结候选。"""

    frame_index: int
    timestamp_ns: int
    score: float
    is_hard_cut: bool
    sources: tuple[str, ...]
    evidence_uris: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_non_negative(self.frame_index, "frame_index")
        _require_non_negative(self.timestamp_ns, "timestamp_ns")
        _require_unit_interval(self.score, "score")
        _require_sources(
            self.sources,
            allowed=TRANSITION_SOURCES,
            field_name="sources",
        )
        object.__setattr__(
            self,
            "evidence_uris",
            _normalise_uris(self.evidence_uris, "evidence_uris"),
        )


@dataclass(frozen=True)
class BoundaryScore:
    """Stage B 在单个时间点产生的语义变化分数。"""

    frame_index: int
    timestamp_ns: int
    score: float
    z_score: float

    def __post_init__(self) -> None:
        _require_non_negative(self.frame_index, "frame_index")
        _require_non_negative(self.timestamp_ns, "timestamp_ns")
        _require_unit_interval(self.score, "score")
        if not math.isfinite(self.z_score):
            raise ValueError("z_score 必须是有限数值")


@dataclass(frozen=True)
class SceneProposal:
    """不改变 Prepared Segment 物理边界的版本化场景候选。"""

    scene_id: str
    start_ns: int
    end_ns: int
    confidence: float
    sources: tuple[str, ...]
    boundary_scores: Mapping[str, float]
    evidence_uris: tuple[str, ...] = ()
    short_span: bool = False
    producer: str = "zpds.scene"
    version: str = "v1"
    config_hash: str = ""

    def __post_init__(self) -> None:
        if not self.scene_id.strip():
            raise ValueError("scene_id 不能为空")
        _require_non_negative(self.start_ns, "start_ns")
        _require_non_negative(self.end_ns, "end_ns")
        if self.end_ns <= self.start_ns:
            raise ValueError("end_ns 必须大于 start_ns")
        _require_unit_interval(self.confidence, "confidence")
        _require_sources(
            self.sources,
            allowed=BOUNDARY_SOURCES,
            field_name="sources",
            allow_empty=True,
        )
        scores = dict(self.boundary_scores)
        unknown = set(scores) - BOUNDARY_SOURCES
        if unknown:
            raise ValueError(f"boundary_scores 包含未知来源: {sorted(unknown)}")
        for source, score in scores.items():
            _require_unit_interval(float(score), f"boundary_scores.{source}")
        if not self.producer.strip() or not self.version.strip():
            raise ValueError("producer 和 version 不能为空")
        object.__setattr__(self, "boundary_scores", scores)
        object.__setattr__(
            self,
            "evidence_uris",
            _normalise_uris(self.evidence_uris, "evidence_uris"),
        )


@dataclass(frozen=True)
class VLMReviewResult:
    """人员 B 的 VLM 复核器输出契约。"""

    scene_id: str
    scene_label: str
    task_label: str
    decision: VLMDecision
    confidence: float
    reasons: str
    evidence_frame_uris: tuple[str, ...] = ()
    producer: str = "zpds.scene.vlm"
    version: str = "v1"
    config_hash: str = ""

    def __post_init__(self) -> None:
        for field_name in ("scene_id", "scene_label", "task_label"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} 不能为空")
        if self.decision not in {"consistent", "inconsistent", "unknown"}:
            raise ValueError(f"非法 decision: {self.decision!r}")
        _require_unit_interval(self.confidence, "confidence")
        if not self.reasons.strip():
            raise ValueError("reasons 不能为空")
        if not self.producer.strip() or not self.version.strip():
            raise ValueError("producer 和 version 不能为空")
        object.__setattr__(
            self,
            "evidence_frame_uris",
            _normalise_uris(
                self.evidence_frame_uris,
                "evidence_frame_uris",
            ),
        )


@dataclass(frozen=True)
class DetectorFrameScores:
    """检测器逐帧得分及可选诊断量，供后续融合使用。"""

    source: str
    scores: tuple[float, ...]
    diagnostics: Mapping[str, tuple[float, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.source not in TRANSITION_SOURCES:
            raise ValueError(f"未知检测器来源: {self.source!r}")
        for score in self.scores:
            _require_unit_interval(score, "scores")
        diagnostics = dict(self.diagnostics)
        for name, values in diagnostics.items():
            if len(values) != len(self.scores):
                raise ValueError(f"诊断序列 {name!r} 长度与 scores 不一致")
            if any(not math.isfinite(value) for value in values):
                raise ValueError(f"诊断序列 {name!r} 包含非有限值")
        object.__setattr__(self, "diagnostics", diagnostics)


__all__ = [
    "BOUNDARY_SOURCES",
    "TRANSITION_SOURCES",
    "BoundaryScore",
    "DetectorFrameScores",
    "SceneProposal",
    "TransitionProposal",
    "VLMDecision",
    "VLMReviewResult",
]
