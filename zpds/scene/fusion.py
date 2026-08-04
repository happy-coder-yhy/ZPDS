"""Stage A 多检测器得分平滑、时间聚类和加权融合。"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Protocol

import numpy as np

from zpds.scene.backends.common import finite_unit, timestamp_ns
from zpds.scene.config import FusionConfig, StageAConfig
from zpds.scene.schemas import (
    TRANSITION_SOURCES,
    BoundaryScore,
    DetectorFrameScores,
    SceneProposal,
    TransitionProposal,
)

SOURCE_ORDER = ("histogram", "ssim", "optical_flow", "brightness")
BOUNDARY_SOURCE_ORDER = (*SOURCE_ORDER, "dino")


class CenterEmbeddingProvider(Protocol):
    """按场景中心时间戳返回一维 embedding。"""

    def __call__(self, timestamp_ns: int) -> np.ndarray: ...


@dataclass(frozen=True)
class _BoundaryCandidate:
    frame_index: int
    timestamp_ns: int
    confidence: float
    is_hard_cut: bool
    sources: tuple[str, ...]
    boundary_scores: Mapping[str, float]
    evidence_uris: tuple[str, ...] = ()


class StageATransitionFusion:
    """把各 Stage A 检测器候选合并为统一 ``TransitionProposal``。

    逐帧得分先按配置做中值平滑。候选点本身的原始得分仍会保留，避免
    5 帧中值窗口把只有一帧的真实硬切完全抹掉。
    """

    def __init__(self, config: StageAConfig) -> None:
        self.config = config

    @staticmethod
    def median_smooth(
        scores: Sequence[float],
        *,
        window_size: int,
    ) -> tuple[float, ...]:
        if window_size <= 0 or window_size % 2 == 0:
            raise ValueError("window_size 必须是正奇数")
        if not scores:
            return ()
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("scores 必须是一维有限数值序列")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores 必须在 [0, 1] 范围内")
        radius = window_size // 2
        padded = np.pad(values, (radius, radius), mode="edge")
        smoothed = [
            float(np.median(padded[index : index + window_size]))
            for index in range(len(values))
        ]
        return tuple(smoothed)

    def _prepare_scores(
        self,
        frame_scores: Mapping[str, DetectorFrameScores],
    ) -> tuple[dict[str, tuple[float, ...]], int | None]:
        unknown = set(frame_scores) - TRANSITION_SOURCES
        if unknown:
            raise ValueError(f"frame_scores 包含未知来源: {sorted(unknown)}")
        expected_length: int | None = None
        smoothed: dict[str, tuple[float, ...]] = {}
        for source, result in frame_scores.items():
            if result.source != source:
                raise ValueError(
                    f"frame_scores 键 {source!r} 与结果来源 {result.source!r} 不一致"
                )
            if expected_length is None:
                expected_length = len(result.scores)
            elif len(result.scores) != expected_length:
                raise ValueError("所有检测器逐帧得分长度必须一致")
            smoothed[source] = self.median_smooth(
                result.scores,
                window_size=self.config.smoothing_window_frames,
            )
        return smoothed, expected_length

    def _cluster_proposals(
        self,
        proposals: Sequence[TransitionProposal],
        *,
        fps: float,
    ) -> list[list[TransitionProposal]]:
        merge_frames = max(1, round(self.config.merge_window_s * fps))
        ordered = sorted(proposals, key=lambda item: (item.frame_index, item.timestamp_ns))
        if not ordered:
            return []
        clusters: list[list[TransitionProposal]] = [[ordered[0]]]
        for proposal in ordered[1:]:
            if proposal.frame_index - clusters[-1][-1].frame_index <= merge_frames:
                clusters[-1].append(proposal)
            else:
                clusters.append([proposal])
        return clusters

    def _is_joint_hard_cut(
        self,
        cluster: Sequence[TransitionProposal],
        frame_scores: Mapping[str, DetectorFrameScores],
        *,
        fps: float,
    ) -> bool:
        if any(proposal.is_hard_cut for proposal in cluster):
            return True
        if not {"ssim", "optical_flow"}.issubset(frame_scores):
            return False

        cluster_start = min(item.frame_index for item in cluster)
        cluster_end = max(item.frame_index for item in cluster)
        ssim_diagnostics = frame_scores["ssim"].diagnostics
        flow_diagnostics = frame_scores["optical_flow"].diagnostics
        similarities = ssim_diagnostics.get("similarity", ())
        residuals = flow_diagnostics.get("residual_motion_px", ())
        if not similarities or not residuals:
            return False

        ssim_frames = [
            index
            for index in range(cluster_start, cluster_end + 1)
            if similarities[index] <= self.config.ssim.hard_cut_similarity
        ]
        flow_frames = [
            index
            for index in range(cluster_start, cluster_end + 1)
            if residuals[index] >= self.config.optical_flow.residual_threshold_px
        ]
        tolerance = max(1, round(self.config.merge_window_s * fps))
        return any(
            abs(ssim_frame - flow_frame) <= tolerance
            for ssim_frame in ssim_frames
            for flow_frame in flow_frames
        )

    def _fuse_cluster(
        self,
        cluster: Sequence[TransitionProposal],
        *,
        smoothed_scores: Mapping[str, tuple[float, ...]],
        frame_scores: Mapping[str, DetectorFrameScores],
        fps: float,
        start_timestamp_ns: int,
    ) -> TransitionProposal:
        by_source: dict[str, list[TransitionProposal]] = {}
        for proposal in cluster:
            for source in proposal.sources:
                by_source.setdefault(source, []).append(proposal)

        source_scores: dict[str, float] = {}
        for source, source_proposals in by_source.items():
            raw_score = max(proposal.score for proposal in source_proposals)
            smooth_values = smoothed_scores.get(source, ())
            smooth_score = max(
                (
                    smooth_values[proposal.frame_index]
                    for proposal in source_proposals
                    if proposal.frame_index < len(smooth_values)
                ),
                default=0.0,
            )
            source_scores[source] = max(raw_score, smooth_score)

        total_weight = sum(self.config.weights.values())
        fused_score = sum(
            self.config.weights[source] * score
            for source, score in source_scores.items()
        ) / total_weight

        representative = max(
            cluster,
            key=lambda proposal: (
                max(
                    self.config.weights[source] * proposal.score
                    for source in proposal.sources
                ),
                -proposal.frame_index,
            ),
        )
        ordered_sources = tuple(
            source for source in SOURCE_ORDER if source in source_scores
        )
        evidence_uris = tuple(
            dict.fromkeys(
                uri
                for proposal in cluster
                for uri in proposal.evidence_uris
            )
        )
        return TransitionProposal(
            frame_index=representative.frame_index,
            timestamp_ns=timestamp_ns(
                representative.frame_index,
                fps=fps,
                start_timestamp_ns=start_timestamp_ns,
            ),
            score=finite_unit(fused_score),
            is_hard_cut=self._is_joint_hard_cut(
                cluster,
                frame_scores,
                fps=fps,
            ),
            sources=ordered_sources,
            evidence_uris=evidence_uris,
        )

    def fuse(
        self,
        proposals: Sequence[TransitionProposal],
        frame_scores: Mapping[str, DetectorFrameScores],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
    ) -> list[TransitionProposal]:
        """融合 Stage A 候选，不创建或修改物理视频边界。"""

        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是大于 0 的有限数值")
        if isinstance(start_timestamp_ns, bool) or start_timestamp_ns < 0:
            raise ValueError("start_timestamp_ns 必须是非负整数")
        for proposal in proposals:
            unknown = set(proposal.sources) - TRANSITION_SOURCES
            if unknown:
                raise ValueError(f"候选包含未知来源: {sorted(unknown)}")

        smoothed, frame_count = self._prepare_scores(frame_scores)
        if frame_count is not None:
            out_of_range = [
                proposal.frame_index
                for proposal in proposals
                if proposal.frame_index >= frame_count
            ]
            if out_of_range:
                raise ValueError(
                    f"候选帧号超出逐帧得分范围: {sorted(set(out_of_range))}"
                )

        clusters = self._cluster_proposals(proposals, fps=fps)
        return [
            self._fuse_cluster(
                cluster,
                smoothed_scores=smoothed,
                frame_scores=frame_scores,
                fps=fps,
                start_timestamp_ns=start_timestamp_ns,
            )
            for cluster in clusters
        ]


class SceneBoundaryFusion:
    """把 Stage A 转场和 Stage B 语义候选定稿为连续 scene 区间。"""

    def __init__(
        self,
        config: FusionConfig,
        *,
        config_hash: str = "",
        center_embedding_provider: CenterEmbeddingProvider | None = None,
    ) -> None:
        self.config = config
        self.config_hash = config_hash
        self._center_embedding_provider = center_embedding_provider

    @staticmethod
    def _from_transition(proposal: TransitionProposal) -> _BoundaryCandidate:
        return _BoundaryCandidate(
            frame_index=proposal.frame_index,
            timestamp_ns=proposal.timestamp_ns,
            confidence=proposal.score,
            is_hard_cut=proposal.is_hard_cut,
            sources=proposal.sources,
            boundary_scores={source: proposal.score for source in proposal.sources},
            evidence_uris=proposal.evidence_uris,
        )

    @staticmethod
    def _from_semantic(boundary: BoundaryScore) -> _BoundaryCandidate:
        return _BoundaryCandidate(
            frame_index=boundary.frame_index,
            timestamp_ns=boundary.timestamp_ns,
            confidence=boundary.score,
            is_hard_cut=False,
            sources=("dino",),
            boundary_scores={"dino": boundary.score},
        )

    @staticmethod
    def _merge_candidates(
        representative: _BoundaryCandidate,
        candidates: Sequence[_BoundaryCandidate],
    ) -> _BoundaryCandidate:
        sources = tuple(
            source
            for source in BOUNDARY_SOURCE_ORDER
            if any(source in candidate.sources for candidate in candidates)
        )
        boundary_scores = {
            source: max(
                candidate.boundary_scores[source]
                for candidate in candidates
                if source in candidate.boundary_scores
            )
            for source in sources
        }
        evidence = tuple(
            dict.fromkeys(
                uri
                for candidate in candidates
                for uri in candidate.evidence_uris
            )
        )
        return _BoundaryCandidate(
            frame_index=representative.frame_index,
            timestamp_ns=representative.timestamp_ns,
            confidence=max(candidate.confidence for candidate in candidates),
            is_hard_cut=representative.is_hard_cut,
            sources=sources,
            boundary_scores=boundary_scores,
            evidence_uris=evidence,
        )

    def _apply_hysteresis(
        self,
        candidates: Sequence[_BoundaryCandidate],
    ) -> list[_BoundaryCandidate]:
        if not candidates:
            return []
        ordered = sorted(candidates, key=lambda item: (item.frame_index, item.timestamp_ns))
        groups: list[list[_BoundaryCandidate]] = [[ordered[0]]]
        for candidate in ordered[1:]:
            if (
                candidate.frame_index - groups[-1][-1].frame_index
                <= self.config.hysteresis_frames
            ):
                groups[-1].append(candidate)
            else:
                groups.append([candidate])

        result: list[_BoundaryCandidate] = []
        for group in groups:
            hard_candidates = [candidate for candidate in group if candidate.is_hard_cut]
            if not hard_candidates:
                representative = max(
                    group,
                    key=lambda item: (item.confidence, -item.frame_index),
                )
                result.append(self._merge_candidates(representative, group))
                continue

            assignments = [[hard] for hard in hard_candidates]
            for candidate in group:
                if candidate.is_hard_cut:
                    continue
                nearest_index = min(
                    range(len(hard_candidates)),
                    key=lambda item: (
                        abs(
                            hard_candidates[item].frame_index
                            - candidate.frame_index
                        ),
                        hard_candidates[item].frame_index,
                    ),
                )
                assignments[nearest_index].append(candidate)
            result.extend(
                self._merge_candidates(hard, assignments[index])
                for index, hard in enumerate(hard_candidates)
            )
        return sorted(result, key=lambda item: (item.timestamp_ns, item.frame_index))

    def _apply_minimum_duration(
        self,
        candidates: Sequence[_BoundaryCandidate],
        *,
        start_ns: int,
        end_ns: int,
    ) -> list[_BoundaryCandidate]:
        minimum_ns = round(self.config.min_scene_duration_s * 1_000_000_000)
        hard = [candidate for candidate in candidates if candidate.is_hard_cut]
        soft = [candidate for candidate in candidates if not candidate.is_hard_cut]
        anchors: list[tuple[int, _BoundaryCandidate | None]] = [
            (start_ns, None),
            *((candidate.timestamp_ns, candidate) for candidate in hard),
            (end_ns, None),
        ]
        kept: list[_BoundaryCandidate] = list(hard)
        for (interval_start, _), (interval_end, _) in pairwise(anchors):
            interval_candidates = [
                candidate
                for candidate in soft
                if interval_start < candidate.timestamp_ns < interval_end
            ]
            selected: list[_BoundaryCandidate] = []
            previous_ns = interval_start
            for candidate in interval_candidates:
                if candidate.timestamp_ns - previous_ns >= minimum_ns:
                    selected.append(candidate)
                    previous_ns = candidate.timestamp_ns
            while selected and interval_end - selected[-1].timestamp_ns < minimum_ns:
                selected.pop()
            kept.extend(selected)
        return sorted(kept, key=lambda item: (item.timestamp_ns, item.frame_index))

    def _normalised_center_embedding(self, timestamp: int) -> np.ndarray:
        assert self._center_embedding_provider is not None
        embedding = np.asarray(
            self._center_embedding_provider(timestamp),
            dtype=np.float64,
        )
        if embedding.ndim != 1 or not len(embedding):
            raise ValueError("中心 embedding 必须是一维非空向量")
        if not np.all(np.isfinite(embedding)):
            raise ValueError("中心 embedding 包含 NaN 或无穷值")
        norm = float(np.linalg.norm(embedding))
        if norm <= np.finfo(np.float64).eps:
            raise ValueError("中心 embedding 不能是零向量")
        return embedding / norm

    def _merge_same_semantics(
        self,
        candidates: Sequence[_BoundaryCandidate],
        *,
        start_ns: int,
        end_ns: int,
    ) -> list[_BoundaryCandidate]:
        if self._center_embedding_provider is None:
            return list(candidates)
        kept = list(candidates)
        index = 0
        while index < len(kept):
            separator = kept[index]
            if separator.is_hard_cut:
                index += 1
                continue
            left_start = start_ns if index == 0 else kept[index - 1].timestamp_ns
            right_end = end_ns if index + 1 == len(kept) else kept[index + 1].timestamp_ns
            left_center = (left_start + separator.timestamp_ns) // 2
            right_center = (separator.timestamp_ns + right_end) // 2
            left_embedding = self._normalised_center_embedding(left_center)
            right_embedding = self._normalised_center_embedding(right_center)
            if left_embedding.shape != right_embedding.shape:
                raise ValueError("相邻场景中心 embedding 维度不一致")
            similarity = float(np.dot(left_embedding, right_embedding))
            if similarity > self.config.same_scene_similarity:
                del kept[index]
                if index > 0:
                    index -= 1
            else:
                index += 1
        return kept

    def _to_scenes(
        self,
        candidates: Sequence[_BoundaryCandidate],
        *,
        start_ns: int,
        end_ns: int,
    ) -> list[SceneProposal]:
        boundary_times = [start_ns, *(item.timestamp_ns for item in candidates), end_ns]
        minimum_ns = round(self.config.min_scene_duration_s * 1_000_000_000)
        scenes: list[SceneProposal] = []
        for index, (scene_start, scene_end) in enumerate(
            pairwise(boundary_times)
        ):
            evidence_boundaries: list[_BoundaryCandidate] = []
            if index > 0:
                evidence_boundaries.append(candidates[index - 1])
            if index < len(candidates):
                evidence_boundaries.append(candidates[index])
            sources = tuple(
                source
                for source in BOUNDARY_SOURCE_ORDER
                if any(source in item.sources for item in evidence_boundaries)
            )
            boundary_scores = {
                source: max(
                    item.boundary_scores[source]
                    for item in evidence_boundaries
                    if source in item.boundary_scores
                )
                for source in sources
            }
            evidence_uris = tuple(
                dict.fromkeys(
                    uri
                    for item in evidence_boundaries
                    for uri in item.evidence_uris
                )
            )
            confidence = (
                float(np.mean([item.confidence for item in evidence_boundaries]))
                if evidence_boundaries
                else 0.0
            )
            scenes.append(
                SceneProposal(
                    scene_id=f"scene_{index:06d}",
                    start_ns=scene_start,
                    end_ns=scene_end,
                    confidence=confidence,
                    sources=sources,
                    boundary_scores=boundary_scores,
                    evidence_uris=evidence_uris,
                    short_span=scene_end - scene_start < minimum_ns,
                    config_hash=self.config_hash,
                )
            )
        return scenes

    def fuse(
        self,
        transitions: Sequence[TransitionProposal],
        semantic_boundaries: Sequence[BoundaryScore],
        *,
        start_ns: int,
        end_ns: int,
        fps: float,
    ) -> list[SceneProposal]:
        if isinstance(start_ns, bool) or start_ns < 0:
            raise ValueError("start_ns 必须是非负整数")
        if isinstance(end_ns, bool) or end_ns <= start_ns:
            raise ValueError("end_ns 必须大于 start_ns")
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("fps 必须是大于 0 的有限数值")

        candidates = [
            *(self._from_transition(item) for item in transitions),
            *(self._from_semantic(item) for item in semantic_boundaries),
        ]
        outside = [
            item.timestamp_ns
            for item in candidates
            if not start_ns <= item.timestamp_ns <= end_ns
        ]
        if outside:
            raise ValueError(f"候选时间戳超出输入区间: {sorted(set(outside))}")
        candidates = [
            item for item in candidates if start_ns < item.timestamp_ns < end_ns
        ]
        candidates = self._apply_hysteresis(candidates)
        candidates = self._apply_minimum_duration(
            candidates,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        candidates = self._merge_same_semantics(
            candidates,
            start_ns=start_ns,
            end_ns=end_ns,
        )
        return self._to_scenes(candidates, start_ns=start_ns, end_ns=end_ns)


__all__ = [
    "BOUNDARY_SOURCE_ORDER",
    "SOURCE_ORDER",
    "CenterEmbeddingProvider",
    "SceneBoundaryFusion",
    "StageATransitionFusion",
]
