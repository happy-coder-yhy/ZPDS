"""人员 B：场景自动分割 + VLM 复核的端到端编排。"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from zpds.core.decisions import Decision
from zpds.core.quality import QualityMetric
from zpds.scene.backend_router import SceneBackendRouter
from zpds.scene.backends import (
    BrightnessTransitionDetector,
    DinoV2SmallEmbedder,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.contracts import TransitionDetector, VLMReviewer
from zpds.scene.fusion import SceneBoundaryFusion, StageATransitionFusion
from zpds.scene.qc_integration import (
    build_scene_decisions,
    build_scene_metrics,
)
from zpds.scene.sampling import extract_representative_frames
from zpds.scene.schemas import (
    BoundaryScore,
    DetectorFrameScores,
    SceneProposal,
    TransitionProposal,
    VLMReviewResult,
)
from zpds.scene.vlm_review import (
    SceneLabels,
    select_review_queue,
)

ProgressCallback = Callable[[str, int, int], None]


class StageBBackend(Protocol):
    def embed(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray: ...

    def detect(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
        candidate_frame_indices: Sequence[int] | None = None,
    ) -> list[BoundaryScore]: ...


@dataclass(frozen=True)
class ScenePipelineRun:
    """一次完整 scene 分割 + VLM 复核的运行结果。"""

    skipped: bool
    skip_reason: str | None
    frame_count: int
    fps: float
    start_ns: int
    end_ns: int
    config_hash: str
    profile: str | None
    scenes: tuple[SceneProposal, ...] = ()
    vlm_results: tuple[VLMReviewResult, ...] = ()
    review_queue: tuple[VLMReviewResult, ...] = ()
    metrics: tuple[QualityMetric, ...] = ()
    decisions: tuple[Decision, ...] = ()


def _transition_detectors(
    config: SceneConfig,
    progress: ProgressCallback | None = None,
) -> list[TransitionDetector]:
    router = SceneBackendRouter.from_config(config)
    factories: dict[str, Callable[[], TransitionDetector]] = {
        "histogram": lambda: HistogramTransitionDetector(config.stage_a.histogram),
        "ssim": lambda: SSIMTransitionDetector(config.stage_a.ssim),
        "optical_flow": lambda: OpticalFlowTransitionDetector(
            config.stage_a.optical_flow,
            progress_callback=(
                (
                    lambda completed, total: progress(
                        "optical_flow", completed, total
                    )
                )
                if progress is not None
                else None
            ),
        ),
        "brightness": lambda: BrightnessTransitionDetector(
            config.stage_a.brightness
        ),
    }
    return [factories[name]() for name in router.policy.stage_a_backends]


def _run_stage_a(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    start_ns: int,
    config: SceneConfig,
    progress: ProgressCallback | None = None,
) -> list[TransitionProposal]:
    detectors = _transition_detectors(config, progress)
    frame_scores: dict[str, DetectorFrameScores] = {}
    proposals: list[TransitionProposal] = []
    total_pairs = max(0, len(frames) - 1)
    for detector in detectors:
        source = str(detector.source)
        if progress is not None:
            progress(source, 0, total_pairs)
        scores = detector.score_frames(frames, fps=fps)
        frame_scores[source] = scores
        if progress is not None:
            progress(source, total_pairs, total_pairs)
        proposals.extend(
            detector.detect(
                frames,
                fps=fps,
                start_timestamp_ns=start_ns,
                frame_scores=scores,
            )
        )
    if progress is not None:
        progress("fusion", 0, 1)
    fused = StageATransitionFusion(config.stage_a).fuse(
        proposals,
        frame_scores,
        fps=fps,
        start_timestamp_ns=start_ns,
    )
    if progress is not None:
        progress("fusion", 1, 1)
    return fused


def _center_embedding_provider(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    start_ns: int,
    embedder: StageBBackend,
):
    cache: dict[int, np.ndarray] = {}

    def provide(timestamp: int) -> np.ndarray:
        relative_ns = timestamp - start_ns
        index = round(relative_ns * fps / 1_000_000_000)
        index = max(0, min(len(frames) - 1, index))
        if index not in cache:
            frame_rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            cache[index] = embedder.embed([frame_rgb])[0]
        return cache[index]

    return provide


def run_scene_pipeline(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    config: SceneConfig,
    stage_b_backend: StageBBackend | None = None,
    vlm_reviewer: VLMReviewer | None = None,
    labels: SceneLabels | None = None,
    start_ns: int = 0,
    progress: ProgressCallback | None = None,
) -> ScenePipelineRun:
    """执行 Stage A → Stage B → 融合 → VLM 复核，并产出 QC 指标与决策。"""

    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("fps 必须是大于 0 的有限数值")
    if isinstance(start_ns, bool) or start_ns < 0:
        raise ValueError("start_ns 必须是非负整数")
    if not frames:
        raise ValueError("frames 不能为空")
    end_ns = start_ns + round(len(frames) * 1_000_000_000 / fps)

    if not config.enabled:
        return ScenePipelineRun(
            skipped=True,
            skip_reason="scene.enabled=false",
            frame_count=len(frames),
            fps=fps,
            start_ns=start_ns,
            end_ns=end_ns,
            config_hash=config.config_hash,
            profile=config.profile,
        )

    transitions = _run_stage_a(
        frames,
        fps=fps,
        start_ns=start_ns,
        config=config,
        progress=progress,
    )
    embedder = stage_b_backend or DinoV2SmallEmbedder(config.stage_b)
    effective_candidates = (
        tuple(item.frame_index for item in transitions) if transitions else None
    )
    if progress is not None:
        progress("dino", 0, len(frames))
    semantic_boundaries = embedder.detect(
        frames,
        fps=fps,
        start_timestamp_ns=start_ns,
        candidate_frame_indices=effective_candidates,
    )
    if progress is not None:
        progress("dino", len(frames), len(frames))
    if progress is not None:
        progress("scene_fusion", 0, 1)
    scenes = SceneBoundaryFusion(
        config.fusion,
        config_hash=config.config_hash,
        center_embedding_provider=_center_embedding_provider(
            frames,
            fps=fps,
            start_ns=start_ns,
            embedder=embedder,
        ),
    ).fuse(
        transitions,
        semantic_boundaries,
        start_ns=start_ns,
        end_ns=end_ns,
        fps=fps,
    )
    if progress is not None:
        progress("scene_fusion", 1, 1)

    vlm_results: list[VLMReviewResult] = []
    if (
        vlm_reviewer is not None
        and config.vlm.enabled
        and scenes
    ):
        for index, scene in enumerate(scenes):
            if progress is not None:
                progress("vlm", index, len(scenes))
            representative = extract_representative_frames(
                frames,
                scene,
                fps=fps,
                segment_start_ns=start_ns,
            )
            vlm_results.append(vlm_reviewer.review(scene, representative))
        if progress is not None:
            progress("vlm", len(scenes), len(scenes))

    review_queue = select_review_queue(
        vlm_results,
        confidence_threshold=config.vlm.review_confidence_threshold,
    )
    metrics = build_scene_metrics(
        scenes,
        vlm_results,
        config_hash=config.config_hash,
    )
    decisions = build_scene_decisions(
        scenes,
        vlm_results,
        config=config,
        vlm_enabled=config.vlm.enabled,
    )
    return ScenePipelineRun(
        skipped=False,
        skip_reason=None,
        frame_count=len(frames),
        fps=fps,
        start_ns=start_ns,
        end_ns=end_ns,
        config_hash=config.config_hash,
        profile=config.profile,
        scenes=tuple(scenes),
        vlm_results=tuple(vlm_results),
        review_queue=tuple(review_queue),
        metrics=tuple(metrics),
        decisions=tuple(decisions),
    )


__all__ = [
    "ProgressCallback",
    "ScenePipelineRun",
    "StageBBackend",
    "VLMReviewer",
    "run_scene_pipeline",
]
