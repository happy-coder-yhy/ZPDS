"""Stage A 多数据源阈值回归；无金标时只测量，不自动安装阈值。"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from dataclasses import asdict
from typing import Any

import numpy as np

from zpds.scene.backend_router import SceneBackendRouter
from zpds.scene.backends import (
    BrightnessTransitionDetector,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.fusion import StageATransitionFusion

ProgressCallback = Callable[[str, int, int], None]


def _distribution(values: Sequence[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "count": 0,
            "min": None,
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    if array.ndim != 1 or not np.all(np.isfinite(array)):
        raise ValueError("回归分布必须是一维有限数值")
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "mean": float(np.mean(array)),
        "p50": float(np.quantile(array, 0.50)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "max": float(np.max(array)),
    }


def _detectors(
    config: SceneConfig,
    progress: ProgressCallback | None,
):
    router = SceneBackendRouter.from_config(config)
    factories = {
        "histogram": lambda: HistogramTransitionDetector(config.stage_a.histogram),
        "ssim": lambda: SSIMTransitionDetector(config.stage_a.ssim),
        "optical_flow": lambda: OpticalFlowTransitionDetector(
            config.stage_a.optical_flow,
            progress_callback=(
                (lambda completed, total: progress("optical_flow", completed, total))
                if progress is not None
                else None
            ),
        ),
        "brightness": lambda: BrightnessTransitionDetector(config.stage_a.brightness),
    }
    return tuple(factories[name]() for name in router.policy.stage_a_backends)


def run_stage_a_regression(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    config: SceneConfig,
    start_timestamp_ns: int = 0,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """运行一次 Stage A 回归并返回可序列化诊断，不修改任何配置。"""

    if not frames:
        raise ValueError("frames 不能为空")
    frame_scores = {}
    proposals = []
    detector_reports: dict[str, Any] = {}
    total_pairs = max(0, len(frames) - 1)

    for detector in _detectors(config, progress):
        if progress is not None:
            progress(detector.source, 0, total_pairs)
        started = time.perf_counter()
        scores = detector.score_frames(frames, fps=fps)
        detector_proposals = detector.detect(
            frames,
            fps=fps,
            start_timestamp_ns=start_timestamp_ns,
            frame_scores=scores,
        )
        elapsed_s = time.perf_counter() - started
        if progress is not None:
            progress(detector.source, total_pairs, total_pairs)
        frame_scores[detector.source] = scores
        proposals.extend(detector_proposals)
        detector_reports[detector.source] = {
            "elapsed_s": elapsed_s,
            "proposal_count": len(detector_proposals),
            "proposal_frame_indices": [
                proposal.frame_index for proposal in detector_proposals
            ],
            "scores": _distribution(scores.scores),
            "diagnostics": {
                name: _distribution(values)
                for name, values in sorted(scores.diagnostics.items())
            },
            "config": asdict(getattr(config.stage_a, detector.source)),
        }

    fused = StageATransitionFusion(config.stage_a).fuse(
        proposals,
        frame_scores,
        fps=fps,
        start_timestamp_ns=start_timestamp_ns,
    )
    return {
        "frame_count": len(frames),
        "fps": fps,
        "duration_s": len(frames) / fps,
        "config_hash": config.config_hash,
        "profile": config.profile,
        "calibration_status": config.governance.calibration_status,
        "requires_adjudicated_gold": True,
        "thresholds_changed": False,
        "detectors": detector_reports,
        "fused_transition_count": len(fused),
        "fused_transitions": [asdict(proposal) for proposal in fused],
    }


__all__ = ["run_stage_a_regression"]
