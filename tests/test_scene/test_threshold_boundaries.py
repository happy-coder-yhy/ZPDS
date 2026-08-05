from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from zpds.scene.backends import (
    BrightnessTransitionDetector,
    DinoV2SmallEmbedder,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.schemas import DetectorFrameScores


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return SceneConfig.load("configs/scene/default.yaml")


def _frames() -> list[np.ndarray]:
    return [np.zeros((16, 16, 3), dtype=np.uint8) for _ in range(2)]


def test_histogram_threshold_is_inclusive(config: SceneConfig) -> None:
    threshold = config.stage_a.histogram.threshold
    detector = HistogramTransitionDetector(config.stage_a.histogram)
    frames = _frames()

    exact = detector.detect(
        frames,
        fps=10.0,
        frame_scores=DetectorFrameScores("histogram", (0.0, threshold)),
    )
    below = detector.detect(
        frames,
        fps=10.0,
        frame_scores=DetectorFrameScores(
            "histogram",
            (0.0, np.nextafter(threshold, 0.0)),
        ),
    )

    assert [proposal.frame_index for proposal in exact] == [1]
    assert below == []


def test_ssim_hard_cut_threshold_is_inclusive(config: SceneConfig) -> None:
    threshold = config.stage_a.ssim.hard_cut_similarity
    detector = SSIMTransitionDetector(config.stage_a.ssim)
    frames = _frames()

    exact = detector.detect(
        frames,
        fps=10.0,
        frame_scores=DetectorFrameScores(
            "ssim",
            (0.0, 1.0 - threshold),
            diagnostics={"similarity": (1.0, threshold)},
        ),
    )
    above = detector.detect(
        frames,
        fps=10.0,
        frame_scores=DetectorFrameScores(
            "ssim",
            (0.0, 1.0 - threshold),
            diagnostics={"similarity": (1.0, np.nextafter(threshold, 1.0))},
        ),
    )

    assert [proposal.frame_index for proposal in exact] == [1]
    assert above == []


def test_optical_flow_residual_threshold_is_inclusive(config: SceneConfig) -> None:
    flow_config = config.stage_a.optical_flow
    threshold = flow_config.residual_threshold_px
    detector = OpticalFlowTransitionDetector(flow_config)
    frames = _frames()

    def scores(residual: float) -> DetectorFrameScores:
        return DetectorFrameScores(
            "optical_flow",
            (0.0, residual / flow_config.residual_hard_scale_px),
            diagnostics={
                "raw_motion_px": (1.0, 1.0),
                "residual_motion_px": (0.0, residual),
                "ransac_inlier_ratio": (1.0, 1.0),
            },
        )

    exact = detector.detect(frames, fps=10.0, frame_scores=scores(threshold))
    below = detector.detect(
        frames,
        fps=10.0,
        frame_scores=scores(np.nextafter(threshold, 0.0)),
    )

    assert [proposal.frame_index for proposal in exact] == [1]
    assert below == []


def test_brightness_jump_threshold_is_inclusive(config: SceneConfig) -> None:
    threshold = config.stage_a.brightness.mean_jump_threshold
    detector = BrightnessTransitionDetector(config.stage_a.brightness)
    frames = _frames()

    def scores(jump: float) -> DetectorFrameScores:
        return DetectorFrameScores(
            "brightness",
            (0.0, jump),
            diagnostics={
                "mean_luma": (0.0, jump),
                "black_ratio": (0.0, 0.0),
            },
        )

    exact = detector.detect(frames, fps=10.0, frame_scores=scores(threshold))
    below = detector.detect(
        frames,
        fps=10.0,
        frame_scores=scores(np.nextafter(threshold, 0.0)),
    )

    assert [proposal.frame_index for proposal in exact] == [1]
    assert below == []


def test_dino_z_score_threshold_is_strictly_greater(
    config: SceneConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    threshold = config.stage_b.z_score_threshold
    frames = _frames()
    embedder = DinoV2SmallEmbedder(
        replace(config.stage_b, min_z_score_samples=1),
        embedding_function=lambda values: np.tile(
            np.eye(1, 384, dtype=np.float32),
            (len(values), 1),
        ),
    )

    monkeypatch.setattr(embedder, "local_z_scores", lambda _changes: (0.0, threshold))
    exact = embedder.score_boundaries(
        frames,
        frame_indices=[0, 1],
        timestamps_ns=[0, 1],
    )
    monkeypatch.setattr(
        embedder,
        "local_z_scores",
        lambda _changes: (0.0, np.nextafter(threshold, np.inf)),
    )
    above = embedder.score_boundaries(
        frames,
        frame_indices=[0, 1],
        timestamps_ns=[0, 1],
    )

    assert exact == []
    assert [boundary.frame_index for boundary in above] == [1]
