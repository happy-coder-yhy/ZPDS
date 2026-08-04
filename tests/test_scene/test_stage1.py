from __future__ import annotations

import math

import numpy as np
import pytest

from zpds.scene.backends import (
    BrightnessTransitionDetector,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.testing import (
    all_stage_a_fixtures,
    black_frame_fixture,
    ego_translation_fixture,
    freeze_fixture,
    gradual_fixture,
    hard_cut_fixture,
    semantic_task_switch_fixture,
)


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return SceneConfig.load("configs/scene/default.yaml")


def _has_boundary(proposals, expected: int, tolerance: int = 1) -> bool:
    return any(abs(proposal.frame_index - expected) <= tolerance for proposal in proposals)


def test_fixture_suite_covers_all_required_scenarios() -> None:
    names = {fixture.name for fixture in all_stage_a_fixtures()}
    assert names == {
        "hard_cut",
        "gradual",
        "black_frames",
        "freeze",
        "ego_translation",
    }
    assert semantic_task_switch_fixture().boundaries[0].kind == "semantic_change"


def test_histogram_detects_hard_cut(config: SceneConfig) -> None:
    fixture = hard_cut_fixture()
    detector = HistogramTransitionDetector(config.stage_a.histogram)

    proposals = detector.detect(fixture.frames, fps=fixture.fps)

    assert _has_boundary(proposals, 10)
    assert all(proposal.sources == ("histogram",) for proposal in proposals)


def test_brightness_detects_black_frame_entry_and_exit(config: SceneConfig) -> None:
    fixture = black_frame_fixture()
    detector = BrightnessTransitionDetector(config.stage_a.brightness)

    proposals = detector.detect(
        fixture.frames,
        fps=fixture.fps,
        start_timestamp_ns=1_000,
    )

    assert _has_boundary(proposals, 6)
    assert _has_boundary(proposals, 11)
    entry = next(item for item in proposals if item.frame_index == 6)
    assert entry.timestamp_ns == 600_001_000


def test_ssim_detects_hard_cut_and_gradual_change(config: SceneConfig) -> None:
    detector = SSIMTransitionDetector(config.stage_a.ssim)
    hard = hard_cut_fixture()
    gradual = gradual_fixture()

    hard_proposals = detector.detect(hard.frames, fps=hard.fps)
    gradual_proposals = detector.detect(gradual.frames, fps=gradual.fps)

    assert _has_boundary(hard_proposals, 10)
    assert _has_boundary(gradual_proposals, 10, tolerance=4)


def test_ssim_is_finite_for_identical_pure_colour_frames(config: SceneConfig) -> None:
    detector = SSIMTransitionDetector(config.stage_a.ssim)
    frame = np.full((32, 48, 3), 127, dtype=np.uint8)

    similarity = detector.similarity(frame, frame.copy())
    scores = detector.score_frames([frame, frame.copy()], fps=10.0)

    assert similarity == pytest.approx(1.0, abs=1e-6)
    assert all(math.isfinite(score) for score in scores.scores)


def test_optical_flow_detects_hard_cut_and_freeze(config: SceneConfig) -> None:
    detector = OpticalFlowTransitionDetector(config.stage_a.optical_flow)
    hard = hard_cut_fixture()
    freeze = freeze_fixture()

    hard_proposals = detector.detect(hard.frames, fps=hard.fps)
    freeze_proposals = detector.detect(freeze.frames, fps=freeze.fps)

    assert _has_boundary(hard_proposals, 10)
    assert _has_boundary(freeze_proposals, 6)


def test_ego_translation_is_suppressed_by_global_motion_compensation(
    config: SceneConfig,
) -> None:
    fixture = ego_translation_fixture()
    detector = OpticalFlowTransitionDetector(config.stage_a.optical_flow)

    scores = detector.score_frames(fixture.frames, fps=fixture.fps)
    proposals = detector.detect(fixture.frames, fps=fixture.fps)

    assert proposals == []
    assert max(scores.diagnostics["raw_motion_px"]) > 0.5
    assert max(scores.diagnostics["residual_motion_px"]) < config.stage_a.optical_flow.residual_threshold_px
    assert max(scores.diagnostics["ransac_inlier_ratio"]) > 0.8


def test_empty_and_single_frame_inputs_are_safe(config: SceneConfig) -> None:
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    detectors = (
        HistogramTransitionDetector(config.stage_a.histogram),
        BrightnessTransitionDetector(config.stage_a.brightness),
        SSIMTransitionDetector(config.stage_a.ssim),
        OpticalFlowTransitionDetector(config.stage_a.optical_flow),
    )

    for detector in detectors:
        assert detector.detect([], fps=10.0) == []
        assert detector.detect([frame], fps=10.0) == []


def test_frame_size_mismatch_is_rejected(config: SceneConfig) -> None:
    detector = OpticalFlowTransitionDetector(config.stage_a.optical_flow)
    frames = [
        np.zeros((24, 32, 3), dtype=np.uint8),
        np.zeros((20, 32, 3), dtype=np.uint8),
    ]
    with pytest.raises(ValueError, match="相同宽高"):
        detector.detect(frames, fps=10.0)
