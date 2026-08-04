from __future__ import annotations

import pytest

from zpds.scene.backends import (
    BrightnessTransitionDetector,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.fusion import StageATransitionFusion
from zpds.scene.schemas import DetectorFrameScores, TransitionProposal
from zpds.scene.testing import hard_cut_fixture


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return SceneConfig.load("configs/scene/default.yaml")


def _scores(
    source: str,
    values: dict[int, float],
    *,
    length: int = 20,
    diagnostics: dict[str, tuple[float, ...]] | None = None,
) -> DetectorFrameScores:
    scores = [0.0] * length
    for index, value in values.items():
        scores[index] = value
    return DetectorFrameScores(source, tuple(scores), diagnostics or {})


def _proposal(
    source: str,
    frame_index: int,
    score: float,
    *,
    evidence: tuple[str, ...] = (),
) -> TransitionProposal:
    return TransitionProposal(
        frame_index=frame_index,
        timestamp_ns=frame_index * 100_000_000,
        score=score,
        is_hard_cut=False,
        sources=(source,),
        evidence_uris=evidence,
    )


def test_median_smoothing_uses_five_frame_window() -> None:
    smoothed = StageATransitionFusion.median_smooth(
        (0.0, 0.0, 1.0, 0.0, 0.0),
        window_size=5,
    )

    assert smoothed == (0.0, 0.0, 0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="正奇数"):
        StageATransitionFusion.median_smooth((0.0,), window_size=4)


def test_fusion_merges_weighted_candidates_and_marks_joint_hard_cut(
    config: SceneConfig,
) -> None:
    similarities = [1.0] * 20
    similarities[10] = 0.2
    residuals = [0.0] * 20
    residuals[11] = 3.0
    frame_scores = {
        "histogram": _scores("histogram", {9: 0.6}),
        "ssim": _scores(
            "ssim",
            {10: 0.9},
            diagnostics={"similarity": tuple(similarities)},
        ),
        "optical_flow": _scores(
            "optical_flow",
            {11: 0.8},
            diagnostics={"residual_motion_px": tuple(residuals)},
        ),
        "brightness": _scores("brightness", {17: 1.0}),
    }
    proposals = [
        _proposal("histogram", 9, 0.6, evidence=("hist.jpg",)),
        _proposal("ssim", 10, 0.9, evidence=("ssim.jpg",)),
        _proposal("optical_flow", 11, 0.8, evidence=("flow.jpg",)),
        _proposal("brightness", 17, 1.0),
    ]

    fused = StageATransitionFusion(config.stage_a).fuse(
        proposals,
        frame_scores,
        fps=10.0,
        start_timestamp_ns=1_000,
    )

    assert len(fused) == 2
    first = fused[0]
    assert first.frame_index == 10
    assert first.timestamp_ns == 1_000_001_000
    assert first.sources == ("histogram", "ssim", "optical_flow")
    assert first.score == pytest.approx(0.66)
    assert first.is_hard_cut is True
    assert first.evidence_uris == ("hist.jpg", "ssim.jpg", "flow.jpg")
    assert fused[1].sources == ("brightness",)


def test_hard_cut_requires_ssim_drop_and_flow_residual(
    config: SceneConfig,
) -> None:
    similarities = [1.0] * 12
    similarities[5] = 0.2
    low_residuals = [0.0] * 12
    frame_scores = {
        "ssim": _scores(
            "ssim",
            {5: 0.8},
            length=12,
            diagnostics={"similarity": tuple(similarities)},
        ),
        "optical_flow": _scores(
            "optical_flow",
            {5: 0.1},
            length=12,
            diagnostics={"residual_motion_px": tuple(low_residuals)},
        ),
    }

    fused = StageATransitionFusion(config.stage_a).fuse(
        [_proposal("ssim", 5, 0.8), _proposal("optical_flow", 5, 0.1)],
        frame_scores,
        fps=10.0,
    )

    assert fused[0].is_hard_cut is False


def test_fusion_rejects_mismatched_score_lengths(config: SceneConfig) -> None:
    frame_scores = {
        "ssim": _scores("ssim", {}, length=10),
        "histogram": _scores("histogram", {}, length=11),
    }
    with pytest.raises(ValueError, match="长度"):
        StageATransitionFusion(config.stage_a).fuse([], frame_scores, fps=10.0)


def test_real_detectors_produce_joint_hard_cut(config: SceneConfig) -> None:
    fixture = hard_cut_fixture()
    detectors = (
        HistogramTransitionDetector(config.stage_a.histogram),
        SSIMTransitionDetector(config.stage_a.ssim),
        OpticalFlowTransitionDetector(config.stage_a.optical_flow),
        BrightnessTransitionDetector(config.stage_a.brightness),
    )
    frame_scores = {
        detector.source: detector.score_frames(fixture.frames, fps=fixture.fps)
        for detector in detectors
    }
    proposals = [
        proposal
        for detector in detectors
        for proposal in detector.detect(fixture.frames, fps=fixture.fps)
    ]

    fused = StageATransitionFusion(config.stage_a).fuse(
        proposals,
        frame_scores,
        fps=fixture.fps,
    )

    hard_cuts = [proposal for proposal in fused if proposal.is_hard_cut]
    assert any(abs(proposal.frame_index - 10) <= 1 for proposal in hard_cuts)
