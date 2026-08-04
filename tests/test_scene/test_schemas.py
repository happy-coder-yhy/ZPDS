from __future__ import annotations

import numpy as np
import pytest

from zpds.scene.backends import HistogramTransitionDetector
from zpds.scene.config import HistogramConfig
from zpds.scene.contracts import TransitionDetector
from zpds.scene.schemas import (
    BoundaryScore,
    DetectorFrameScores,
    SceneProposal,
    TransitionProposal,
    VLMReviewResult,
)


def test_transition_proposal_validates_score_and_source() -> None:
    with pytest.raises(ValueError, match="score"):
        TransitionProposal(1, 1, 1.1, False, ("ssim",))
    with pytest.raises(ValueError, match="未知来源"):
        TransitionProposal(1, 1, 0.5, False, ("unknown",))


def test_boundary_and_scene_proposals_validate_intervals() -> None:
    boundary = BoundaryScore(3, 300, 0.8, 2.1)
    assert boundary.z_score == 2.1

    with pytest.raises(ValueError, match="end_ns"):
        SceneProposal(
            scene_id="scene-1",
            start_ns=10,
            end_ns=10,
            confidence=0.8,
            sources=("dino",),
            boundary_scores={"dino": 0.8},
        )


def test_vlm_result_contract_rejects_unknown_decision() -> None:
    with pytest.raises(ValueError, match="decision"):
        VLMReviewResult(
            scene_id="scene-1",
            scene_label="kitchen",
            task_label="cutting",
            decision="maybe",  # type: ignore[arg-type]
            confidence=0.5,
            reasons="test",
        )


def test_detector_scores_require_aligned_diagnostics() -> None:
    with pytest.raises(ValueError, match="长度"):
        DetectorFrameScores("ssim", (0.0, 0.5), {"similarity": (1.0,)})


def test_histogram_detector_satisfies_protocol() -> None:
    detector = HistogramTransitionDetector(HistogramConfig())
    assert isinstance(detector, TransitionDetector)
    empty = detector.detect([], fps=10.0)
    assert empty == []


def test_scene_proposal_copies_boundary_score_mapping() -> None:
    scores = {"ssim": 0.8}
    proposal = SceneProposal(
        scene_id="scene-1",
        start_ns=0,
        end_ns=10,
        confidence=0.8,
        sources=("ssim",),
        boundary_scores=scores,
    )
    scores["ssim"] = 0.1
    assert proposal.boundary_scores["ssim"] == 0.8


def test_detector_rejects_non_array_frame() -> None:
    detector = HistogramTransitionDetector(HistogramConfig())
    with pytest.raises(TypeError, match="numpy.ndarray"):
        detector.detect(["not-an-image"], fps=10.0)  # type: ignore[list-item]

    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="fps"):
        detector.detect([frame], fps=0.0)
