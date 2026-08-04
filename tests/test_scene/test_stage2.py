from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from zpds.scene.config import SceneConfig
from zpds.scene.contracts import BoundaryFusion
from zpds.scene.fusion import SceneBoundaryFusion
from zpds.scene.schemas import BoundaryScore, TransitionProposal

SECOND = 1_000_000_000


@pytest.fixture(scope="module")
def config() -> SceneConfig:
    return SceneConfig.load("configs/scene/default.yaml")


def _transition(
    frame_index: int,
    timestamp_ns: int,
    *,
    score: float = 0.8,
    hard: bool = False,
    sources: tuple[str, ...] = ("ssim",),
    evidence: tuple[str, ...] = (),
) -> TransitionProposal:
    return TransitionProposal(
        frame_index=frame_index,
        timestamp_ns=timestamp_ns,
        score=score,
        is_hard_cut=hard,
        sources=sources,
        evidence_uris=evidence,
    )


def test_no_boundaries_produces_one_complete_scene(config: SceneConfig) -> None:
    fusion = SceneBoundaryFusion(config.fusion, config_hash="config-hash")

    scenes = fusion.fuse([], [], start_ns=0, end_ns=10 * SECOND, fps=10.0)

    assert isinstance(fusion, BoundaryFusion)
    assert len(scenes) == 1
    assert scenes[0].start_ns == 0
    assert scenes[0].end_ns == 10 * SECOND
    assert scenes[0].sources == ()
    assert scenes[0].boundary_scores == {}
    assert scenes[0].confidence == 0.0
    assert scenes[0].config_hash == "config-hash"


def test_hard_cut_is_kept_even_when_it_creates_short_scene(
    config: SceneConfig,
) -> None:
    fusion = SceneBoundaryFusion(config.fusion)
    hard_cut = _transition(
        10,
        SECOND,
        hard=True,
        sources=("ssim", "optical_flow"),
    )

    scenes = fusion.fuse(
        [hard_cut],
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert [(scene.start_ns, scene.end_ns) for scene in scenes] == [
        (0, SECOND),
        (SECOND, 10 * SECOND),
    ]
    assert scenes[0].short_span is True
    assert scenes[1].short_span is False


def test_two_adjacent_hard_cuts_are_both_preserved(config: SceneConfig) -> None:
    fusion = SceneBoundaryFusion(config.fusion)
    hard_cuts = [
        _transition(40, 4 * SECOND, hard=True),
        _transition(41, 4 * SECOND + 100_000_000, hard=True),
    ]

    scenes = fusion.fuse(
        hard_cuts,
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert len(scenes) == 3
    assert scenes[1].start_ns == 4 * SECOND
    assert scenes[1].end_ns == 4 * SECOND + 100_000_000
    assert scenes[1].short_span is True


def test_soft_boundary_shorter_than_minimum_duration_is_suppressed(
    config: SceneConfig,
) -> None:
    fusion = SceneBoundaryFusion(config.fusion)
    candidates = [
        _transition(10, SECOND, score=0.9),
        _transition(50, 5 * SECOND, score=0.7),
    ]

    scenes = fusion.fuse(
        candidates,
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert [(scene.start_ns, scene.end_ns) for scene in scenes] == [
        (0, 5 * SECOND),
        (5 * SECOND, 10 * SECOND),
    ]


def test_hysteresis_merges_stage_a_and_dino_evidence(config: SceneConfig) -> None:
    fusion = SceneBoundaryFusion(config.fusion, config_hash="abc")
    transition = _transition(
        50,
        5 * SECOND,
        score=0.7,
        evidence=("boundary.jpg",),
    )
    semantic = BoundaryScore(
        frame_index=51,
        timestamp_ns=5 * SECOND + 100_000_000,
        score=0.9,
        z_score=3.0,
    )

    scenes = fusion.fuse(
        [transition],
        [semantic],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert len(scenes) == 2
    assert scenes[0].end_ns == semantic.timestamp_ns
    assert scenes[0].sources == ("ssim", "dino")
    assert scenes[0].boundary_scores == {"ssim": 0.7, "dino": 0.9}
    assert scenes[0].evidence_uris == ("boundary.jpg",)
    assert scenes[0].config_hash == "abc"


def test_same_semantic_centers_merge_soft_boundary(config: SceneConfig) -> None:
    provider = lambda _timestamp: np.array([1.0, 0.0])
    fusion = SceneBoundaryFusion(
        config.fusion,
        center_embedding_provider=provider,
    )

    scenes = fusion.fuse(
        [_transition(50, 5 * SECOND)],
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert len(scenes) == 1
    assert (scenes[0].start_ns, scenes[0].end_ns) == (0, 10 * SECOND)


def test_same_semantic_centers_never_remove_hard_cut(config: SceneConfig) -> None:
    provider = lambda _timestamp: np.array([1.0, 0.0])
    fusion = SceneBoundaryFusion(
        config.fusion,
        center_embedding_provider=provider,
    )

    scenes = fusion.fuse(
        [_transition(50, 5 * SECOND, hard=True)],
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert len(scenes) == 2


def test_different_semantic_centers_keep_soft_boundary(config: SceneConfig) -> None:
    def provider(timestamp: int) -> np.ndarray:
        if timestamp < 5 * SECOND:
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])

    fusion = SceneBoundaryFusion(
        config.fusion,
        center_embedding_provider=provider,
    )

    scenes = fusion.fuse(
        [_transition(50, 5 * SECOND)],
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert len(scenes) == 2


def test_output_intervals_are_monotonic_and_non_overlapping(
    config: SceneConfig,
) -> None:
    fusion = SceneBoundaryFusion(config.fusion)
    transitions = [
        _transition(40, 4 * SECOND),
        _transition(70, 7 * SECOND),
    ]

    scenes = fusion.fuse(
        transitions,
        [],
        start_ns=0,
        end_ns=10 * SECOND,
        fps=10.0,
    )

    assert scenes[0].start_ns == 0
    assert scenes[-1].end_ns == 10 * SECOND
    assert all(
        current.end_ns == following.start_ns
        for current, following in pairwise(scenes)
    )
    assert all(scene.start_ns < scene.end_ns for scene in scenes)


def test_candidate_outside_input_interval_is_rejected(config: SceneConfig) -> None:
    fusion = SceneBoundaryFusion(config.fusion)
    with pytest.raises(ValueError, match="超出"):
        fusion.fuse(
            [_transition(110, 11 * SECOND)],
            [],
            start_ns=0,
            end_ns=10 * SECOND,
            fps=10.0,
        )
