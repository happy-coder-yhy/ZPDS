"""人员 B：scene QC 指标与决策集成测试。"""

from __future__ import annotations

from pathlib import Path

from zpds.core.decisions import Disposition, ReasonCode
from zpds.scene.config import SceneConfig
from zpds.scene.qc_integration import (
    build_scene_decisions,
    build_scene_metrics,
)
from zpds.scene.schemas import SceneProposal, VLMReviewResult

DEFAULT_CONFIG = Path("configs/scene/default.yaml")


def _scene(
    scene_id: str,
    start_ns: int,
    confidence: float,
) -> SceneProposal:
    return SceneProposal(
        scene_id=scene_id,
        start_ns=start_ns,
        end_ns=start_ns + 5_000_000_000,
        confidence=confidence,
        sources=("dino",),
        boundary_scores={"dino": confidence},
        config_hash="hash-a",
    )


def _vlm(
    scene_id: str,
    decision: str,
    confidence: float,
) -> VLMReviewResult:
    return VLMReviewResult(
        scene_id=scene_id,
        scene_label="kitchen",
        task_label="cooking",
        decision=decision,  # type: ignore[arg-type]
        confidence=confidence,
        reasons="fake",
        config_hash="hash-a",
    )


class TestSceneMetrics:
    def test_metrics_values(self) -> None:
        scenes = [
            _scene("s1", 0, 0.9),
            _scene("s2", 5_000_000_000, 0.4),
        ]
        results = [
            _vlm("s1", "consistent", 0.9),
            _vlm("s2", "inconsistent", 0.95),
        ]
        metrics = {
            metric.name: metric
            for metric in build_scene_metrics(
                scenes,
                results,
                config_hash="hash-a",
            )
        }

        assert metrics["scene_count"].value == 2
        assert metrics["boundary_confidence_min"].value == 0.4
        assert metrics["boundary_confidence_mean"].value == 0.65
        assert metrics["vlm_consistent_ratio"].value == 0.5
        assert metrics["vlm_reviewed_ratio"].value == 1.0

    def test_empty_scenes(self) -> None:
        metrics = build_scene_metrics([], [], config_hash="hash-a")
        by_name = {metric.name: metric for metric in metrics}
        assert by_name["scene_count"].value == 0
        assert by_name["boundary_confidence_min"].value is None
        assert by_name["vlm_reviewed_ratio"].value is None


class TestSceneDecisions:
    def test_inconsistent_and_low_confidence(self) -> None:
        config = SceneConfig.load(DEFAULT_CONFIG)
        scenes = [
            _scene("s1", 0, 0.9),
            _scene("s2", 5_000_000_000, 0.4),
        ]
        results = [
            _vlm("s1", "consistent", 0.9),
            _vlm("s2", "inconsistent", 0.95),
        ]
        decisions = build_scene_decisions(
            scenes,
            results,
            config=config,
            vlm_enabled=True,
        )
        reasons = [decision.reason for decision in decisions]

        assert ReasonCode.SEMANTIC_INCONSISTENCY in reasons
        assert ReasonCode.SCENE_BOUNDARY_LOW_CONFIDENCE in reasons
        inconsistent = next(
            decision
            for decision in decisions
            if decision.reason == ReasonCode.SEMANTIC_INCONSISTENCY
        )
        low = next(
            decision
            for decision in decisions
            if decision.reason == ReasonCode.SCENE_BOUNDARY_LOW_CONFIDENCE
        )
        assert inconsistent.disposition == Disposition.QUARANTINE
        assert inconsistent.detail["producer"] == "zpds.scene.qc"
        assert inconsistent.detail["config_hash"] == "hash-a"
        assert low.disposition == Disposition.QUARANTINE
        assert low.timestamp_ns == 5_000_000_000

    def test_semantic_not_run_when_vlm_enabled_but_no_results(self) -> None:
        config = SceneConfig.load(DEFAULT_CONFIG)
        decisions = build_scene_decisions(
            [_scene("s1", 0, 0.9)],
            [],
            config=config,
            vlm_enabled=True,
        )
        assert any(
            decision.reason == ReasonCode.SEMANTIC_NOT_RUN
            for decision in decisions
        )
