"""Stage 10 scene 级联检查器测试。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np

from zpds.core.decisions import ReasonCode
from zpds.qc.cascade import get_stage_checker
from zpds.qc.stage10_scene import _check_stage10
from zpds.scene.config import SceneConfig
from zpds.scene.pipeline import run_scene_pipeline
from zpds.scene.schemas import VLMReviewResult
from zpds.scene.testing import hard_cut_fixture

DEFAULT_CONFIG = Path("configs/scene/default.yaml")


class EmptyStageB:
    embedding_dimension = 384

    def embed(self, frames_rgb):
        embeddings = np.zeros((len(frames_rgb), 384), dtype=np.float64)
        for index, frame in enumerate(frames_rgb):
            value = int(frame.reshape(-1)[0]) % 384
            embeddings[index, value] = 1.0
        return embeddings

    def detect(
        self,
        frames_bgr,
        *,
        fps,
        start_timestamp_ns=0,
        candidate_frame_indices=None,
    ):
        return []


class FakeReviewer:
    def __init__(self, decision: str, confidence: float) -> None:
        self._decision = decision
        self._confidence = confidence

    def review(self, scene, representative_frames_rgb):
        return VLMReviewResult(
            scene_id=scene.scene_id,
            scene_label="kitchen",
            task_label="cooking",
            decision=self._decision,  # type: ignore[arg-type]
            confidence=self._confidence,
            reasons="fake reviewer",
            config_hash=scene.config_hash,
        )


def _run_pipeline(decision: str = "consistent", confidence: float = 0.9):
    fixture = hard_cut_fixture(fps=10.0)
    # 默认配置已关闭，显式启用以验证「启用」路径
    config = dataclasses.replace(
        SceneConfig.load(DEFAULT_CONFIG), enabled=True
    )
    run = run_scene_pipeline(
        fixture.frames,
        fps=fixture.fps,
        config=config,
        stage_b_backend=EmptyStageB(),
        vlm_reviewer=FakeReviewer(decision, confidence),
    )
    return run, config


def test_stage10_checker_registered() -> None:
    assert get_stage_checker(10) is _check_stage10


def test_skip_without_scene_run() -> None:
    assert _check_stage10({}) == []


def test_skip_when_stage_disabled() -> None:
    run, config = _run_pipeline()
    assert run.scenes
    decisions = _check_stage10(
        {
            "scene_pipeline_run": run,
            "scene_config": config,
            "stage_config": {"enabled": False},
        }
    )
    assert decisions == []


def test_skip_when_pipeline_skipped() -> None:
    config = dataclasses.replace(
        SceneConfig.load(DEFAULT_CONFIG),
        enabled=False,
    )
    fixture = hard_cut_fixture(fps=10.0)
    run = run_scene_pipeline(
        fixture.frames,
        fps=fixture.fps,
        config=config,
        vlm_reviewer=FakeReviewer("consistent", 0.9),
    )
    assert run.skipped
    assert _check_stage10({"scene_pipeline_run": run, "scene_config": config}) == []


def test_inconsistent_vlm_produces_semantic_decision() -> None:
    run, config = _run_pipeline(decision="inconsistent", confidence=0.95)
    assert run.vlm_results
    decisions = _check_stage10(
        {"scene_pipeline_run": run, "scene_config": config}
    )
    assert any(
        decision.reason == ReasonCode.SEMANTIC_INCONSISTENCY
        for decision in decisions
    )
    for decision in decisions:
        assert decision.detail["config_hash"] == config.config_hash

