"""人员 B：scene 分割 + VLM 复核流水线与写出/校验测试。"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from zpds.scene.config import SceneConfig
from zpds.scene.pipeline import run_scene_pipeline
from zpds.scene.schemas import VLMReviewResult
from zpds.scene.testing import hard_cut_fixture
from zpds.scene.validator import sha256_file, validate_scene_outputs
from zpds.scene.writer import write_scene_run

DEFAULT_CONFIG = Path("configs/scene/default.yaml")


class EmptyStageB:
    """不依赖 torch 的 Stage B 替身：embedding 为帧号 one-hot。"""

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


def _fixture_frames():
    fixture = hard_cut_fixture(fps=10.0)
    return fixture.frames, fixture.fps


class TestPipeline:
    def test_end_to_end_with_consistent_vlm(self) -> None:
        frames, fps = _fixture_frames()
        config = SceneConfig.load(DEFAULT_CONFIG)
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=config,
            stage_b_backend=EmptyStageB(),
            vlm_reviewer=FakeReviewer("consistent", 0.9),
        )

        assert not run.skipped
        assert run.scenes
        assert len(run.vlm_results) == len(run.scenes)
        assert run.review_queue == ()
        names = [metric.name for metric in run.metrics]
        assert "scene_count" in names
        assert "vlm_reviewed_ratio" in names
        assert all(
            decision.reason.value != "semantic_inconsistency"
            for decision in run.decisions
        )

    def test_low_confidence_vlm_enters_review_queue(self) -> None:
        frames, fps = _fixture_frames()
        config = SceneConfig.load(DEFAULT_CONFIG)
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=config,
            stage_b_backend=EmptyStageB(),
            vlm_reviewer=FakeReviewer("consistent", 0.3),
        )

        assert run.review_queue
        assert all(
            item.confidence < config.vlm.review_confidence_threshold
            for item in run.review_queue
        )

    def test_disabled_config_skips_pipeline(self) -> None:
        frames, fps = _fixture_frames()
        config = dataclasses.replace(
            SceneConfig.load(DEFAULT_CONFIG),
            enabled=False,
        )
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=config,
            vlm_reviewer=FakeReviewer("consistent", 0.9),
        )

        assert run.skipped
        assert run.skip_reason == "scene.enabled=false"
        assert run.scenes == ()
        assert run.vlm_results == ()


class TestWriterValidator:
    def test_write_and_validate_round_trip(
        self, tmp_path: Path
    ) -> None:
        frames, fps = _fixture_frames()
        config = SceneConfig.load(DEFAULT_CONFIG)
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=config,
            stage_b_backend=EmptyStageB(),
            vlm_reviewer=FakeReviewer("consistent", 0.9),
        )
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"fake raw bytes")
        raw_before = sha256_file(raw)
        output = tmp_path / "out"
        written = write_scene_run(
            output,
            input_path=raw,
            config_hash=config.config_hash,
            profile=config.profile,
            fps=fps,
            frame_count=len(frames),
            start_ns=0,
            end_ns=run.end_ns,
            scenes=run.scenes,
            vlm_results=run.vlm_results,
            review_queue=run.review_queue,
        )

        assert written.scene_file.is_file()
        assert written.vlm_file.is_file()
        assert written.summary_file.is_file()
        scene_frame = pd.read_parquet(written.scene_file)
        assert len(scene_frame) == len(run.scenes)
        assert "start_ns" in scene_frame.columns
        assert "config_hash" in scene_frame.columns
        report = validate_scene_outputs(
            output,
            raw_path=raw,
            raw_sha256_before=raw_before,
            expected_scene_count=len(run.scenes),
        )
        assert report.ok, report.issues

    def test_skipped_run_writes_summary_only(
        self, tmp_path: Path
    ) -> None:
        raw = tmp_path / "raw.mp4"
        raw.write_bytes(b"fake raw bytes")
        output = tmp_path / "out"
        written = write_scene_run(
            output,
            input_path=raw,
            config_hash="hash",
            profile=None,
            fps=10.0,
            frame_count=20,
            start_ns=0,
            end_ns=2_000_000_000,
            scenes=(),
            vlm_results=(),
            review_queue=(),
            skipped=True,
            skip_reason="scene.enabled=false",
        )

        assert written.summary_file.is_file()
        assert not written.scene_file.exists()
        assert not written.vlm_file.exists()


class TestRunPipelineWiring:
    def test_missing_video_raises(self) -> None:
        from scripts.run_pipeline import build_parser, run

        args = build_parser().parse_args(
            [
                "--source",
                "missing.mp4",
                "--profile",
                "guida_ego",
            ]
        )
        with pytest.raises(FileNotFoundError, match="输入视频不存在"):
            run(args)
