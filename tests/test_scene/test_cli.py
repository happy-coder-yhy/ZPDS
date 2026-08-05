from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.run_scene_detection import build_parser, read_video, run, run_scene_detection
from zpds.scene.backends.dino import DinoV2SmallEmbedder
from zpds.scene.config import SceneConfig
from zpds.scene.sampling import (
    extract_representative_frames,
    representative_frame_indices,
)
from zpds.scene.schemas import SceneProposal
from zpds.scene.testing import hard_cut_fixture


def _write_video(path: Path, frames, fps: float) -> None:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame in frames:
        writer.write(frame)
    writer.release()


def _semantic_embeddings(frames_rgb) -> np.ndarray:
    result = np.zeros((len(frames_rgb), 384), dtype=np.float32)
    for index, frame in enumerate(frames_rgb):
        result[index, 0 if float(np.mean(frame)) < 128.0 else 1] = 1.0
    return result


def test_read_video_and_stage1_cli_json(tmp_path: Path) -> None:
    fixture = hard_cut_fixture()
    video_path = tmp_path / "hard-cut.mp4"
    output_path = tmp_path / "scene-result.json"
    _write_video(video_path, fixture.frames, fixture.fps)

    video = read_video(video_path)
    assert len(video.frames) == len(fixture.frames)
    assert video.fps == fixture.fps

    args = argparse.Namespace(
        input=str(video_path),
        config="configs/scene/default.yaml",
        profile=None,
        stage="1",
        start_ns=0,
        max_frames=None,
        output_json=str(output_path),
    )
    assert run(args) == 0

    document = json.loads(output_path.read_text(encoding="utf-8"))
    assert document["stage"] == "1"
    assert document["frame_count"] == 20
    assert document["transitions"]
    assert any(
        abs(item["frame_index"] - 10) <= 1
        for item in document["transitions"]
    )
    assert document["semantic_boundaries"] == []
    assert document["scenes"] == []


def test_stage2_in_memory_uses_fake_embedding_without_model_download() -> None:
    config = SceneConfig.load("configs/scene/default.yaml")
    frames = [
        np.full((8, 8, 3), 0 if index < 35 else 255, dtype=np.uint8)
        for index in range(70)
    ]
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )

    result = run_scene_detection(
        frames,
        fps=1.0,
        config=config,
        stage="2",
        stage_b_backend=embedder,
    )

    assert result.transitions == ()
    assert len(result.semantic_boundaries) == 1
    assert result.semantic_boundaries[0].frame_index == 35
    assert result.scenes == ()


def test_all_stage_produces_final_scenes_with_fake_embedding() -> None:
    config = SceneConfig.load("configs/scene/default.yaml")
    frames = [
        np.full((16, 16, 3), 0 if index < 35 else 255, dtype=np.uint8)
        for index in range(70)
    ]
    embedder = DinoV2SmallEmbedder(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )

    result = run_scene_detection(
        frames,
        fps=1.0,
        config=config,
        stage="all",
        stage_b_backend=embedder,
    )

    assert len(result.semantic_boundaries) == 1
    assert len(result.scenes) >= 2
    assert result.scenes[0].start_ns == 0
    assert result.scenes[-1].end_ns == 70_000_000_000
    assert all(
        current.end_ns == following.start_ns
        for current, following in zip(result.scenes, result.scenes[1:])
    )


def test_all_stage_passes_stage1_candidates_to_dino() -> None:
    config = SceneConfig.load("configs/scene/default.yaml")
    fixture = hard_cut_fixture()

    class RecordingDino(DinoV2SmallEmbedder):
        candidate_frames = None

        def detect(
            self,
            frames_bgr,
            *,
            fps,
            start_timestamp_ns=0,
            candidate_frame_indices=None,
        ):
            self.candidate_frames = candidate_frame_indices
            return super().detect(
                frames_bgr,
                fps=fps,
                start_timestamp_ns=start_timestamp_ns,
                candidate_frame_indices=candidate_frame_indices,
            )

    embedder = RecordingDino(
        config.stage_b,
        embedding_function=_semantic_embeddings,
    )

    run_scene_detection(
        fixture.frames,
        fps=fixture.fps,
        config=config,
        stage="all",
        stage_b_backend=embedder,
    )

    assert embedder.candidate_frames is not None
    assert any(abs(index - 10) <= 1 for index in embedder.candidate_frames)


def test_scene_disabled_skips_all_backends(tmp_path: Path) -> None:
    document = yaml.safe_load(
        Path("configs/scene/default.yaml").read_text(encoding="utf-8")
    )
    document["scene"]["enabled"] = False
    config_path = tmp_path / "disabled.yaml"
    config_path.write_text(yaml.safe_dump(document), encoding="utf-8")
    config = SceneConfig.load(config_path)

    result = run_scene_detection(
        [np.zeros((8, 8, 3), dtype=np.uint8)],
        fps=10.0,
        config=config,
        stage="all",
    )

    assert result.skipped is True
    assert result.skip_reason == "scene.enabled=false"
    assert result.transitions == ()
    assert result.semantic_boundaries == ()
    assert result.scenes == ()


def test_representative_frames_are_first_middle_and_last() -> None:
    scene = SceneProposal(
        scene_id="scene_000001",
        start_ns=2_000_000_000,
        end_ns=8_000_000_000,
        confidence=0.8,
        sources=("dino",),
        boundary_scores={"dino": 0.8},
    )
    frames = [np.full((2, 2, 3), index, dtype=np.uint8) for index in range(100)]

    indices = representative_frame_indices(scene, fps=10.0, frame_count=100)
    selected = extract_representative_frames(frames, scene, fps=10.0)

    assert indices == (20, 49, 79)
    assert tuple(int(frame[0, 0, 0]) for frame in selected) == indices


def test_short_scene_reuses_single_representative_frame() -> None:
    scene = SceneProposal(
        scene_id="scene_000001",
        start_ns=0,
        end_ns=50_000_000,
        confidence=0.0,
        sources=(),
        boundary_scores={},
    )

    assert representative_frame_indices(scene, fps=10.0, frame_count=10) == (0, 0, 0)


def test_with_vlm_is_reserved_and_never_fabricates_reviews() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "unused.mp4",
            "--stage",
            "all",
            "--with-vlm",
        ]
    )

    with pytest.raises(RuntimeError, match="拒绝伪造复核结果"):
        run(args)


def test_with_vlm_rejects_non_all_stage() -> None:
    args = build_parser().parse_args(
        [
            "--input",
            "unused.mp4",
            "--stage",
            "1",
            "--with-vlm",
        ]
    )

    with pytest.raises(ValueError, match="仅可与 --stage all"):
        run(args)
