import argparse
import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import yaml

from scripts import batch_run_hands, run_hands
from zpds.hands.config import HandsPipelineConfig
from zpds.hands.estimator_factory import (
    EstimatorRuntime,
    EstimatorUnavailableError,
    create_hand_estimator,
)
from zpds.hands.orchestration import (
    FrameWriterUnavailableError,
    InferenceWriterBundle,
    create_inference_writers,
)
from zpds.hands.schemas import (
    HandBBox,
    HandKeypoints,
    PreparedFrame,
    RawHandResult,
)
from zpds.hands.testing import FakeHandEstimator


class _FakeReader:
    def __init__(self, _segment_dir: Path, video_stream_id: str | None) -> None:
        self.segment_id = "seg_000001"
        self.video_stream_id = video_stream_id or "ego_rgb"
        self._frames = [
            PreparedFrame(
                frame_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
                output_frame_index=index,
                timestamp_ns=index * 33_333_333,
                source_frame_index=index + 10,
                source_timestamp_ns=1_000_000_000 + index * 33_333_333,
            )
            for index in range(3)
        ]

    @property
    def expected_frame_count(self) -> int:
        return len(self._frames)

    def __iter__(self) -> Iterator[PreparedFrame]:
        return iter(self._frames)


class _PersistingWriter:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.records = []

    def write(self, record) -> None:
        self.records.append(record)

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_bytes(b"stage3-test-artifact")


def _hand() -> RawHandResult:
    return RawHandResult(
        handedness="Left",
        handedness_score=0.9,
        keypoints=HandKeypoints(
            normalized=[(0.1, 0.1, 0.0)] * 21,
            pixel=[(3.0, 4.0)] * 21,
        ),
        bbox=HandBBox(1.0, 2.0, 10.0, 12.0),
        detection_score=0.8,
    )


def _write_config(tmp_path: Path) -> Path:
    document = {
        "hands": {
            "ego_bbox_backend": "wilor",
            "non_ego_bbox_backend": "mediapipe",
            "fallback_2d_backend": "mediapipe",
            "mediapipe": {
                "backend": "solutions_hands",
                "fallback_backend": "solutions_hands",
                "tasks": {
                    "model_path": "models/hand_landmarker.task",
                    "delegate": "cpu",
                },
                "solutions": {"model_complexity": 1},
            },
            "wilor": {
                "enabled": True,
                "ego_bbox_every_frame": True,
                "write_frame_status": True,
                "upstream_commit": "test-commit",
                "checkpoint_path": "models/wilor/model.pt",
                "checkpoint_sha256": "test-wilor-sha",
            },
        }
    }
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _args(tmp_path: Path, *, max_frames: int | None = None) -> argparse.Namespace:
    values = [
        "--segment",
        str(tmp_path),
        "--stream-id",
        "ego_rgb",
        "--source-kind",
        "ego",
        "--config",
        str(_write_config(tmp_path)),
        "--output",
        str(tmp_path / "outputs" / "hands_2d.parquet"),
    ]
    if max_frames is not None:
        values.extend(["--max-frames", str(max_frames)])
    return run_hands.build_parser().parse_args(values)


def test_wilor_cli_orchestration_writes_full_frame_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path)
    estimator = FakeHandEstimator(
        [[_hand()], [], RuntimeError("single-frame failure")]
    )
    writers: dict[str, _PersistingWriter] = {}

    def estimator_factory(
        primary_model: str,
        _config,
    ) -> EstimatorRuntime:
        assert primary_model == "wilor"
        return EstimatorRuntime(
            estimator=estimator,
            model_name="wilor",
            model_version="test",
            checkpoint_sha256="test-wilor-sha",
            upstream_git_commit="test-commit",
            active_backend="wilor",
        )

    def writer_factory(
        primary_model: str,
        *,
        frame_status_path: str,
        bbox_path: str,
    ) -> InferenceWriterBundle:
        assert primary_model == "wilor"
        writers["status"] = _PersistingWriter(frame_status_path)
        writers["bbox"] = _PersistingWriter(bbox_path)
        return InferenceWriterBundle(
            frame_status=writers["status"],
            bbox=writers["bbox"],
        )

    monkeypatch.setattr(run_hands, "PreparedSegmentReader", _FakeReader)
    monkeypatch.setattr(
        run_hands,
        "_read_segment_json",
        lambda _path: {"record_revision": "r0001"},
    )
    monkeypatch.setattr(
        run_hands,
        "_image_dimensions",
        lambda *_args: (32, 24),
    )

    assert run_hands.run(
        args,
        estimator_factory=estimator_factory,
        inference_writer_factory=writer_factory,
        verify_wilor_assets=False,
    ) == 0

    manifest = json.loads(
        (tmp_path / "outputs" / "hands_run.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["primary_model"] == "wilor"
    assert manifest["run_mode"] == "production"
    assert manifest["completed"] is True
    assert manifest["full_frame_coverage"] is True
    assert manifest["wilor_requirement_satisfied"] is True
    assert manifest["statistics"]["frame_status"] == {
        "requested": 3,
        "detected": 1,
        "no_hand": 1,
        "failed": 1,
        "skipped_invalid_input": 0,
    }
    assert len(writers["status"].records) == 3
    assert len(writers["bbox"].records) == 3
    assert estimator.closed
    assert not (tmp_path / "outputs" / "hands_2d.parquet").exists()


def test_max_frames_is_recorded_as_incomplete_smoke_run(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, max_frames=2)
    estimator = FakeHandEstimator([[], [], []])

    def estimator_factory(_primary_model: str, _config) -> EstimatorRuntime:
        return EstimatorRuntime(
            estimator=estimator,
            model_name="wilor",
            model_version="test",
            checkpoint_sha256="test-wilor-sha",
            upstream_git_commit="test-commit",
            active_backend="wilor",
        )

    def writer_factory(
        _primary_model: str,
        *,
        frame_status_path: str,
        bbox_path: str,
    ) -> InferenceWriterBundle:
        return InferenceWriterBundle(
            frame_status=_PersistingWriter(frame_status_path),
            bbox=_PersistingWriter(bbox_path),
        )

    monkeypatch.setattr(run_hands, "PreparedSegmentReader", _FakeReader)
    monkeypatch.setattr(
        run_hands,
        "_read_segment_json",
        lambda _path: {"record_revision": "r0001"},
    )
    monkeypatch.setattr(
        run_hands,
        "_image_dimensions",
        lambda *_args: (32, 24),
    )

    assert run_hands.run(
        args,
        estimator_factory=estimator_factory,
        inference_writer_factory=writer_factory,
        verify_wilor_assets=False,
    ) == 0
    manifest = json.loads(
        (tmp_path / "outputs" / "hands_run.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["run_mode"] == "smoke"
    assert manifest["completed"] is False
    assert manifest["full_frame_coverage"] is False
    assert manifest["statistics"]["frame_status"]["requested"] == 2


def test_wilor_runtime_contract_failure_closes_estimator(
    tmp_path: Path,
    monkeypatch,
) -> None:
    args = _args(tmp_path, max_frames=1)
    estimator = FakeHandEstimator([[]])

    def estimator_factory(_primary_model: str, _config) -> EstimatorRuntime:
        return EstimatorRuntime(
            estimator=estimator,
            model_name="wilor",
            model_version="test",
            checkpoint_sha256="wrong-sha",
            upstream_git_commit="test-commit",
            active_backend="wilor",
        )

    monkeypatch.setattr(run_hands, "PreparedSegmentReader", _FakeReader)
    monkeypatch.setattr(
        run_hands,
        "_read_segment_json",
        lambda _path: {"record_revision": "r0001"},
    )
    monkeypatch.setattr(
        run_hands,
        "_image_dimensions",
        lambda *_args: (32, 24),
    )

    with pytest.raises(ValueError, match="checkpoint_sha256"):
        run_hands.run(
            args,
            estimator_factory=estimator_factory,
            verify_wilor_assets=False,
        )
    assert estimator.closed


def test_batch_rejects_incomplete_wilor_frame_statistics(
    tmp_path: Path,
) -> None:
    segment_dir = tmp_path / "seg_000001"
    segment_dir.mkdir()
    paths = batch_run_hands._output_paths(
        tmp_path / "hands",
        "seg_000001",
        "ego_rgb",
    )
    paths["directory"].mkdir(parents=True)
    paths["frame_status"].write_bytes(b"status")
    paths["bbox"].write_bytes(b"bbox")
    batch_run_hands._write_json_atomic(
        paths["manifest"],
        {
            "completed": True,
            "run_mode": "production",
            "segment_id": "seg_000001",
            "video_stream_id": "ego_rgb",
            "max_frames": None,
            "config_sha256": "config-hash",
            "checkpoint_sha256": "model-hash",
            "primary_model": "wilor",
            "upstream_git_commit": "commit",
            "wilor_requirement_satisfied": True,
            "statistics": {
                "expected_frame_count": 3,
                "frame_status": {
                    "requested": 2,
                    "detected": 1,
                    "no_hand": 1,
                    "failed": 0,
                    "skipped_invalid_input": 0,
                },
            },
        },
    )

    can_skip, reason = batch_run_hands._existing_output_can_be_skipped(
        segment_dir=segment_dir,
        segment={},
        segment_id="seg_000001",
        stream_id="ego_rgb",
        paths=paths,
        expected_config_sha256="config-hash",
        expected_checkpoint_sha256="model-hash",
        max_frames=None,
        report_required=False,
        preview_required=False,
        primary_model="wilor",
        expected_upstream_git_commit="commit",
    )

    assert can_skip is False
    assert "统计不完整" in reason


def test_default_wilor_factories_never_silently_use_mediapipe(
    tmp_path: Path,
) -> None:
    config = HandsPipelineConfig.load(_write_config(tmp_path))

    with pytest.raises(EstimatorUnavailableError, match="不能静默"):
        create_hand_estimator("wilor", config)
    with pytest.raises(FrameWriterUnavailableError, match="尚未接入"):
        create_inference_writers(
            "wilor",
            frame_status_path="status.parquet",
            bbox_path="bbox.parquet",
        )
