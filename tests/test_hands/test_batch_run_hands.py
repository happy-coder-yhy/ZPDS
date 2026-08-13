import argparse
import json
from pathlib import Path

from scripts import batch_run_hands
from zpds.hands.schemas import HandObservation
from zpds.hands.writer import write_hand_observations


def _write_segment(directory: Path, segment_id: str) -> None:
    directory.mkdir(parents=True)
    (directory / "segment.json").write_text(
        json.dumps(
            {
                "prep_revision": "r0001",
                "segment_id": segment_id,
                "timeline": {"start_ns": 0, "end_ns": 1_000_000_000},
                "streams": [
                    {
                        "stream_id": "ego_rgb",
                        "modality": "rgb",
                        "format": "mp4",
                        "shape": [24, 32, 3],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def _observation(segment_id: str) -> HandObservation:
    return HandObservation(
        segment_id=segment_id,
        video_stream_id="ego_rgb",
        output_frame_index=0,
        timestamp_ns=0,
        source_frame_index=0,
        source_timestamp_ns=0,
        detection_id=0,
        handedness="right",
        handedness_score=0.9,
        bbox_xyxy=(1.0, 1.0, 20.0, 20.0),
        keypoints_2d=[(2.0 + index / 2, 2.0 + index / 2) for index in range(21)],
        keypoints_z_relative=[0.0] * 21,
        model_name="wilor",
        model_version="test",
    )


def _arguments(
    segments_root: Path,
    output_root: Path,
    summary_path: Path,
) -> argparse.Namespace:
    return batch_run_hands.build_parser().parse_args(
        [
            "--segments-root",
            str(segments_root),
            "--config",
            str(segments_root / "config.yaml"),
            "--output-root",
            str(output_root),
            "--summary-output",
            str(summary_path),
        ]
    )


def test_discover_segments_is_sorted(tmp_path: Path) -> None:
    _write_segment(tmp_path / "seg_000002", "seg_000002")
    _write_segment(tmp_path / "seg_000001", "seg_000001")
    (tmp_path / "other").mkdir()

    discovered = batch_run_hands.discover_segments(tmp_path, "seg_*")

    assert [path.name for path in discovered] == ["seg_000001", "seg_000002"]


def test_existing_complete_output_can_be_skipped(tmp_path: Path) -> None:
    segment_dir = tmp_path / "seg_000001"
    _write_segment(segment_dir, "seg_000001")
    segment, segment_id, stream_id = batch_run_hands._segment_identity(
        segment_dir,
        None,
    )
    paths = batch_run_hands._output_paths(
        tmp_path / "hands",
        segment_dir,
        segment_id,
        stream_id,
    )
    write_hand_observations(
        [_observation(segment_id)],
        paths["parquet"],
        checkpoint_sha256="model-hash",
        config_sha256="config-hash",
    )
    paths["directory"].mkdir(parents=True, exist_ok=True)
    paths["frame_status"].touch()
    paths["bbox"].touch()
    batch_run_hands._write_json_atomic(
        paths["manifest"],
        {
            "completed": True,
            "segment_id": segment_id,
            "video_stream_id": stream_id,
            "max_frames": None,
            "config_sha256": "config-hash",
            "checkpoint_sha256": "model-hash",
            "primary_model": "wilor",
            "upstream_git_commit": "commit",
            "wilor_requirement_satisfied": True,
            "statistics": {
                "expected_frame_count": 1,
                "frame_status": {
                    "requested": 1,
                    "detected": 1,
                    "no_hand": 0,
                    "failed": 0,
                    "skipped_invalid_input": 0,
                },
            },
            "validation_status": "pass",
        },
    )

    can_skip, reason = batch_run_hands._existing_output_can_be_skipped(
        segment_dir=segment_dir,
        segment_id=segment_id,
        stream_id=stream_id,
        paths=paths,
        expected_config_sha256="config-hash",
        expected_checkpoint_sha256="model-hash",
        max_frames=None,
        expected_upstream_git_commit="commit",
    )

    assert can_skip is True
    assert "校验通过" in reason


def test_changed_max_frames_prevents_skip(tmp_path: Path) -> None:
    segment_dir = tmp_path / "seg_000001"
    _write_segment(segment_dir, "seg_000001")
    segment, segment_id, stream_id = batch_run_hands._segment_identity(
        segment_dir,
        None,
    )
    paths = batch_run_hands._output_paths(
        tmp_path / "hands",
        segment_dir,
        segment_id,
        stream_id,
    )
    paths["directory"].mkdir(parents=True)
    paths["parquet"].touch()
    paths["frame_status"].touch()
    paths["bbox"].touch()
    batch_run_hands._write_json_atomic(
        paths["manifest"],
        {
            "completed": True,
            "segment_id": segment_id,
            "video_stream_id": stream_id,
            "max_frames": 10,
            "config_sha256": "config-hash",
            "checkpoint_sha256": "model-hash",
        },
    )

    can_skip, reason = batch_run_hands._existing_output_can_be_skipped(
        segment_dir=segment_dir,
        segment_id=segment_id,
        stream_id=stream_id,
        paths=paths,
        expected_config_sha256="config-hash",
        expected_checkpoint_sha256="model-hash",
        max_frames=None,
    )

    assert can_skip is False
    assert "max_frames" in reason


def test_batch_continues_after_one_segment_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segments_root = tmp_path / "segments"
    for index in range(1, 4):
        _write_segment(
            segments_root / f"seg_{index:06d}",
            f"seg_{index:06d}",
        )
    output_root = tmp_path / "hands"
    summary_path = tmp_path / "batch_summary.json"
    args = _arguments(segments_root, output_root, summary_path)

    monkeypatch.setattr(
        batch_run_hands,
        "_expected_provenance",
        lambda config_path: (
            "config-hash",
            "model-hash",
            "commit",
        ),
    )
    monkeypatch.setattr(
        batch_run_hands,
        "_existing_output_can_be_skipped",
        lambda **kwargs: (False, "尚无可复用产物"),
    )

    def fake_single_run(single_args: argparse.Namespace) -> int:
        segment_id = Path(single_args.segment).name
        if segment_id == "seg_000002":
            raise RuntimeError("synthetic failure")
        write_hand_observations(
            [_observation(segment_id)],
            single_args.output,
            checkpoint_sha256="model-hash",
            config_sha256="config-hash",
        )
        batch_run_hands._write_json_atomic(
            Path(single_args.manifest_output),
            {
                "completed": True,
                "segment_id": segment_id,
                "video_stream_id": "ego_rgb",
                "max_frames": None,
                "config_sha256": "config-hash",
                "checkpoint_sha256": "model-hash",
                "statistics": {"frames_processed": 1},
                "validation_status": "pass",
            },
        )
        return 0

    monkeypatch.setattr(batch_run_hands.single_cli, "run", fake_single_run)

    exit_code = batch_run_hands.run(args)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert exit_code == 1
    assert summary["counts"] == {
        "total": 3,
        "completed": 2,
        "skipped": 0,
        "failed": 1,
    }
    assert [item["status"] for item in summary["items"]] == [
        "completed",
        "failed",
        "completed",
    ]
    assert "synthetic failure" in summary["items"][1]["error"]


def test_batch_skips_existing_valid_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    segments_root = tmp_path / "segments"
    _write_segment(segments_root / "seg_000001", "seg_000001")
    summary_path = tmp_path / "batch_summary.json"
    args = _arguments(segments_root, tmp_path / "hands", summary_path)

    monkeypatch.setattr(
        batch_run_hands,
        "_expected_provenance",
        lambda config_path: (
            "config-hash",
            "model-hash",
            "commit",
        ),
    )
    monkeypatch.setattr(
        batch_run_hands,
        "_existing_output_can_be_skipped",
        lambda **kwargs: (True, "已有产物校验为 pass"),
    )
    monkeypatch.setattr(
        batch_run_hands,
        "_collect_output_statistics",
        lambda paths: {"rows": 2},
    )

    def must_not_run(args: argparse.Namespace) -> int:
        raise AssertionError("有效已有结果不应重新运行")

    monkeypatch.setattr(batch_run_hands.single_cli, "run", must_not_run)

    assert batch_run_hands.run(args) == 0
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["counts"]["skipped"] == 1
    assert summary["items"][0]["reason"] == "已有产物校验为 pass"
