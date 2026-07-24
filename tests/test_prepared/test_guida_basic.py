import csv
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from zpds.cli import main
from zpds.config import PipelineConfigLoader
from zpds.prepared import GuidaBasicCleaner, PreparedValidator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"
GUIDA_THRESHOLDS = ROOT / "configs" / "qc_thresholds" / "guida_ego.yaml"


def _write_video(path: Path, *, frames: int, offset: int) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (16, 12),
    )
    assert writer.isOpened()
    for index in range(frames):
        image = np.full((12, 16, 3), (index * 13 + offset) % 255, dtype=np.uint8)
        image[:, index % 16, :] = 255 - image[:, index % 16, :]
        writer.write(image)
    writer.release()


def _make_guida_session(
    raw_root: Path,
    *,
    imu_gap: bool = False,
    imu_offset_ns: int = 0,
) -> Path:
    session = raw_root / "session"
    (session / "imu").mkdir(parents=True)
    frame_count = 12
    start_ns = 1_000_000_000
    (session / "meta.json").write_text(
        json.dumps(
            {
                "session": {"output_folder_name": "guida_clean_demo"},
                "streams": {
                    "color": {
                        "enabled": True,
                        "format": "MJPG",
                        "fps": 10,
                        "width": 16,
                        "height": 12,
                        "intrinsics": {
                            "fx": 8,
                            "fy": 8,
                            "cx": 8,
                            "cy": 6,
                            "width": 16,
                            "height": 12,
                        },
                    },
                    "depth": {
                        "enabled": True,
                        "format": "Y16",
                        "fps": 10,
                        "width": 16,
                        "height": 12,
                    },
                },
                "imu": {
                    "csv": "imu/imu_000000.csv",
                    "sample_rate_hz": 20,
                    "accel_unit": "m/s^2",
                    "gyro_unit": "rad/s",
                },
            }
        ),
        encoding="utf-8",
    )
    records = [
        {"schema": "guida.video_container.v2.raw"},
        {
            "type": "segment_start",
            "segment": 0,
            "color_video": "color_000000.mkv",
            "depth_video": "depth_000000.mkv",
        },
        *[
            {
                "type": "frame",
                "segment": 0,
                "seq": index,
                "timestamp_ns": start_ns + index * 100_000_000,
            }
            for index in range(frame_count)
        ],
    ]
    (session / "index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    _write_video(session / "color_000000.mkv", frames=frame_count, offset=10)
    _write_video(session / "depth_000000.mkv", frames=frame_count, offset=30)
    with (session / "imu" / "imu_000000.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.writer(file, lineterminator="\n")
        writer.writerow(["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"])
        for index in range(25):
            timestamp = start_ns + imu_offset_ns + index * 50_000_000
            if imu_gap and index >= 12:
                timestamp += 200_000_000
            writer.writerow([timestamp, 0.0, 0.0, 9.8, 0.0, 0.0, 0.0])
            writer.writerow([timestamp, 0.0, 0.0, 9.8, 0.01, 0.0, 0.0])
    return session


def _config():
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    value["segmentation"]["min_segment_duration_s"] = 0.2
    return PipelineConfigLoader().load_mapping(value)


def test_guida_basic_clean_writes_traceable_prepared_revision(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    session = _make_guida_session(raw_root)
    cleaner = GuidaBasicCleaner(
        pipeline_config=_config(),
        thresholds_path=GUIDA_THRESHOLDS,
        code_version="test-wp3",
    )

    result = cleaner.clean(
        session,
        tmp_path / "dataset",
        raw_session_uri="raw://session",
    )

    assert result.source_frame_count == 12
    assert result.imu_sample_count == 50
    assert result.segment_ids == ("seg_guida_clean_demo_0001",)
    segment_dir = result.revision_dir / result.segment_ids[0]
    assert PreparedValidator().validate(segment_dir, raw_root=raw_root) == []
    segment = json.loads((segment_dir / "segment.json").read_text(encoding="utf-8"))
    assert segment["quality"]["decision"] == "quarantine"
    assert segment["source_span"]["start_ns"] == 1_000_000_000
    video_map = json.loads(
        (segment_dir / "alignments" / "video_source_map.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(video_map["rows"]) == 12
    assert video_map["rows"][0]["source_seq"] == 0
    assert video_map["rows"][-1]["source_seq"] == 11
    imu_map = json.loads(
        (segment_dir / "alignments" / "imu_source_map.json").read_text(
            encoding="utf-8"
        )
    )
    assert imu_map["map_type"] == "imu_source_map"
    assert len(imu_map["rows"]) == 48
    assert [row["source_row"] for row in imu_map["rows"]] == list(range(48))
    video_imu = json.loads(
        (segment_dir / "alignments" / "video_imu_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    assert video_imu["map_type"] == "video_imu_alignment"
    assert max(row["error_ns"] for row in video_imu["rows"]) == 0


def test_guida_basic_clean_splits_at_hard_imu_gap(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    session = _make_guida_session(raw_root, imu_gap=True)
    cleaner = GuidaBasicCleaner(
        pipeline_config=_config(),
        thresholds_path=GUIDA_THRESHOLDS,
        code_version="test-wp3",
    )

    result = cleaner.clean(
        session,
        tmp_path / "dataset",
        raw_session_uri="raw://session",
    )

    assert any(issue.code == "imu_gap" for issue in result.issues)
    assert result.removed_spans
    assert len(result.segment_ids) >= 1


def test_guida_basic_clean_reports_segment_alignment_issue(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    session = _make_guida_session(raw_root, imu_offset_ns=25_000_000)
    cleaner = GuidaBasicCleaner(
        pipeline_config=_config(),
        thresholds_path=GUIDA_THRESHOLDS,
        code_version="test-wp3",
    )

    result = cleaner.clean(
        session,
        tmp_path / "dataset",
        raw_session_uri="raw://session",
    )

    assert any(issue.code == "clock_misalign" for issue in result.issues)
    report = json.loads(
        (result.revision_dir / "cleaning_report.json").read_text(encoding="utf-8")
    )
    assert any(issue["code"] == "clock_misalign" for issue in report["issues"])


def test_guida_clean_and_prepared_validate_cli(tmp_path: Path, capsys) -> None:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_guida_session(raw_root)
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["segmentation"]["min_segment_duration_s"] = 0.2
    config_path = tmp_path / "pipeline.yaml"
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    dataset = tmp_path / "dataset"

    clean_exit = main(
        [
            "clean",
            "guida",
            "--config",
            str(config_path),
            "--raw-root",
            str(raw_root),
            "--input-ref",
            "raw://session",
            "--output-root",
            str(dataset),
            "--code-version",
            "test-wp3",
        ]
    )
    clean_output = json.loads(capsys.readouterr().out)

    assert clean_exit == 0
    assert clean_output["status"] == "completed"
    segment_dir = (
        Path(clean_output["revision_dir"]) / clean_output["segment_ids"][0]
    )
    validate_exit = main(
        [
            "prepared",
            "validate",
            "--segment-dir",
            str(segment_dir),
            "--raw-root",
            str(raw_root),
        ]
    )
    validate_output = json.loads(capsys.readouterr().out)

    assert validate_exit == 0
    assert validate_output["status"] == "valid"
    assert validate_output["raw_hashes_checked"] is True
