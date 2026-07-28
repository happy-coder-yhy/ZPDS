from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from batch_prepare import generate_segment
from segment.calibration import extract_calibration
from segment.depth_writer import build_depth_sample_map, write_depth_stream
from segment.segment_writer import build_segment_json
from segment.validator import validate_depth_streams
from zpds_prepare.readers.guida_reader import read_session


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_session(root: Path, *, dtype: str = "uint16", unit: str = "mm"):
    meta = {
        "device": {"name": "guida-test"},
        "streams": {
            "color": {
                "fps": 10.0,
                "width": 4,
                "height": 3,
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.5},
            },
            "depth": {
                "path": "depth",
                "fps": 10.0,
                "frame_count": 3,
                "width": 4,
                "height": 3,
                "dtype": dtype,
                "unit": unit,
                "invalid_value": 0,
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.5},
            },
        },
        "recording_stats": {"total_frames": 3, "dropped_frames": 0},
        "imu": {"sample_rate_hz": 100.0},
    }
    (root / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    timestamps = (1_000_000_000, 1_100_000_000, 1_200_000_000)
    (root / "index.jsonl").write_text(
        "\n".join(
            json.dumps(
                {"type": "frame", "seq": index, "segment": 0, "timestamp_ns": ts}
            )
            for index, ts in enumerate(timestamps)
        ),
        encoding="utf-8",
    )
    (root / "color_000000.mkv").write_bytes(b"fixture-placeholder")
    (root / "imu").mkdir()
    (root / "imu" / "imu_000000.csv").write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n"
        "1000000000,0,0,9.8,0,0,0\n",
        encoding="utf-8",
    )
    (root / "depth").mkdir()
    for index in range(3):
        image = np.full((3, 4), 100 + index, dtype=np.uint16)
        image[0, 0] = 0
        assert cv2.imwrite(str(root / "depth" / f"{index:08d}.png"), image)
    return read_session(str(root), require_depth=True)


def test_build_depth_sample_map_preserves_source_rate(tmp_path: Path) -> None:
    session = _build_session(tmp_path)
    stream = session.depth_streams["ego_depth"]

    sample_map = build_depth_sample_map(
        stream,
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
    )

    assert sample_map["output_frame_index"].tolist() == [0, 1, 2]
    assert sample_map["output_timestamp_ns"].tolist() == [0, 100_000_000, 200_000_000]
    assert sample_map["source_frame_index"].tolist() == [0, 1, 2]
    assert sample_map["mapping_method"].unique().tolist() == ["identity"]
    assert sample_map["time_error_ns"].tolist() == [0, 0, 0]


def test_write_and_validate_guida_depth_stream(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    session = _build_session(raw_dir)
    stream = session.depth_streams["ego_depth"]
    segment_dir = tmp_path / "prepared" / "seg_000001"

    result = write_depth_stream(
        stream,
        output_dir=str(segment_dir),
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
    )
    segment = build_segment_json(
        dataset_path=str(raw_dir),
        span={
            "source_start_ns": 1_000_000_000,
            "source_end_ns": 1_200_000_000,
        },
        depth_results=[result],
        source_assets=[
            {
                "source_asset_id": "raw_depth_0",
                "uri": "depth",
                "sha256": "fixture",
            }
        ],
    )

    report = validate_depth_streams(segment_dir, segment)

    assert report["status"] == "pass"
    assert report["checks"]["depth_streams_valid"] == "pass"
    assert report["statistics"]["depth_frames_ego_depth"] == 3
    sample_map = pd.read_parquet(
        segment_dir / "maps" / "ego_depth_sample_map.parquet"
    )
    assert len(sample_map) == 3
    output = cv2.imread(
        str(segment_dir / "data" / "depth" / "ego_depth" / "00000000.png"),
        cv2.IMREAD_UNCHANGED,
    )
    assert output.dtype == np.uint16


def test_write_depth_reuses_complete_existing_output(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    stream = _build_session(raw_dir).depth_streams["ego_depth"]
    segment_dir = tmp_path / "prepared" / "seg_000001"

    first = write_depth_stream(
        stream,
        output_dir=str(segment_dir),
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
    )
    second = write_depth_stream(
        stream,
        output_dir=str(segment_dir),
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
    )

    assert first.get("cached") is None
    assert second["cached"] is True
    assert second["frames"] == first["frames"] == 3
    assert second["zero_ratio"] == first["zero_ratio"]


def test_write_depth_rejects_declared_dtype_mismatch(tmp_path: Path) -> None:
    session = _build_session(tmp_path, dtype="uint8")
    stream = session.depth_streams["ego_depth"]

    with pytest.raises(ValueError, match="dtype 与 meta.json 不一致"):
        write_depth_stream(
            stream,
            output_dir=str(tmp_path / "prepared"),
            source_start_ns=1_000_000_000,
            source_end_ns=1_200_000_000,
        )


def test_write_depth_from_lossless_mkv(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _build_session(raw_dir)
    source_images = raw_dir / "depth"
    depth_mkv = raw_dir / "depth_000000.mkv"
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(source_images / "%08d.png"),
            "-c:v",
            "ffv1",
            "-pix_fmt",
            "gray16le",
            str(depth_mkv),
        ],
        check=True,
        timeout=60,
    )
    shutil.rmtree(source_images)
    meta_path = raw_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["streams"]["depth"]["path"] = "depth_000000.mkv"
    meta_path.write_text(json.dumps(meta), encoding="utf-8")

    session = read_session(str(raw_dir), require_depth=True)
    assert session.depth_streams["ego_depth"].source_kind == "video"

    result = write_depth_stream(
        session.depth_streams["ego_depth"],
        output_dir=str(tmp_path / "prepared" / "seg_000001"),
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
    )

    assert result["dtype"] == "uint16"
    assert result["frames"] == 3


def test_generate_segment_includes_formal_guida_depth(tmp_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        pytest.skip("ffmpeg is not installed")

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    _build_session(raw_dir)

    rgb_source = raw_dir / "rgb_source"
    rgb_source.mkdir()
    for index in range(3):
        image = np.full((3, 4, 3), 20 + index, dtype=np.uint8)
        assert cv2.imwrite(str(rgb_source / f"{index:08d}.png"), image)
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            "10",
            "-i",
            str(rgb_source / "%08d.png"),
            "-c:v",
            "ffv1",
            str(raw_dir / "color_000000.mkv"),
        ],
        check=True,
        timeout=60,
    )
    shutil.rmtree(rgb_source)

    session = read_session(str(raw_dir), require_depth=True)
    calibration = extract_calibration(str(raw_dir / "meta.json"))
    segment_dir = tmp_path / "prepared" / "seg_000001"
    source_assets = [
        {
            "source_asset_id": "raw_color_0",
            "uri": "color_000000.mkv",
            "sha256": _sha256(raw_dir / "color_000000.mkv"),
        },
        {"source_asset_id": "raw_depth_0", "uri": "depth", "sha256": "y"},
        {
            "source_asset_id": "raw_index",
            "uri": "index.jsonl",
            "sha256": _sha256(raw_dir / "index.jsonl"),
        },
        {
            "source_asset_id": "raw_imu_0",
            "uri": "imu/imu_000000.csv",
            "sha256": _sha256(raw_dir / "imu" / "imu_000000.csv"),
        },
        {
            "source_asset_id": "raw_meta",
            "uri": "meta.json",
            "sha256": _sha256(raw_dir / "meta.json"),
        },
    ]

    result = generate_segment(
        dataset_path=str(raw_dir),
        source_start_ns=1_000_000_000,
        source_end_ns=1_200_000_000,
        segment_id="seg_000001",
        output_dir=str(segment_dir),
        session=session,
        calibration=calibration,
        cfg={"output": {"target_fps": 10.0}},
        source_assets=source_assets,
        profile="guida",
    )

    assert result["status"] == "pass"
    assert result["depth_frames"] == 3
    segment = json.loads((segment_dir / "segment.json").read_text(encoding="utf-8"))
    depth_stream = next(
        stream for stream in segment["streams"] if stream["stream_id"] == "ego_depth"
    )
    assert depth_stream["format"] == "png_sequence"
    assert depth_stream["dtype"] == "uint16"
    assert depth_stream["unit"] == "mm"
    assert depth_stream["origin"]["sample_map_uri"] == (
        "maps/ego_depth_sample_map.parquet"
    )
    imu_stream = next(
        stream for stream in segment["streams"] if stream["stream_id"] == "ego_imu"
    )
    assert imu_stream["origin"]["source_asset_id"] == "raw_imu_0"
