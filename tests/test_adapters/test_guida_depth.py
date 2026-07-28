from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from zpds_prepare.readers.guida_reader import read_depth_stream, read_session


def _write_guida_fixture(
    root: Path,
    *,
    depth_unit: str | None = "mm",
    depth_dtype: str | None = "uint16",
    include_depth: bool = True,
) -> None:
    depth_meta: dict[str, object] = {
        "path": "depth",
        "fps": 10.0,
        "frame_count": 3,
        "width": 4,
        "height": 3,
        "invalid_value": 0,
        "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.5},
    }
    if depth_dtype is None:
        depth_meta["format"] = "Y16"
    else:
        depth_meta["dtype"] = depth_dtype
    if depth_unit is not None:
        depth_meta["unit"] = depth_unit

    meta = {
        "device": {"name": "guida-test"},
        "streams": {
            "color": {
                "fps": 10.0,
                "width": 4,
                "height": 3,
                "intrinsics": {"fx": 1.0, "fy": 1.0, "cx": 2.0, "cy": 1.5},
            },
            "depth": depth_meta,
        },
        "recording_stats": {"total_frames": 3, "dropped_frames": 0},
        "imu": {"sample_rate_hz": 100.0},
    }
    (root / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False),
        encoding="utf-8",
    )
    frames = [
        {"type": "frame", "seq": index, "segment": 0, "timestamp_ns": timestamp}
        for index, timestamp in enumerate((1_000_000_000, 1_100_000_000, 1_200_000_000))
    ]
    (root / "index.jsonl").write_text(
        "\n".join(json.dumps(frame) for frame in frames),
        encoding="utf-8",
    )
    (root / "color_000000.mkv").write_bytes(b"fixture-placeholder")
    imu_dir = root / "imu"
    imu_dir.mkdir()
    (imu_dir / "imu_000000.csv").write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n"
        "1000000000,0,0,9.8,0,0,0\n",
        encoding="utf-8",
    )

    if include_depth:
        depth_dir = root / "depth"
        depth_dir.mkdir()
        for index in range(3):
            image = np.full((3, 4), index + 1, dtype=np.uint16)
            image[0, 0] = 0
            assert cv2.imwrite(str(depth_dir / f"{index:08d}.png"), image)


def test_read_session_discovers_guida_depth_sequence(tmp_path: Path) -> None:
    _write_guida_fixture(tmp_path)

    session = read_session(str(tmp_path), require_depth=True)

    assert list(session.video_streams) == ["ego_rgb"]
    assert list(session.depth_streams) == ["ego_depth"]
    depth = session.depth_streams["ego_depth"]
    assert depth.source_kind == "image_sequence"
    assert depth.frame_count == 3
    assert depth.dtype == "uint16"
    assert depth.unit == "mm"
    assert len(depth.source_files) == 3
    assert depth.timestamps_ns == [1_000_000_000, 1_100_000_000, 1_200_000_000]


def test_read_depth_stream_can_keep_unknown_unit(tmp_path: Path) -> None:
    _write_guida_fixture(tmp_path, depth_unit=None)

    depth = read_depth_stream(str(tmp_path), required=True)

    assert depth is not None
    assert depth.unit == "unknown"
    assert depth.metadata["unit_status"] == "unverified"


def test_read_depth_stream_maps_y16_to_uint16(tmp_path: Path) -> None:
    _write_guida_fixture(tmp_path, depth_dtype=None)

    depth = read_depth_stream(str(tmp_path), required=True)

    assert depth is not None
    assert depth.dtype == "uint16"


def test_read_depth_stream_required_rejects_missing_asset(tmp_path: Path) -> None:
    _write_guida_fixture(tmp_path, include_depth=False)

    with pytest.raises(FileNotFoundError, match="未找到原始深度资产"):
        read_depth_stream(str(tmp_path), required=True)
