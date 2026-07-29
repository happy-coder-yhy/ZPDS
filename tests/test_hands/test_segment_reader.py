import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from zpds.hands.schemas import PreparedFrame
from zpds.hands.segment_reader import (
    PreparedSegmentError,
    PreparedSegmentReader,
    SampleMapValidationError,
    StreamNotFoundError,
    VideoDecodeError,
)

FRAME_SIZE = (32, 24)
FRAME_INTERVAL_NS = 33_333_333


def _write_video(path: Path, bgr_colors: list[tuple[int, int, int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        30.0,
        FRAME_SIZE,
    )
    assert writer.isOpened()
    try:
        width, height = FRAME_SIZE
        for color in bgr_colors:
            frame = np.full((height, width, 3), color, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()


def _sample_map(row_count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "output_frame_index": pd.Series(range(row_count), dtype="int64"),
            "output_timestamp_ns": pd.Series(
                [index * FRAME_INTERVAL_NS for index in range(row_count)],
                dtype="int64",
            ),
            "source_frame_index": pd.Series(
                [index + 10 for index in range(row_count)],
                dtype="int64",
            ),
            "source_timestamp_ns": pd.Series(
                [1_000_000_000 + index * FRAME_INTERVAL_NS for index in range(row_count)],
                dtype="int64",
            ),
        }
    )


def _build_segment(
    root: Path,
    *,
    frame_colors: list[tuple[int, int, int]] | None = None,
    sample_map: pd.DataFrame | None = None,
) -> Path:
    segment_dir = root / "seg_000001"
    video_path = segment_dir / "data" / "ego_rgb.mp4"
    map_path = segment_dir / "maps" / "ego_rgb_sample_map.parquet"

    colors = (
        frame_colors
        if frame_colors is not None
        else [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
    )
    rows = sample_map if sample_map is not None else _sample_map(len(colors))

    _write_video(video_path, colors)
    map_path.parent.mkdir(parents=True, exist_ok=True)
    rows.to_parquet(map_path, index=False)

    segment = {
        "segment_id": "seg_000001",
        "streams": [
            {
                "stream_id": "ego_rgb",
                "modality": "rgb",
                "uri": "data/ego_rgb.mp4",
                "origin": {
                    "sample_map_uri": "maps/ego_rgb_sample_map.parquet",
                },
            }
        ],
    }
    (segment_dir / "segment.json").write_text(
        json.dumps(segment),
        encoding="utf-8",
    )
    return segment_dir


def _read_segment_json(segment_dir: Path) -> dict:
    return json.loads((segment_dir / "segment.json").read_text(encoding="utf-8"))


def _write_segment_json(segment_dir: Path, segment: dict) -> None:
    (segment_dir / "segment.json").write_text(json.dumps(segment), encoding="utf-8")


def test_reader_yields_rgb_frames_with_sample_map_metadata(tmp_path: Path) -> None:
    segment_dir = _build_segment(tmp_path)

    reader = PreparedSegmentReader(segment_dir)
    frames = list(reader)

    assert reader.segment_id == "seg_000001"
    assert reader.video_stream_id == "ego_rgb"
    assert len(reader) == 3
    assert all(isinstance(frame, PreparedFrame) for frame in frames)
    assert [frame.output_frame_index for frame in frames] == [0, 1, 2]
    assert [frame.timestamp_ns for frame in frames] == [
        0,
        FRAME_INTERVAL_NS,
        2 * FRAME_INTERVAL_NS,
    ]
    assert [frame.source_frame_index for frame in frames] == [10, 11, 12]
    assert [frame.source_timestamp_ns for frame in frames] == [
        1_000_000_000,
        1_000_000_000 + FRAME_INTERVAL_NS,
        1_000_000_000 + 2 * FRAME_INTERVAL_NS,
    ]

    # 第一帧写入的是 BGR 红色；Reader 输出必须转换为 RGB 红色。
    first_pixel = frames[0].frame_rgb[0, 0]
    assert int(first_pixel[0]) > 200
    assert int(first_pixel[2]) < 30


def test_reader_can_be_iterated_more_than_once(tmp_path: Path) -> None:
    reader = PreparedSegmentReader(_build_segment(tmp_path))

    assert [frame.output_frame_index for frame in reader] == [0, 1, 2]
    assert [frame.output_frame_index for frame in reader] == [0, 1, 2]


def test_reader_preserves_nullable_source_metadata(tmp_path: Path) -> None:
    sample_map = _sample_map(2)
    sample_map["source_frame_index"] = pd.Series([10, None], dtype="Int64")
    sample_map["source_timestamp_ns"] = pd.Series(
        [1_000_000_000, None],
        dtype="Int64",
    )
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0), (255, 255, 255)],
        sample_map=sample_map,
    )

    frames = list(PreparedSegmentReader(segment_dir))

    assert frames[0].source_frame_index == 10
    assert frames[1].source_frame_index is None
    assert frames[1].source_timestamp_ns is None


def test_reader_requires_stream_id_when_multiple_rgb_streams(tmp_path: Path) -> None:
    segment_dir = _build_segment(tmp_path)
    segment = _read_segment_json(segment_dir)
    second_stream = dict(segment["streams"][0])
    second_stream["stream_id"] = "wrist_rgb"
    segment["streams"].append(second_stream)
    _write_segment_json(segment_dir, segment)

    with pytest.raises(StreamNotFoundError, match="多个 RGB Stream"):
        PreparedSegmentReader(segment_dir)

    reader = PreparedSegmentReader(segment_dir, video_stream_id="wrist_rgb")
    assert reader.video_stream_id == "wrist_rgb"


@pytest.mark.parametrize("stream_id", ["missing_rgb", "imu"])
def test_reader_rejects_missing_or_non_rgb_stream(tmp_path: Path, stream_id: str) -> None:
    segment_dir = _build_segment(tmp_path)
    if stream_id == "imu":
        segment = _read_segment_json(segment_dir)
        segment["streams"].append(
            {
                "stream_id": "imu",
                "modality": "imu",
                "uri": "data/imu.parquet",
            }
        )
        _write_segment_json(segment_dir, segment)

    with pytest.raises(StreamNotFoundError):
        PreparedSegmentReader(segment_dir, video_stream_id=stream_id)


def test_reader_rejects_missing_segment_json(tmp_path: Path) -> None:
    segment_dir = tmp_path / "seg_missing_json"
    segment_dir.mkdir()

    with pytest.raises(PreparedSegmentError, match="segment.json 不存在"):
        PreparedSegmentReader(segment_dir)


def test_reader_rejects_missing_video(tmp_path: Path) -> None:
    segment_dir = _build_segment(tmp_path)
    (segment_dir / "data" / "ego_rgb.mp4").unlink()

    with pytest.raises(VideoDecodeError, match="RGB 视频不存在"):
        PreparedSegmentReader(segment_dir)


def test_reader_rejects_missing_sample_map(tmp_path: Path) -> None:
    segment_dir = _build_segment(tmp_path)
    (segment_dir / "maps" / "ego_rgb_sample_map.parquet").unlink()

    with pytest.raises(SampleMapValidationError, match="Sample Map 不存在"):
        PreparedSegmentReader(segment_dir)


def test_reader_rejects_sample_map_missing_required_column(tmp_path: Path) -> None:
    sample_map = _sample_map(2).drop(columns=["source_timestamp_ns"])
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0), (0, 0, 0)],
        sample_map=sample_map,
    )

    with pytest.raises(SampleMapValidationError, match="source_timestamp_ns"):
        PreparedSegmentReader(segment_dir)


@pytest.mark.parametrize(
    ("column", "values", "message"),
    [
        ("output_frame_index", [0, 2], "从 0 连续递增"),
        ("output_timestamp_ns", [0, 0], "严格递增"),
        ("source_frame_index", [10, -1], "不能包含负值"),
    ],
)
def test_reader_rejects_invalid_sample_map_values(
    tmp_path: Path,
    column: str,
    values: list[int],
    message: str,
) -> None:
    sample_map = _sample_map(2)
    sample_map[column] = pd.Series(values, dtype="int64")
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0), (0, 0, 0)],
        sample_map=sample_map,
    )

    with pytest.raises(SampleMapValidationError, match=message):
        PreparedSegmentReader(segment_dir)


def test_reader_rejects_non_integer_timestamp_dtype(tmp_path: Path) -> None:
    sample_map = _sample_map(2)
    sample_map["output_timestamp_ns"] = pd.Series([0.0, 1.5], dtype="float64")
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0), (0, 0, 0)],
        sample_map=sample_map,
    )

    with pytest.raises(SampleMapValidationError, match="整数 dtype"):
        PreparedSegmentReader(segment_dir)


def test_reader_detects_video_shorter_than_sample_map(tmp_path: Path) -> None:
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0)],
        sample_map=_sample_map(2),
    )

    with pytest.raises(VideoDecodeError, match="帧数少于"):
        list(PreparedSegmentReader(segment_dir))


def test_reader_detects_video_longer_than_sample_map(tmp_path: Path) -> None:
    segment_dir = _build_segment(
        tmp_path,
        frame_colors=[(0, 0, 0), (0, 0, 0)],
        sample_map=_sample_map(1),
    )

    with pytest.raises(VideoDecodeError, match="帧数多于"):
        list(PreparedSegmentReader(segment_dir))


@pytest.mark.parametrize(
    ("field", "uri"),
    [
        ("uri", "../outside.mp4"),
        ("sample_map_uri", "../outside.parquet"),
    ],
)
def test_reader_rejects_paths_outside_segment(
    tmp_path: Path,
    field: str,
    uri: str,
) -> None:
    segment_dir = _build_segment(tmp_path)
    segment = _read_segment_json(segment_dir)
    if field == "uri":
        segment["streams"][0]["uri"] = uri
    else:
        segment["streams"][0]["origin"]["sample_map_uri"] = uri
    _write_segment_json(segment_dir, segment)

    with pytest.raises(PreparedSegmentError, match="Segment 目录之外"):
        PreparedSegmentReader(segment_dir)
