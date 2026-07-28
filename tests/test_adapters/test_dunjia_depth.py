from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from zpds_prepare.readers import dunjia_reader


def test_required_dunjia_depth_cannot_be_disabled(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="不能禁用 include_depth"):
        dunjia_reader.read_session(
            str(tmp_path / "missing.mcap"),
            include_depth=False,
            require_depth=True,
        )


class _FakeReader:
    def __init__(self, messages: list[tuple]) -> None:
        self._messages = messages

    def iter_decoded_messages(self):
        yield from self._messages


def _timestamp(seconds: int, nanos: int = 0):
    return SimpleNamespace(seconds=seconds, nanos=nanos)


def _message(topic: str, decoded, *, log_time: int):
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(log_time=log_time, publish_time=log_time + 7)
    return None, channel, message, decoded


def test_read_session_builds_formal_dunjia_depth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "session.mcap"
    source.write_bytes(b"fixture")
    depth_image = np.arange(12, dtype=np.uint16).reshape(3, 4)
    encoded_ok, encoded_depth = cv2.imencode(".png", depth_image)
    assert encoded_ok

    messages = [
        _message(
            dunjia_reader.TOPIC_CAMERA0_CALIB,
            SimpleNamespace(width=4, height=3),
            log_time=1_000_000_000,
        ),
        _message(
            dunjia_reader.TOPIC_CAMERA0,
            SimpleNamespace(timestamp=_timestamp(1), data=b"h264-a"),
            log_time=1_000_000_000,
        ),
        _message(
            dunjia_reader.TOPIC_CAMERA0,
            SimpleNamespace(timestamp=_timestamp(1, 40_000_000), data=b"h264-b"),
            log_time=1_040_000_000,
        ),
        _message(
            dunjia_reader.TOPIC_DEPTH,
            SimpleNamespace(
                timestamp=_timestamp(1),
                frame_id="headcam_depth_optical_frame",
                format="png",
                data=encoded_depth.tobytes(),
            ),
            log_time=1_000_000_000,
        ),
        _message(
            dunjia_reader.TOPIC_DEPTH,
            SimpleNamespace(
                timestamp=_timestamp(1, 40_000_000),
                frame_id="headcam_depth_optical_frame",
                format="png",
                data=encoded_depth.tobytes(),
            ),
            log_time=1_040_000_000,
        ),
    ]
    monkeypatch.setattr(
        dunjia_reader,
        "_open_mcap",
        lambda _path: (_FakeReader(messages), io.BytesIO()),
    )
    cache_dir = tmp_path / "cache"
    observed_cache_dirs: list[Path] = []

    def fake_get_video(_dataset, _topic, cache_dir=None):
        observed_cache_dirs.append(Path(cache_dir))
        return str(tmp_path / "camera0.mp4")

    monkeypatch.setattr(dunjia_reader, "get_video_for_topic", fake_get_video)

    session = dunjia_reader.read_session(
        str(source),
        cache_dir=cache_dir,
        require_depth=True,
    )

    assert observed_cache_dirs == [cache_dir]
    depth = session.depth_streams["ego_depth"]
    assert depth.source_kind == "mcap_compressed_image"
    assert depth.frame_count == 2
    assert depth.timestamps_ns == [1_000_000_000, 1_040_000_000]
    assert depth.index_frames[0]["log_time_ns"] == 1_000_000_000
    assert depth.index_frames[0]["publish_time_ns"] == 1_000_000_007
    assert depth.dtype == "uint16"
    assert (depth.width, depth.height) == (4, 3)
    assert depth.fps == 25.0
    assert depth.unit == "unknown"
    assert depth.invalid_value is None
    assert depth.frame_id == "headcam_depth_optical_frame"
    assert depth.metadata["topic"] == dunjia_reader.TOPIC_DEPTH
    assert depth.metadata["source_asset_id"] == "raw_mcap"


def test_dunjia_video_cache_is_outside_raw_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "session.mcap"
    source.write_bytes(b"fixture")
    cache_dir = tmp_path / "cache"
    captured_output: list[Path] = []

    def fake_reconstruct(_dataset, _topic, output_path):
        captured_output.append(Path(output_path))
        return output_path

    monkeypatch.setattr(dunjia_reader, "reconstruct_video", fake_reconstruct)

    result = dunjia_reader.get_video_for_topic(
        str(source),
        dunjia_reader.TOPIC_CAMERA0,
        cache_dir=cache_dir,
    )

    assert Path(result) == cache_dir.resolve() / "cam0.mp4"
    assert captured_output == [cache_dir.resolve() / "cam0.mp4"]
    assert not Path(result).is_relative_to(raw_dir.resolve())

    with pytest.raises(ValueError, match="不能位于原始 MCAP 目录内"):
        dunjia_reader.get_video_for_topic(
            str(source),
            dunjia_reader.TOPIC_CAMERA0,
            cache_dir=raw_dir / "cache",
        )
