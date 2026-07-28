from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from zpds_prepare.readers import umi_reader


class _FakeReader:
    def __init__(self, messages: list[tuple]) -> None:
        self._messages = messages

    def iter_decoded_messages(self):
        yield from self._messages


def _message(
    topic: str,
    *,
    timestamp_ns: int,
    log_time_ns: int,
    raw_value: float,
):
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(
        log_time=log_time_ns,
        publish_time=log_time_ns + 7,
    )
    decoded = SimpleNamespace(
        header=SimpleNamespace(timestamp=timestamp_ns),
        frame_id="",
        value=raw_value,
    )
    return None, channel, message, decoded


def test_read_session_builds_dual_raw_magnetic_encoder_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "01767.mcap"
    source.write_bytes(b"fixture")
    base = 1_767_657_922_053_519_000
    messages = [
        _message(
            umi_reader.TOPIC_ROBOT0_MAGNETIC_ENCODER,
            timestamp_ns=base,
            log_time_ns=base + 1,
            raw_value=0.10300199687480927,
        ),
        _message(
            umi_reader.TOPIC_ROBOT1_MAGNETIC_ENCODER,
            timestamp_ns=base + 2,
            log_time_ns=base + 3,
            raw_value=0.20300199687480927,
        ),
        _message(
            umi_reader.TOPIC_ROBOT0_MAGNETIC_ENCODER,
            timestamp_ns=base + 4_000_000,
            log_time_ns=base + 4_000_001,
            raw_value=0.10400199687480927,
        ),
        _message(
            umi_reader.TOPIC_ROBOT1_MAGNETIC_ENCODER,
            timestamp_ns=base + 4_000_002,
            log_time_ns=base + 4_000_003,
            raw_value=0.20400199687480927,
        ),
    ]
    monkeypatch.setattr(
        umi_reader,
        "_open_mcap",
        lambda _path: (_FakeReader(messages), io.BytesIO()),
    )

    session = umi_reader.read_session(
        str(source),
        cache_dir=tmp_path / "cache",
    )

    assert set(session.time_series_streams) == {
        "robot0_magnetic_encoder",
        "robot1_magnetic_encoder",
    }
    robot0 = session.time_series_streams["robot0_magnetic_encoder"]
    assert robot0.modality == "magnetic_encoder"
    assert robot0.role == "sensor"
    assert robot0.timestamps_ns == [base, base + 4_000_000]
    assert isinstance(robot0.rows, pd.DataFrame)
    assert robot0.rows["log_time_ns"].dtype == "int64"
    assert robot0.rows["publish_time_ns"].dtype == "int64"
    assert robot0.rows["raw_value"].dtype == "float64"
    assert robot0.rows["log_time_ns"].tolist() == [
        base + 1,
        base + 4_000_001,
    ]
    assert robot0.metadata["source_topic"] == (
        "/robot0/sensor/magnetic_encoder"
    )
    assert robot0.metadata["source_field"] == "value"
    assert robot0.metadata["unit"] == "unknown"
    assert robot0.metadata["semantic_status"] == "raw_unverified"


def test_umi_video_cache_is_outside_raw_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    source = raw_dir / "01767.mcap"
    source.write_bytes(b"fixture")
    cache_dir = tmp_path / "cache"
    captured_output: list[Path] = []

    def fake_reconstruct(_dataset, _topic, output_path):
        captured_output.append(Path(output_path))
        return output_path

    monkeypatch.setattr(umi_reader, "reconstruct_video", fake_reconstruct)

    result = umi_reader.get_video_for_topic(
        str(source),
        umi_reader.TOPIC_ROBOT0_CAMERA,
        cache_dir=cache_dir,
    )

    assert Path(result) == cache_dir.resolve() / "robot0_camera0.mp4"
    assert captured_output == [
        cache_dir.resolve() / "robot0_camera0.mp4"
    ]
    assert not Path(result).is_relative_to(raw_dir.resolve())

    with pytest.raises(ValueError, match="cannot be inside"):
        umi_reader.get_video_for_topic(
            str(source),
            umi_reader.TOPIC_ROBOT0_CAMERA,
            cache_dir=raw_dir / "cache",
        )
