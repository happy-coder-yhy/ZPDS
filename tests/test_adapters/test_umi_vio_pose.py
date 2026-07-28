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
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
    header_topic: str,
):
    channel = SimpleNamespace(topic=topic)
    message = SimpleNamespace(
        log_time=timestamp_ns + 1,
        publish_time=timestamp_ns + 2,
    )
    decoded = SimpleNamespace(
        header=SimpleNamespace(
            timestamp=timestamp_ns,
            topic_name=header_topic,
        ),
        frame_id="world",
        pose=SimpleNamespace(
            position=SimpleNamespace(
                x=position[0],
                y=position[1],
                z=position[2],
            ),
            orientation=SimpleNamespace(
                x=orientation[0],
                y=orientation[1],
                z=orientation[2],
                w=orientation[3],
            ),
        ),
    )
    return None, channel, message, decoded


def test_read_session_builds_dual_raw_vio_pose_streams(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "01767.mcap"
    source.write_bytes(b"fixture")
    base = 1_767_657_922_190_087_268
    interval = 33_333_333
    messages = []
    for robot_id in ("robot0", "robot1"):
        topic = getattr(
            umi_reader,
            f"TOPIC_{robot_id.upper()}_VIO_POSE",
        )
        header_topic = umi_reader.TOPIC_ROBOT0_VIO_POSE
        messages.extend(
            [
                _message(
                    topic,
                    timestamp_ns=base,
                    position=(0.1, 0.2, 0.3),
                    orientation=(0.0, 0.0, 0.0, 1.0),
                    header_topic=header_topic,
                ),
                _message(
                    topic,
                    timestamp_ns=base + interval,
                    position=(0.11, 0.21, 0.31),
                    orientation=(0.0, 0.0, 0.01, 0.99995),
                    header_topic=header_topic,
                ),
            ]
        )

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
        "robot0_vio_pose",
        "robot1_vio_pose",
    }
    robot1 = session.time_series_streams["robot1_vio_pose"]
    assert robot1.modality == "vio_pose"
    assert robot1.role == "state"
    assert robot1.timestamps_ns == [base, base + interval]
    assert isinstance(robot1.rows, pd.DataFrame)
    assert robot1.rows["log_time_ns"].dtype == "int64"
    assert robot1.rows["tx"].dtype == "float64"
    assert robot1.rows["qw"].dtype == "float64"
    assert robot1.rows["source_frame_id"].tolist() == ["world", "world"]
    assert robot1.rows["source_header_topic"].tolist() == [
        umi_reader.TOPIC_ROBOT0_VIO_POSE,
        umi_reader.TOPIC_ROBOT0_VIO_POSE,
    ]
    assert robot1.metadata["source_topic"] == (
        umi_reader.TOPIC_ROBOT1_VIO_POSE
    )
    assert robot1.metadata["source_topic_authority"] == "mcap_channel"
    assert robot1.metadata["translation_unit"] == "unknown"
    assert robot1.metadata["child_frame"] == "unknown"
    assert robot1.metadata["transform_direction"] == "unknown"
