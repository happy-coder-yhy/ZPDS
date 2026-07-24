import json
import pickle
import sqlite3
from pathlib import Path

import cv2
import numpy as np
import pytest

from zpds.adapters.hdf5 import Hdf5Inspector, Hdf5Reader
from zpds.adapters.log import LogParser
from zpds.adapters.mcap import McapInspector, McapReader
from zpds.adapters.pickle import inspect_pickle, summarize_primitive_pickle
from zpds.adapters.profiled_mcap import ProfiledMcapAdapter
from zpds.adapters.rosbag import Ros2Db3Adapter
from zpds.adapters.video import VideoInspector


def test_video_inspect_and_full_decode(tmp_path: Path) -> None:
    path = tmp_path / "tiny.avi"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (8, 6),
    )
    assert writer.isOpened()
    for value in (0, 64, 128):
        writer.write(np.full((6, 8, 3), value, dtype=np.uint8))
    writer.release()

    inventory = VideoInspector().inspect(str(path))
    report = VideoInspector().scan(str(path))

    assert inventory.total_frames == 3
    assert inventory.streams[0].width == 8
    assert report.passed
    assert report.decoded_records == 3


def test_hdf5_inspect_and_chunked_read(tmp_path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path = tmp_path / "tiny.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("robot/joints", data=np.arange(15).reshape(5, 3))

    inventory = Hdf5Inspector().inspect(str(path))
    chunks = list(Hdf5Reader(str(path)).iter_dataset("robot/joints", chunk_rows=2))

    assert inventory.metadata["dataset_count"] == 1
    assert [chunk.shape[0] for chunk in chunks] == [2, 2, 1]
    report = Hdf5Inspector().scan(str(path))
    assert report.decoded_records == 5
    assert report.metadata["full_dataset_scan"] is True


def test_mcap_reads_decodes_and_keeps_two_clocks(tmp_path: Path) -> None:
    pytest.importorskip("mcap_protobuf")
    from google.protobuf.wrappers_pb2 import StringValue
    from mcap_protobuf.writer import Writer

    path = tmp_path / "tiny.mcap"
    writer = Writer(str(path))
    writer.write_message(
        "/sensor/imu",
        StringValue(value="first"),
        log_time=100,
        publish_time=90,
    )
    writer.write_message(
        "/sensor/imu",
        StringValue(value="second"),
        log_time=200,
        publish_time=190,
    )
    writer.finish()

    inventory = McapInspector().inspect(str(path))
    raw = list(McapReader(str(path)).iter_messages())
    decoded = list(McapReader(str(path)).iter_decoded())
    report = McapInspector().analyze_time(str(path))

    assert [clock.clock_id for clock in inventory.clocks] == [
        "mcap_log_time",
        "mcap_publish_time",
    ]
    assert len(raw) == len(decoded) == 2
    assert report.passed
    assert report.metadata["regressions"] == {"log_time": 0, "publish_time": 0}


def test_mcap_scan_decodes_embedded_png_payload(tmp_path: Path) -> None:
    pytest.importorskip("mcap_protobuf")
    pytest.importorskip("foxglove_schemas_protobuf")
    from foxglove_schemas_protobuf.CompressedImage_pb2 import CompressedImage
    from mcap_protobuf.writer import Writer

    path = tmp_path / "depth.mcap"
    ok, encoded = cv2.imencode(
        ".png",
        np.arange(48, dtype=np.uint16).reshape(6, 8),
    )
    assert ok
    writer = Writer(str(path))
    writer.write_message(
        "/camera/depth/compressed",
        CompressedImage(format="png", data=encoded.tobytes()),
        log_time=100,
        publish_time=90,
    )
    writer.finish()

    report = McapInspector().scan_embedded_media(str(path))

    assert report.passed
    assert report.checked_records == 1
    assert report.decoded_records == 1
    assert report.metadata["decoded_images"] == 1


def test_dunjia_profile_requires_real_topic_layout(tmp_path: Path) -> None:
    pytest.importorskip("mcap_protobuf")
    from google.protobuf.wrappers_pb2 import StringValue
    from mcap_protobuf.writer import Writer

    path = tmp_path / "dunjia.mcap"
    writer = Writer(str(path))
    for index, topic in enumerate(
        (
            "/robot0/sensor/imu",
            "/robot0/sensor/depth/compressed",
            "/robot0/sensor/camera0/compressed",
        )
    ):
        writer.write_message(topic, StringValue(value=topic), log_time=index + 1)
    writer.finish()

    report = ProfiledMcapAdapter("dunjia_ego").validate(str(path))

    assert report.passed
    assert report.metadata["profile"] == "dunjia_ego"


def test_ros2_db3_is_read_only_and_streamed(tmp_path: Path) -> None:
    path = tmp_path / "rosbag.db3"
    with sqlite3.connect(path) as connection:
        connection.executescript(
            "CREATE TABLE topics("
            "id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT);"
            "CREATE TABLE messages("
            "id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);"
            "INSERT INTO topics VALUES (1, '/imu', 'sensor_msgs/msg/Imu', 'cdr');"
            "INSERT INTO messages VALUES (1, 1, 100, x'0102');"
        )

    adapter = Ros2Db3Adapter()
    inventory = adapter.inspect(str(path))
    messages = list(adapter.iter_messages(str(path)))
    scan = adapter.scan(str(path))

    assert inventory.metadata["message_count"] == 1
    assert messages[0].stream_id == "/imu"
    assert messages[0].payload == b"\x01\x02"
    assert scan.checked_records == 1
    assert scan.decoded_records == 0
    assert scan.metadata["cdr_decoded"] is False


def test_log_summary_is_bounded(tmp_path: Path) -> None:
    path = tmp_path / "device.log"
    path.write_text(
        "".join(
            json.dumps({"timestamp_ns": index, "level": "INFO"}) + "\n"
            for index in range(120)
        ),
        encoding="utf-8",
    )

    summary = LogParser().parse(str(path))

    assert summary["event_count"] == 120
    assert len(summary["events"]) == 100
    assert summary["events_truncated"] is True


def test_pickle_is_statically_inspected_then_summarized_in_child(tmp_path: Path) -> None:
    path = tmp_path / "annotations.pkl"
    path.write_bytes(pickle.dumps({"frames": [1, 2, 3]}, protocol=4))

    inspection = inspect_pickle(path)
    summary = summarize_primitive_pickle(path)

    assert inspection.opcode_count > 0
    assert summary == {"keys": ["'frames'"], "length": 1, "type": "dict"}
