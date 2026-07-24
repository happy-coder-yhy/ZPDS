import json
from pathlib import Path

from zpds.adapters.guida import GuidaAdapter
from zpds.core.types import StreamKind


def make_guida_session(root: Path, *, regress: bool = False) -> Path:
    session = root / "guida-session"
    (session / "imu").mkdir(parents=True)
    (session / "meta.json").write_text(
        json.dumps(
            {
                "session": {"output_folder_name": "guida_demo"},
                "streams": {
                    "color": {
                        "enabled": True,
                        "width": 4,
                        "height": 3,
                        "fps": 30,
                        "intrinsics": {
                            "fx": 2.0,
                            "fy": 2.0,
                            "cx": 1.5,
                            "cy": 1.0,
                            "width": 4,
                            "height": 3,
                        },
                    },
                    "depth": {"enabled": True, "width": 4, "height": 3, "fps": 30},
                },
                "imu": {
                    "csv": "imu/imu.csv",
                    "sample_rate_hz": 100,
                    "accel_unit": "m/s^2",
                    "gyro_unit": "rad/s",
                },
            }
        ),
        encoding="utf-8",
    )
    timestamps = [1_000_000_000, 1_033_333_333, 900_000_000 if regress else 1_100_000_000]
    records = [
        {"schema": "guida.index.v1"},
        {
            "type": "segment_start",
            "color_video": "color_000000.mkv",
            "depth_video": "depth_000000.mkv",
        },
        *[
            {"type": "frame", "seq": index, "timestamp_ns": timestamp}
            for index, timestamp in enumerate(timestamps)
        ],
    ]
    (session / "index.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    (session / "color_000000.mkv").write_bytes(b"fixture")
    (session / "depth_000000.mkv").write_bytes(b"fixture")
    (session / "imu" / "imu_000000.csv").write_text(
        "timestamp_ns,ax,ay,az,gx,gy,gz\n"
        "1000000000,0,0,9.8,0,0,0\n"
        "1010000000,0.1,0,9.8,0,0.1,0\n",
        encoding="utf-8",
    )
    return session


def test_guida_inspect_builds_authoritative_catalog(tmp_path: Path) -> None:
    session = make_guida_session(tmp_path)

    inventory = GuidaAdapter().inspect(str(session))

    assert inventory.session_id == "guida_demo"
    assert inventory.total_frames == 3
    assert inventory.metadata["index_authoritative"] is True
    assert [clock.clock_id for clock in inventory.clocks] == [
        "guida_index_timestamp",
        "guida_imu_timestamp",
        "video_container_time",
    ]
    assert {stream.kind for stream in inventory.streams} == {
        StreamKind.COLOR,
        StreamKind.DEPTH,
        StreamKind.IMU,
    }
    assert inventory.calibrations[0].calibration_id == "color_intrinsics"


def test_guida_declared_imu_mismatch_is_explainable_warning(tmp_path: Path) -> None:
    session = make_guida_session(tmp_path)

    report = GuidaAdapter().validate(str(session))

    assert report.passed
    assert [issue.code for issue in report.issues] == ["imu_declared_path_missing"]
    assert report.issues[0].level.value == "warn"


def test_guida_time_regression_is_blocking(tmp_path: Path) -> None:
    session = make_guida_session(tmp_path, regress=True)

    report = GuidaAdapter().analyze_time(str(session))

    assert not report.passed
    assert {issue.code for issue in report.issues} == {"timestamp_regression"}
