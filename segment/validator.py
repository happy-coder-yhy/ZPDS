"""
写出后验证：确认所有文件可读、数据一致。
"""

import hashlib
import json
import time
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from segment.annotation_validator import validate_hand_object_stream
from segment.mask_validator import validate_mask_stream
from segment.vio_pose_validator import validate_vio_pose_streams


def _probe_video(path: Path, attempts: int = 10) -> tuple[bool, int]:
    """等待 Windows/OpenCV 完成 MP4 尾部索引刷新后探测视频。"""
    frame_count = -1
    for attempt in range(attempts):
        cap = cv2.VideoCapture(str(path))
        opened = cap.isOpened()
        if opened:
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            readable, frame = cap.read()
        else:
            readable, frame = False, None
        cap.release()

        if opened and readable and frame is not None and frame_count >= 0:
            return True, frame_count
        if attempt + 1 < attempts:
            time.sleep(0.2)
    return False, frame_count


def sha256_hex(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


@lru_cache(maxsize=64)
def _cached_sha256(path: str, size: int, mtime_ns: int) -> str:
    """按路径和文件状态缓存 Raw 哈希，避免多 Segment 重复读取大文件。"""
    del size, mtime_ns
    return sha256_hex(path)


def validate_source_hashes(segment: dict) -> dict:
    """在 Raw 可访问时核对 source_assets 中的普通 SHA256。"""
    errors: list[str] = []
    session_uri = segment.get("source_session", {}).get("session_uri")
    if not session_uri:
        return {
            "checks": {"source_hashes": "skip"},
            "statistics": {"source_hashes_verified": 0},
            "errors": [],
        }

    session_path = Path(session_uri)
    source_root = session_path if session_path.is_dir() else session_path.parent
    verified = 0
    for asset in segment.get("source_assets", []):
        expected = asset.get("sha256")
        if not expected or asset.get("hash_kind", "sha256") != "sha256":
            continue
        if (
            asset.get("source_asset_id") == "raw_mcap"
            and session_path.is_file()
        ):
            source_path = session_path
        else:
            source_path = source_root / str(asset.get("uri", ""))
        if not source_path.is_file():
            continue
        stat = source_path.stat()
        actual = _cached_sha256(
            str(source_path.resolve()),
            stat.st_size,
            stat.st_mtime_ns,
        )
        verified += 1
        if actual.lower() != str(expected).lower():
            errors.append(
                f"Source hash mismatch: {asset.get('source_asset_id')}"
            )

    return {
        "checks": {
            "source_hashes": (
                "fail" if errors else "pass" if verified else "skip"
            )
        },
        "statistics": {"source_hashes_verified": verified},
        "errors": errors,
    }


def validate_depth_streams(seg_dir: Path, segment: dict) -> dict:
    """验证无损 PNG 深度流、sample map、dtype、单位和来源引用。"""
    errors: list[str] = []
    checks: dict[str, str] = {}
    stats: dict[str, object] = {}
    depth_streams = [
        stream
        for stream in segment.get("streams", [])
        if stream.get("modality") == "depth"
    ]
    if not depth_streams:
        return {
            "status": "skip",
            "checks": {"depth_streams_valid": "skip"},
            "statistics": {},
            "errors": [],
        }

    source_asset_ids = {
        asset.get("source_asset_id")
        for asset in segment.get("source_assets", [])
    }
    any_warning = False
    calibration: dict = {}
    calibration_path = seg_dir / segment.get("calibration_uri", "")
    if calibration_path.is_file():
        try:
            calibration = json.loads(
                calibration_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError, json.JSONDecodeError):
            calibration = {}

    for stream in depth_streams:
        stream_id = stream.get("stream_id", "unknown_depth")
        origin = stream.get("origin", {})
        is_mcap_embedded = (
            origin.get("source_asset_id") == "raw_mcap"
            and origin.get("operation") == "trim_decode_embedded_png"
        )
        if stream.get("format") != "png_sequence":
            checks[f"depth_{stream_id}_png_sequence"] = "skip"
            continue

        depth_dir = seg_dir / stream.get("uri", "")
        if not depth_dir.is_dir():
            errors.append(f"[{stream_id}] Missing depth directory: {depth_dir}")
            checks[f"depth_{stream_id}_files"] = "fail"
            continue

        sample_map_uri = stream.get("origin", {}).get("sample_map_uri", "")
        sample_map_path = seg_dir / sample_map_uri
        if not sample_map_uri or not sample_map_path.is_file():
            errors.append(f"[{stream_id}] Missing depth sample_map: {sample_map_path}")
            checks[f"depth_{stream_id}_sample_map"] = "fail"
            continue

        try:
            sample_map = pd.read_parquet(str(sample_map_path))
        except (OSError, ValueError, ImportError, KeyError) as exc:
            errors.append(f"[{stream_id}] Depth sample_map unreadable: {exc}")
            checks[f"depth_{stream_id}_sample_map"] = "fail"
            continue

        required_columns = {
            "output_frame_index",
            "output_timestamp_ns",
            "output_file",
            "source_frame_index",
            "source_timestamp_ns",
            "source_file",
            "mapping_method",
            "time_error_ns",
        }
        missing_columns = sorted(required_columns - set(sample_map.columns))
        if missing_columns:
            errors.append(
                f"[{stream_id}] Depth sample_map missing columns: {missing_columns}"
            )
            checks[f"depth_{stream_id}_sample_map"] = "fail"
            continue

        if is_mcap_embedded:
            clock_columns = {
                "source_timestamp_ns",
                "source_log_time_ns",
                "source_publish_time_ns",
            }
            missing_clock_columns = sorted(
                clock_columns - set(sample_map.columns)
            )
            if missing_clock_columns:
                errors.append(
                    f"[{stream_id}] Dunjia depth sample_map missing clocks: "
                    f"{missing_clock_columns}"
                )
                checks[f"depth_{stream_id}_dual_clock"] = "fail"
            else:
                clock_ok = True
                for column in sorted(clock_columns):
                    values = sample_map[column].to_numpy(dtype=np.int64)
                    if len(values) > 1 and not np.all(np.diff(values) > 0):
                        clock_ok = False
                        errors.append(
                            f"[{stream_id}] {column} is not monotonic"
                        )
                source_clock = sample_map[
                    "source_timestamp_ns"
                ].to_numpy(dtype=np.int64)
                log_clock = sample_map[
                    "source_log_time_ns"
                ].to_numpy(dtype=np.int64)
                stats[f"depth_max_message_log_delta_ns_{stream_id}"] = int(
                    np.max(np.abs(source_clock - log_clock))
                )
                checks[f"depth_{stream_id}_dual_clock"] = (
                    "pass" if clock_ok else "fail"
                )

        expected_indices = list(range(len(sample_map)))
        actual_indices = sample_map["output_frame_index"].tolist()
        if actual_indices != expected_indices:
            errors.append(f"[{stream_id}] Depth output_frame_index is not contiguous")

        output_timestamps = sample_map["output_timestamp_ns"].to_numpy(dtype=np.int64)
        if len(output_timestamps) > 1 and not np.all(np.diff(output_timestamps) > 0):
            errors.append(f"[{stream_id}] Depth output timestamps are not monotonic")

        source_timestamps = sample_map["source_timestamp_ns"].to_numpy(dtype=np.int64)
        if len(source_timestamps) > 1 and not np.all(np.diff(source_timestamps) > 0):
            errors.append(f"[{stream_id}] Depth source timestamps are not monotonic")

        if not (sample_map["mapping_method"] == "identity").all():
            errors.append(f"[{stream_id}] Depth mapping must be identity")
        if not (sample_map["time_error_ns"] == 0).all():
            errors.append(f"[{stream_id}] Identity depth mapping has non-zero time error")

        timeline_end_ns = int(segment.get("timeline", {}).get("end_ns", 0))
        if len(output_timestamps):
            median_interval_ns = (
                int(np.median(np.diff(output_timestamps)))
                if len(output_timestamps) > 1
                else 0
            )
            boundary_tolerance_ns = max(
                80_000_000,
                median_interval_ns * 2,
            )
            boundary_delta_ns = timeline_end_ns - int(output_timestamps[-1])
            stats[f"depth_boundary_delta_ns_{stream_id}"] = (
                boundary_delta_ns
            )
            boundary_ok = (
                int(output_timestamps[0]) >= 0
                and boundary_delta_ns >= 0
                and boundary_delta_ns <= boundary_tolerance_ns
            )
            if not boundary_ok:
                errors.append(
                    f"[{stream_id}] Depth does not cover Segment boundary: "
                    f"last={int(output_timestamps[-1])}, "
                    f"timeline_end={timeline_end_ns}, "
                    f"tolerance={boundary_tolerance_ns}"
                )
            checks[f"depth_{stream_id}_boundary"] = (
                "pass" if boundary_ok else "fail"
            )

        output_files = [depth_dir / str(name) for name in sample_map["output_file"]]
        missing_files = [str(path) for path in output_files if not path.is_file()]
        if missing_files:
            errors.append(
                f"[{stream_id}] Missing {len(missing_files)} depth output files; "
                f"first={missing_files[0]}"
            )
            checks[f"depth_{stream_id}_files"] = "fail"
            continue

        actual_pngs = sorted(depth_dir.glob("*.png"))
        stats[f"depth_frames_{stream_id}"] = len(actual_pngs)
        stats[f"depth_sample_map_rows_{stream_id}"] = len(sample_map)
        if len(actual_pngs) != len(sample_map):
            errors.append(
                f"[{stream_id}] Depth PNG count ({len(actual_pngs)}) != "
                f"sample_map rows ({len(sample_map)})"
            )

        dtypes: set[str] = set()
        resolutions: set[tuple[int, int]] = set()
        zero_pixels = 0
        total_pixels = 0
        sample_count = min(20, len(output_files))
        step = max(1, len(output_files) // sample_count) if sample_count else 1
        for path in output_files[::step][:sample_count]:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                errors.append(f"[{stream_id}] Depth PNG unreadable: {path}")
                continue
            if image.ndim != 2:
                errors.append(
                    f"[{stream_id}] Depth PNG must be single-channel: "
                    f"{path}, shape={image.shape}"
                )
                continue
            dtypes.add(str(image.dtype))
            resolutions.add((int(image.shape[1]), int(image.shape[0])))
            zero_pixels += int(np.count_nonzero(image == 0))
            total_pixels += int(image.size)

        declared_dtype = stream.get("dtype", "unknown")
        stats[f"depth_dtype_{stream_id}"] = sorted(dtypes)
        if len(dtypes) != 1 or (
            declared_dtype != "unknown" and declared_dtype not in dtypes
        ):
            errors.append(
                f"[{stream_id}] Depth dtype mismatch: "
                f"declared={declared_dtype}, actual={sorted(dtypes)}"
            )

        declared_shape = stream.get("shape", [])
        stats[f"depth_resolution_{stream_id}"] = sorted(resolutions)
        if len(resolutions) != 1:
            errors.append(
                f"[{stream_id}] Depth resolution inconsistent: {sorted(resolutions)}"
            )
        elif len(declared_shape) == 2:
            actual_width, actual_height = next(iter(resolutions))
            if declared_shape != [actual_height, actual_width]:
                errors.append(
                    f"[{stream_id}] Depth shape mismatch: "
                    f"declared={declared_shape}, actual={[actual_height, actual_width]}"
                )

        if total_pixels:
            stats[f"depth_zero_ratio_{stream_id}"] = zero_pixels / total_pixels

        unit = stream.get("unit")
        if not unit:
            errors.append(f"[{stream_id}] Depth unit field is missing")
        elif unit == "unknown":
            any_warning = True
            checks[f"depth_{stream_id}_unit"] = "warn"
        else:
            checks[f"depth_{stream_id}_unit"] = "pass"

        source_asset_id = origin.get("source_asset_id")
        if source_asset_id not in source_asset_ids:
            errors.append(
                f"[{stream_id}] Unknown source_asset_id: {source_asset_id}"
            )

        if is_mcap_embedded:
            depth_calibration = next(
                (
                    camera
                    for camera in calibration.get("cameras", [])
                    if camera.get("stream_id") == stream_id
                ),
                None,
            )
            calibration_ok = depth_calibration is not None
            if depth_calibration is not None:
                declared_shape = stream.get("shape", [])
                if len(declared_shape) != 2:
                    calibration_ok = False
                else:
                    expected_resolution = [
                        int(declared_shape[1]),
                        int(declared_shape[0]),
                    ]
                    calibration_ok = (
                        depth_calibration.get("resolution")
                        == expected_resolution
                        and depth_calibration.get("frame_id")
                        == stream.get("frame_id")
                    )
            if not calibration_ok:
                errors.append(
                    f"[{stream_id}] Missing or inconsistent depth calibration"
                )
            checks[f"depth_{stream_id}_calibration"] = (
                "pass" if calibration_ok else "fail"
            )

        stream_errors = [
            error for error in errors if error.startswith(f"[{stream_id}]")
        ]
        checks[f"depth_{stream_id}_files"] = "fail" if stream_errors else "pass"
        checks[f"depth_{stream_id}_sample_map"] = (
            "fail" if stream_errors else "pass"
        )

    status = "fail" if errors else "warn" if any_warning else "pass"
    checks["depth_streams_valid"] = status
    return {
        "status": status,
        "checks": checks,
        "statistics": stats,
        "errors": errors,
    }


def _read_source_magnetic_encoders(
    segment: dict,
    topics: set[str],
) -> tuple[dict[str, list[tuple[int, int, int, float]]] | None, str | None]:
    """Read encoder source rows once when the original MCAP is accessible."""
    session_uri = segment.get("source_session", {}).get("session_uri")
    if not session_uri:
        return None, None
    source_path = Path(session_uri)
    if not source_path.is_file() or source_path.suffix.lower() != ".mcap":
        return None, None

    try:
        from mcap.reader import make_reader
        from mcap_protobuf.decoder import DecoderFactory

        source_start_ns = int(segment["source_span"]["start_ns"])
        source_end_ns = int(segment["source_span"]["end_ns"])
        rows: dict[str, list[tuple[int, int, int, float]]] = {
            topic: []
            for topic in topics
        }
        with source_path.open("rb") as source_file:
            reader = make_reader(
                source_file,
                decoder_factories=[DecoderFactory()],
            )
            for _schema, channel, message, decoded in (
                reader.iter_decoded_messages(topics=list(topics))
            ):
                timestamp_ns = int(decoded.header.timestamp)
                if source_start_ns <= timestamp_ns <= source_end_ns:
                    rows[channel.topic].append(
                        (
                            timestamp_ns,
                            int(message.log_time),
                            int(message.publish_time),
                            float(decoded.value),
                        )
                    )
        return rows, None
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        return None, str(exc)


def validate_magnetic_encoder_streams(
    seg_dir: Path,
    segment: dict,
) -> dict:
    """Validate dual UMI encoder streams and their Raw MCAP lineage."""
    encoder_streams = [
        stream
        for stream in segment.get("streams", [])
        if stream.get("modality") == "magnetic_encoder"
    ]
    if not encoder_streams:
        return {
            "status": "skip",
            "checks": {"magnetic_encoder_streams": "skip"},
            "statistics": {},
            "errors": [],
        }

    checks: dict[str, str] = {}
    stats: dict[str, object] = {}
    errors: list[str] = []
    source_asset_ids = {
        asset.get("source_asset_id")
        for asset in segment.get("source_assets", [])
    }
    source_start_ns = int(segment["source_span"]["start_ns"])
    timeline_end_ns = int(segment["timeline"]["end_ns"])
    required_columns = {
        "timestamp_ns",
        "source_timestamp_ns",
        "log_time_ns",
        "publish_time_ns",
        "raw_value",
        "robot_id",
        "source_topic",
        "unit",
        "semantic_status",
    }
    frames: dict[str, pd.DataFrame] = {}
    topic_to_stream: dict[str, str] = {}
    observed_robot_ids: set[str] = set()

    for stream in encoder_streams:
        stream_id = str(stream.get("stream_id", "unknown_encoder"))
        stream_errors: list[str] = []
        path = seg_dir / str(stream.get("uri", ""))
        if not path.is_file():
            stream_errors.append(f"[{stream_id}] Missing encoder parquet")
        else:
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError, ImportError, KeyError) as exc:
                stream_errors.append(
                    f"[{stream_id}] Encoder parquet unreadable: {exc}"
                )
            else:
                frames[stream_id] = frame
                missing_columns = sorted(
                    required_columns - set(frame.columns)
                )
                if missing_columns:
                    stream_errors.append(
                        f"[{stream_id}] Missing columns: {missing_columns}"
                    )
                elif frame.empty:
                    stream_errors.append(
                        f"[{stream_id}] Encoder parquet is empty"
                    )
                else:
                    timestamp_ns = frame["timestamp_ns"].to_numpy(
                        dtype=np.int64,
                    )
                    source_timestamp_ns = frame[
                        "source_timestamp_ns"
                    ].to_numpy(dtype=np.int64)
                    log_time_ns = frame["log_time_ns"].to_numpy(
                        dtype=np.int64,
                    )
                    publish_time_ns = frame[
                        "publish_time_ns"
                    ].to_numpy(dtype=np.int64)
                    raw_value = frame["raw_value"].to_numpy(
                        dtype=np.float64,
                    )

                    for name, values in {
                        "timestamp_ns": timestamp_ns,
                        "source_timestamp_ns": source_timestamp_ns,
                        "log_time_ns": log_time_ns,
                        "publish_time_ns": publish_time_ns,
                    }.items():
                        if len(values) > 1 and not np.all(
                            np.diff(values) > 0
                        ):
                            stream_errors.append(
                                f"[{stream_id}] {name} is not monotonic"
                            )

                    if not np.array_equal(
                        timestamp_ns,
                        source_timestamp_ns - source_start_ns,
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Relative timestamps do not "
                            "match source timestamps"
                        )
                    if (
                        int(timestamp_ns[0]) < 0
                        or int(timestamp_ns[-1]) > timeline_end_ns
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Timestamps outside Segment"
                        )
                    if not np.isfinite(raw_value).all():
                        stream_errors.append(
                            f"[{stream_id}] raw_value contains NaN/Inf"
                        )

                    expected_robot_id = stream_id.removesuffix(
                        "_magnetic_encoder"
                    )
                    robot_ids = set(
                        frame["robot_id"].astype(str).unique()
                    )
                    if robot_ids != {expected_robot_id}:
                        stream_errors.append(
                            f"[{stream_id}] Robot rows are mixed: "
                            f"{sorted(robot_ids)}"
                        )
                    observed_robot_ids.update(robot_ids)

                    origin = stream.get("origin", {})
                    source_topic = str(origin.get("source_topic", ""))
                    if set(
                        frame["source_topic"].astype(str).unique()
                    ) != {source_topic}:
                        stream_errors.append(
                            f"[{stream_id}] source_topic mismatch"
                        )
                    if (
                        origin.get("source_asset_id")
                        not in source_asset_ids
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Unknown source_asset_id"
                        )
                    if origin.get("operation") != "trim_preserve_raw_value":
                        stream_errors.append(
                            f"[{stream_id}] Unexpected operation"
                        )
                    if (
                        stream.get("unit") != "unknown"
                        or stream.get("semantic_status")
                        != "raw_unverified"
                        or set(frame["unit"].astype(str).unique())
                        != {"unknown"}
                        or set(
                            frame["semantic_status"]
                            .astype(str)
                            .unique()
                        )
                        != {"raw_unverified"}
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Raw semantic contract violated"
                        )

                    expected_count = int(stream.get("sample_count", -1))
                    if expected_count != len(frame):
                        stream_errors.append(
                            f"[{stream_id}] sample_count "
                            f"({expected_count}) != rows ({len(frame)})"
                        )
                    stats[f"magnetic_encoder_rows_{stream_id}"] = len(
                        frame
                    )
                    topic_to_stream[source_topic] = stream_id

        errors.extend(stream_errors)
        checks[f"magnetic_encoder_{stream_id}"] = (
            "fail" if stream_errors else "pass"
        )

    expected_robot_ids = {"robot0", "robot1"}
    if observed_robot_ids != expected_robot_ids:
        errors.append(
            "Magnetic encoder robot coverage mismatch: "
            f"{sorted(observed_robot_ids)}"
        )
        checks["magnetic_encoder_dual_robot"] = "fail"
    else:
        checks["magnetic_encoder_dual_robot"] = "pass"

    source_rows, source_error = _read_source_magnetic_encoders(
        segment,
        set(topic_to_stream),
    )
    if source_error:
        errors.append(f"Magnetic encoder source read failed: {source_error}")
        checks["magnetic_encoder_source_match"] = "fail"
    elif source_rows is None:
        checks["magnetic_encoder_source_match"] = "skip"
    else:
        source_match = True
        for topic, stream_id in topic_to_stream.items():
            frame = frames[stream_id]
            raw_rows = source_rows.get(topic, [])
            if len(raw_rows) != len(frame):
                source_match = False
                errors.append(
                    f"[{stream_id}] Source rows ({len(raw_rows)}) "
                    f"!= parquet rows ({len(frame)})"
                )
                continue
            if not raw_rows:
                continue
            source_matrix = np.asarray(raw_rows, dtype=object)
            comparisons = [
                np.array_equal(
                    frame["source_timestamp_ns"].to_numpy(
                        dtype=np.int64,
                    ),
                    source_matrix[:, 0].astype(np.int64),
                ),
                np.array_equal(
                    frame["log_time_ns"].to_numpy(dtype=np.int64),
                    source_matrix[:, 1].astype(np.int64),
                ),
                np.array_equal(
                    frame["publish_time_ns"].to_numpy(dtype=np.int64),
                    source_matrix[:, 2].astype(np.int64),
                ),
                np.array_equal(
                    frame["raw_value"].to_numpy(dtype=np.float64),
                    source_matrix[:, 3].astype(np.float64),
                ),
            ]
            if not all(comparisons):
                source_match = False
                errors.append(
                    f"[{stream_id}] Parquet values differ from Raw MCAP"
                )
        checks["magnetic_encoder_source_match"] = (
            "pass" if source_match else "fail"
        )

    checks["magnetic_encoder_streams"] = (
        "fail" if errors else "pass"
    )
    return {
        "status": "fail" if errors else "pass",
        "checks": checks,
        "statistics": stats,
        "errors": errors,
    }


def validate_segment(output_dir: str) -> dict:
    """对已生成的 Prepared Segment 做写出后验证。

    Returns:
        {
            "status": "pass" | "fail",
            "checks": {...},
            "statistics": {...},
            "errors": [...],
        }
    """
    seg_dir = Path(output_dir)
    errors = []
    checks = {}
    stats = {}

    # ---- 1. segment.json 存在且可解析 ----
    seg_path = seg_dir / "segment.json"
    if not seg_path.exists():
        return {"status": "fail", "checks": {}, "statistics": {}, "errors": ["segment.json not found"]}

    with open(seg_path, encoding="utf-8") as f:
        segment = json.load(f)

    # ---- 2. 引用的文件全部存在 ----
    referenced = []
    for stream in segment.get("streams", []):
        uri = seg_dir / stream["uri"]
        referenced.append(str(uri))
        if not uri.exists():
            errors.append(f"Missing stream file: {uri}")

    calib_uri = seg_dir / segment.get("calibration_uri", "")
    if calib_uri.exists():
        referenced.append(str(calib_uri))
    else:
        errors.append(f"Missing calibration: {calib_uri}")

    # sample_map
    for stream in segment.get("streams", []):
        sm_uri = stream.get("origin", {}).get("sample_map_uri", "")
        if sm_uri:
            sm_path = seg_dir / sm_uri
            referenced.append(str(sm_path))
            if not sm_path.exists():
                errors.append(f"Missing sample_map: {sm_path}")

    checks["referenced_files_exist"] = "pass" if not any(
        "Missing" in e for e in errors
    ) else "fail"

    # ---- 2b. Raw 资产哈希（源文件可访问时）----
    source_hash_validation = validate_source_hashes(segment)
    checks.update(source_hash_validation["checks"])
    stats.update(source_hash_validation["statistics"])
    errors.extend(source_hash_validation["errors"])

    # ---- 3. 视频流可解码（按 segment.json 中的 streams 遍历） ----
    video_streams = [s for s in segment.get("streams", [])
                     if s.get("format") == "mp4"]
    all_video_ok = True
    video_frame_counts: dict[str, int] = {}
    for vs in video_streams:
        vpath = seg_dir / vs["uri"]
        try:
            video_ok, frame_count = _probe_video(vpath)
            video_frame_counts[vs["stream_id"]] = frame_count
            if not video_ok:
                all_video_ok = False
                errors.append(f"Video decode failed: {vs['uri']}")
        except (cv2.error, OSError):
            all_video_ok = False
            errors.append(f"Video open failed: {vs['uri']}")
    checks["video_decode"] = "pass" if all_video_ok and video_streams else "fail"

    # ---- 4. 视频帧数 == sample_map 行数（按 segment.json 中每个视频流检查） ----
    all_video_match = True
    for vs in video_streams:
        sm_uri = vs.get("origin", {}).get("sample_map_uri", "")
        if not sm_uri:
            continue
        sm_path = seg_dir / sm_uri
        if not sm_path.exists():
            errors.append(f"Missing sample_map: {sm_path}")
            all_video_match = False
            continue
        sm = pd.read_parquet(str(sm_path))
        stats[f"sample_map_rows_{vs['stream_id']}"] = len(sm)

        video_frames = video_frame_counts.get(vs["stream_id"], -1)
        stats[f"rgb_frames_{vs['stream_id']}"] = video_frames

        match = abs(video_frames - len(sm)) <= 2
        if not match:
            all_video_match = False
            errors.append(
                f"[{vs['stream_id']}] Video frames ({video_frames}) != sample_map rows ({len(sm)})"
            )
    checks["video_sample_map_count_match"] = "pass" if all_video_match and video_streams else "skip"

    # ---- 5. sample_map 时间单调（检查每个视频流的 sample_map） ----
    all_sm_mono = True
    for vs in video_streams:
        sm_uri = vs.get("origin", {}).get("sample_map_uri", "")
        if not sm_uri:
            continue
        sm_path = seg_dir / sm_uri
        if sm_path.exists():
            sm = pd.read_parquet(str(sm_path))
            ts = sm["output_timestamp_ns"].values
            monotonic = bool(np.all(np.diff(ts) > 0))
            if not monotonic:
                all_sm_mono = False
                errors.append(f"[{vs['stream_id']}] Sample map timestamps not monotonic")
            if "time_error_ns" in sm.columns:
                stats[f"max_mapping_error_ns_{vs['stream_id']}"] = int(sm["time_error_ns"].abs().max())
    checks["sample_map_monotonic"] = "pass" if all_sm_mono and video_streams else "skip"

    # ---- 6. 深度流完整性、dtype、单位和来源 ----
    depth_validation = validate_depth_streams(seg_dir, segment)
    checks.update(depth_validation["checks"])
    stats.update(depth_validation["statistics"])
    errors.extend(depth_validation["errors"])

    # ---- 7. IMU 可读且时间单调（按 segment.json 中每个 IMU 流检查） ----
    imu_streams = [s for s in segment.get("streams", []) if s.get("modality") == "imu"]
    for imu_s in imu_streams:
        imu_path = seg_dir / imu_s["uri"]
        if not imu_path.exists():
            errors.append(f"Missing IMU file: {imu_path}")
            continue
        imu = pd.read_parquet(str(imu_path))
        stats[f"imu_samples_{imu_s['stream_id']}"] = len(imu)
        ts = imu["timestamp_ns"].values
        imu_mono = bool(np.all(np.diff(ts) >= 0))
        if not imu_mono:
            errors.append(f"[{imu_s['stream_id']}] IMU timestamps not monotonic")
        checks[f"imu_monotonic_{imu_s['stream_id']}"] = "pass" if imu_mono else "fail"

        min_ts = imu["timestamp_ns"].min()
        checks[f"imu_starts_near_zero_{imu_s['stream_id']}"] = (
            "pass" if min_ts >= 0 and min_ts < 1_000_000_000 else "warn"
        )

    # ---- 8. UMI 磁编码器可追溯性与双路隔离 ----
    encoder_validation = validate_magnetic_encoder_streams(
        seg_dir,
        segment,
    )
    checks.update(encoder_validation["checks"])
    stats.update(encoder_validation["statistics"])
    errors.extend(encoder_validation["errors"])

    # ---- 9. UMI VIO 位姿、连续区间和 Raw 可追溯性 ----
    vio_validation = validate_vio_pose_streams(seg_dir, segment)
    checks.update(vio_validation["checks"])
    stats.update(vio_validation["statistics"])
    errors.extend(vio_validation["errors"])

    # ---- 10. 标注流验证（按 modality 分发） ----
    annotation_streams = [
        s for s in segment.get("streams", [])
        if s.get("role") == "annotation" or s.get("modality") == "hand_object_detection"
    ]
    if annotation_streams:
        # 提取源视频尺寸（用于 bbox 校验）
        video_streams_all = [
            s for s in segment.get("streams", [])
            if s.get("format") == "mp4"
        ]
        video_width = None
        video_height = None
        if video_streams_all:
            shape = video_streams_all[0].get("shape", [])
            if len(shape) >= 2:
                video_height, video_width = shape[0], shape[1]

        timeline_end_ns = segment["timeline"]["end_ns"]

        for ann_s in annotation_streams:
            modality = ann_s.get("modality", "")
            stream_id = ann_s.get("stream_id", "unknown")

            if modality == "hand_object_detection":
                ann_result = validate_hand_object_stream(
                    seg_dir=seg_dir,
                    stream=ann_s,
                    timeline_end_ns=timeline_end_ns,
                    video_width=video_width,
                    video_height=video_height,
                )
            elif modality == "instance_segmentation":
                ann_result = validate_mask_stream(
                    seg_dir=seg_dir,
                    stream=ann_s,
                    timeline_end_ns=timeline_end_ns,
                    video_width=video_width,
                    video_height=video_height,
                )
            else:
                continue

            # 合并结果
            for check_name, result in ann_result["checks"].items():
                checks[f"annotation_{stream_id}_{check_name}"] = result

            for stat_name, value in ann_result["statistics"].items():
                stats[f"annotation_{stream_id}_{stat_name}"] = value

            if ann_result["errors"]:
                errors.extend(ann_result["errors"])

    # ---- 10. 统计 ----
    stats["duration_ns"] = segment["timeline"]["end_ns"] - segment["timeline"]["start_ns"]

    # ---- 汇总 ----
    all_pass = len(errors) == 0
    status = "pass" if all_pass else "fail"

    return {
        "status": status,
        "checks": checks,
        "statistics": stats,
        "errors": errors,
    }


def write_validation_report(validation: dict, output_dir: str) -> str:
    """写出 validation.json。

    Returns:
        输出文件路径
    """
    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(validation, f, indent=2, ensure_ascii=False)
    return str(output_path)


def write_annotation_validation_report(validation: dict, output_dir: str) -> str:
    """从完整 validation 结果中提取标注部分，写出 annotation_validation.json。

    只保留 checks/statistics/errors 中与 annotation 相关的条目。

    Returns:
        输出文件路径，若无标注数据则返回空字符串
    """
    ann_checks = {
        k: v for k, v in validation.get("checks", {}).items()
        if k.startswith("annotation_")
    }
    ann_stats = {
        k: v for k, v in validation.get("statistics", {}).items()
        if k.startswith("annotation_")
    }
    ann_errors = [
        e for e in validation.get("errors", [])
        if any(kw in e.lower() for kw in ["annotation", "标注", "bbox", "mask", "rle", "entity", "hand", "object"])
    ]

    if not ann_checks and not ann_stats:
        return ""

    # 计算标注子状态
    ann_status = "pass"
    if any(v == "fail" for v in ann_checks.values()):
        ann_status = "fail"
    elif any(v == "warn" for v in ann_checks.values()):
        ann_status = "warn"

    report = {
        "status": ann_status,
        "checks": ann_checks,
        "statistics": ann_stats,
        "errors": ann_errors,
    }

    reports_dir = Path(output_dir) / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "annotation_validation.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return str(output_path)
