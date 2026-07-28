"""Validate UMI VIO pose streams and their Raw MCAP lineage."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from segment.vio_pose_writer import (
    QUATERNION_NORM_TOLERANCE,
    assign_continuity_groups,
)

REQUIRED_COLUMNS = {
    "timestamp_ns",
    "source_timestamp_ns",
    "log_time_ns",
    "publish_time_ns",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "parent_frame",
    "child_frame",
    "source_frame_id",
    "robot_id",
    "source_topic",
    "source_header_topic",
    "translation_unit",
    "orientation_representation",
    "semantic_status",
    "continuity_group_id",
    "continuity_start_reason",
}


def _read_source_vio_poses(
    segment: dict,
    topics: set[str],
) -> tuple[dict[str, list[tuple]] | None, str | None]:
    """Read Raw VIO rows once when the source MCAP remains accessible."""
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
        rows: dict[str, list[tuple]] = {topic: [] for topic in topics}
        with source_path.open("rb") as source_file:
            reader = make_reader(
                source_file,
                decoder_factories=[DecoderFactory()],
            )
            for _schema, channel, message, decoded in (
                reader.iter_decoded_messages(topics=list(topics))
            ):
                timestamp_ns = int(decoded.header.timestamp)
                if not source_start_ns <= timestamp_ns <= source_end_ns:
                    continue
                pose = decoded.pose
                rows[channel.topic].append(
                    (
                        timestamp_ns,
                        int(message.log_time),
                        int(message.publish_time),
                        float(pose.position.x),
                        float(pose.position.y),
                        float(pose.position.z),
                        float(pose.orientation.x),
                        float(pose.orientation.y),
                        float(pose.orientation.z),
                        float(pose.orientation.w),
                        str(decoded.frame_id or ""),
                        str(decoded.header.topic_name or ""),
                    )
                )
        return rows, None
    except (ImportError, OSError, ValueError, AttributeError) as exc:
        return None, str(exc)


def _validate_continuity(
    frame: pd.DataFrame,
    stream: dict,
    stream_id: str,
) -> list[str]:
    errors: list[str] = []
    group_ids = frame["continuity_group_id"].to_numpy(dtype=np.int64)
    reasons = frame["continuity_start_reason"].fillna("").astype(str)
    source_timestamps = frame["source_timestamp_ns"].to_numpy(
        dtype=np.int64
    )

    if int(group_ids[0]) != 0:
        errors.append(f"[{stream_id}] First continuity group must be 0")
    group_diffs = np.diff(group_ids)
    if len(group_diffs) and not np.isin(group_diffs, [0, 1]).all():
        errors.append(
            f"[{stream_id}] continuity_group_id must increase by one"
        )
    if reasons.iloc[0] != "segment_start":
        errors.append(
            f"[{stream_id}] First continuity reason must be segment_start"
        )
    for index in range(1, len(frame)):
        starts_group = group_ids[index] != group_ids[index - 1]
        if starts_group != bool(reasons.iloc[index]):
            errors.append(
                f"[{stream_id}] Continuity reason mismatch at row {index}"
            )
            break

    for group_id in np.unique(group_ids):
        group_timestamps = source_timestamps[group_ids == group_id]
        if len(group_timestamps) > 1 and not np.all(
            np.diff(group_timestamps) > 0
        ):
            errors.append(
                f"[{stream_id}] Source time is not monotonic in "
                f"continuity group {group_id}"
            )

    continuity = stream.get("continuity", {})
    if (
        continuity.get("group_column") != "continuity_group_id"
        or continuity.get("interpolation_across_groups") != "forbidden"
        or continuity.get("explicit_reset_signal_available") is not False
    ):
        errors.append(f"[{stream_id}] Invalid continuity contract")
        return errors

    gap_threshold_ns = int(continuity.get("gap_threshold_ns", 0))
    expected_groups, expected_reasons, _ = assign_continuity_groups(
        source_timestamps,
        frame["source_frame_id"].fillna("").astype(str).to_numpy(),
        frame[["qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64),
        minimum_gap_ns=gap_threshold_ns,
    )
    if not np.array_equal(group_ids, expected_groups) or (
        reasons.tolist() != expected_reasons
    ):
        errors.append(
            f"[{stream_id}] Continuity groups do not match observable breaks"
        )

    expected_group_count = int(stream.get("continuity_group_count", -1))
    actual_group_count = int(pd.Series(group_ids).nunique())
    if expected_group_count != actual_group_count:
        errors.append(
            f"[{stream_id}] continuity_group_count "
            f"({expected_group_count}) != {actual_group_count}"
        )
    return errors


def validate_vio_pose_streams(seg_dir: Path, segment: dict) -> dict:
    """Validate dual VIO streams, coordinate contracts, and Raw lineage."""
    pose_streams = [
        stream
        for stream in segment.get("streams", [])
        if stream.get("modality") == "vio_pose"
    ]
    if not pose_streams:
        return {
            "status": "skip",
            "checks": {"vio_pose_streams": "skip"},
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
    frames: dict[str, pd.DataFrame] = {}
    topic_to_stream: dict[str, str] = {}
    observed_robot_ids: set[str] = set()

    for stream in pose_streams:
        stream_id = str(stream.get("stream_id", "unknown_vio_pose"))
        stream_errors: list[str] = []
        path = seg_dir / str(stream.get("uri", ""))
        if not path.is_file():
            stream_errors.append(f"[{stream_id}] Missing VIO parquet")
        else:
            try:
                frame = pd.read_parquet(path)
            except (OSError, ValueError, ImportError, KeyError) as exc:
                stream_errors.append(
                    f"[{stream_id}] VIO parquet unreadable: {exc}"
                )
            else:
                frames[stream_id] = frame
                missing_columns = sorted(
                    REQUIRED_COLUMNS - set(frame.columns)
                )
                if missing_columns:
                    stream_errors.append(
                        f"[{stream_id}] Missing columns: {missing_columns}"
                    )
                elif frame.empty:
                    stream_errors.append(
                        f"[{stream_id}] VIO parquet is empty"
                    )
                else:
                    timestamp_ns = frame["timestamp_ns"].to_numpy(
                        dtype=np.int64
                    )
                    source_timestamp_ns = frame[
                        "source_timestamp_ns"
                    ].to_numpy(dtype=np.int64)
                    log_time_ns = frame["log_time_ns"].to_numpy(
                        dtype=np.int64
                    )
                    publish_time_ns = frame[
                        "publish_time_ns"
                    ].to_numpy(dtype=np.int64)

                    if not np.array_equal(
                        timestamp_ns,
                        source_timestamp_ns - source_start_ns,
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Relative timestamps do not "
                            "match source timestamps"
                        )
                    if (
                        int(timestamp_ns.min()) < 0
                        or int(timestamp_ns.max()) > timeline_end_ns
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Timestamps outside Segment"
                        )
                    for name, values in {
                        "log_time_ns": log_time_ns,
                        "publish_time_ns": publish_time_ns,
                    }.items():
                        if len(values) > 1 and not np.all(
                            np.diff(values) > 0
                        ):
                            stream_errors.append(
                                f"[{stream_id}] {name} is not monotonic"
                            )

                    pose_values = frame[
                        ["tx", "ty", "tz", "qx", "qy", "qz", "qw"]
                    ].to_numpy(dtype=np.float64)
                    if not np.isfinite(pose_values).all():
                        stream_errors.append(
                            f"[{stream_id}] Pose contains NaN/Inf"
                        )
                    quaternion_norms = np.linalg.norm(
                        pose_values[:, 3:],
                        axis=1,
                    )
                    if (
                        not np.isfinite(quaternion_norms).all()
                        or np.any(
                            np.abs(quaternion_norms - 1.0)
                            > QUATERNION_NORM_TOLERANCE
                        )
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Quaternion norm is invalid"
                        )

                    expected_robot_id = stream_id.removesuffix("_vio_pose")
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
                    if origin.get("source_asset_id") not in source_asset_ids:
                        stream_errors.append(
                            f"[{stream_id}] Unknown source_asset_id"
                        )
                    if (
                        origin.get("operation")
                        != "trim_preserve_raw_pose_no_interpolation"
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Unexpected operation"
                        )

                    contract = stream.get("coordinate_contract", {})
                    if (
                        contract.get("source_frame_field") != "frame_id"
                        or contract.get("child_frame") != "unknown"
                        or contract.get("transform_direction") != "unknown"
                        or contract.get("translation_unit") != "unknown"
                        or contract.get("orientation_representation")
                        != "quaternion_xyzw"
                        or contract.get("source_topic_authority")
                        != "mcap_channel"
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Invalid coordinate contract"
                        )
                    if not (
                        frame["parent_frame"].astype(str)
                        == frame["source_frame_id"].astype(str)
                    ).all():
                        stream_errors.append(
                            f"[{stream_id}] parent_frame changed Raw frame_id"
                        )
                    if (
                        frame["source_frame_id"].astype(str).eq("").any()
                        or set(frame["child_frame"].astype(str).unique())
                        != {"unknown"}
                        or set(
                            frame["translation_unit"].astype(str).unique()
                        )
                        != {"unknown"}
                        or set(
                            frame["orientation_representation"]
                            .astype(str)
                            .unique()
                        )
                        != {"quaternion_xyzw"}
                        or set(
                            frame["semantic_status"].astype(str).unique()
                        )
                        != {"raw_unverified"}
                    ):
                        stream_errors.append(
                            f"[{stream_id}] Raw pose semantic contract "
                            "violated"
                        )

                    stream_errors.extend(
                        _validate_continuity(frame, stream, stream_id)
                    )

                    expected_count = int(stream.get("sample_count", -1))
                    if expected_count != len(frame):
                        stream_errors.append(
                            f"[{stream_id}] sample_count "
                            f"({expected_count}) != rows ({len(frame)})"
                        )
                    stats[f"vio_pose_rows_{stream_id}"] = len(frame)
                    stats[f"vio_pose_groups_{stream_id}"] = int(
                        frame["continuity_group_id"].nunique()
                    )
                    stats[f"vio_header_topic_mismatch_{stream_id}"] = int(
                        (
                            frame["source_header_topic"].astype(str)
                            != frame["source_topic"].astype(str)
                        ).sum()
                    )
                    topic_to_stream[source_topic] = stream_id

        errors.extend(stream_errors)
        checks[f"vio_pose_{stream_id}"] = (
            "fail" if stream_errors else "pass"
        )

    expected_robot_ids = {"robot0", "robot1"}
    if observed_robot_ids != expected_robot_ids:
        errors.append(
            "VIO pose robot coverage mismatch: "
            f"{sorted(observed_robot_ids)}"
        )
        checks["vio_pose_dual_robot"] = "fail"
    else:
        checks["vio_pose_dual_robot"] = "pass"

    source_rows, source_error = _read_source_vio_poses(
        segment,
        set(topic_to_stream),
    )
    if source_error:
        errors.append(f"VIO pose source read failed: {source_error}")
        checks["vio_pose_source_match"] = "fail"
    elif source_rows is None:
        checks["vio_pose_source_match"] = "skip"
    else:
        source_match = True
        numeric_columns = [
            "source_timestamp_ns",
            "log_time_ns",
            "publish_time_ns",
            "tx",
            "ty",
            "tz",
            "qx",
            "qy",
            "qz",
            "qw",
        ]
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
            for column_index, column_name in enumerate(numeric_columns):
                dtype = np.int64 if column_index < 3 else np.float64
                if not np.array_equal(
                    frame[column_name].to_numpy(dtype=dtype),
                    source_matrix[:, column_index].astype(dtype),
                ):
                    source_match = False
                    errors.append(
                        f"[{stream_id}] {column_name} differs from Raw MCAP"
                    )
            for column_name, column_index in (
                ("source_frame_id", 10),
                ("source_header_topic", 11),
            ):
                if not np.array_equal(
                    frame[column_name].astype(str).to_numpy(),
                    source_matrix[:, column_index].astype(str),
                ):
                    source_match = False
                    errors.append(
                        f"[{stream_id}] {column_name} differs from Raw MCAP"
                    )
        checks["vio_pose_source_match"] = (
            "pass" if source_match else "fail"
        )

    checks["vio_pose_streams"] = "fail" if errors else "pass"
    return {
        "status": "fail" if errors else "pass",
        "checks": checks,
        "statistics": stats,
        "errors": errors,
    }


__all__ = ["validate_vio_pose_streams"]
