"""Write UMI VIO poses without inventing coordinate-frame semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zpds_prepare.readers.session_model import TimeSeriesStream

DEFAULT_CONTINUITY_GAP_NS = 500_000_000
QUATERNION_NORM_TOLERANCE = 0.01

REQUIRED_METADATA = {
    "robot_id",
    "source_topic",
    "translation_unit",
    "orientation_representation",
    "semantic_status",
    "child_frame",
    "transform_direction",
}
REQUIRED_COLUMNS = {
    "log_time_ns",
    "publish_time_ns",
    "tx",
    "ty",
    "tz",
    "qx",
    "qy",
    "qz",
    "qw",
    "source_frame_id",
    "source_header_topic",
}


def assign_continuity_groups(
    timestamps_ns: np.ndarray,
    source_frame_ids: np.ndarray,
    quaternions_xyzw: np.ndarray,
    *,
    minimum_gap_ns: int = DEFAULT_CONTINUITY_GAP_NS,
) -> tuple[np.ndarray, list[str], int]:
    """Split poses only on observable clock/frame/quaternion discontinuities."""
    count = len(timestamps_ns)
    if count == 0:
        return np.asarray([], dtype=np.int64), [], minimum_gap_ns
    if len(source_frame_ids) != count or len(quaternions_xyzw) != count:
        raise ValueError("Continuity inputs must have equal row counts")

    positive_intervals = np.diff(timestamps_ns)
    positive_intervals = positive_intervals[positive_intervals > 0]
    cadence_threshold_ns = (
        int(np.median(positive_intervals) * 10)
        if len(positive_intervals)
        else 0
    )
    gap_threshold_ns = max(int(minimum_gap_ns), cadence_threshold_ns)

    group_ids: np.ndarray = np.zeros(count, dtype=np.int64)
    reasons = [""] * count
    reasons[0] = "segment_start"
    group_id = 0

    for index in range(1, count):
        reason = ""
        if timestamps_ns[index] <= timestamps_ns[index - 1]:
            reason = "timestamp_non_increasing"
        elif (
            timestamps_ns[index] - timestamps_ns[index - 1]
            > gap_threshold_ns
        ):
            reason = "timestamp_gap"
        elif source_frame_ids[index] != source_frame_ids[index - 1]:
            reason = "reference_frame_change"
        elif not np.isfinite(quaternions_xyzw[index]).all() or (
            np.linalg.norm(quaternions_xyzw[index]) < 1e-12
        ):
            reason = "invalid_quaternion"

        if reason:
            group_id += 1
            reasons[index] = reason
        group_ids[index] = group_id

    return group_ids, reasons, gap_threshold_ns


def normalize_vio_pose(
    stream: TimeSeriesStream,
    source_start_ns: int,
    source_end_ns: int,
) -> tuple[pd.DataFrame, int]:
    """Trim one VIO stream while preserving Raw pose values and clocks."""
    if stream.modality != "vio_pose":
        raise ValueError(f"{stream.stream_id} is not a vio_pose stream")

    missing_metadata = sorted(REQUIRED_METADATA - set(stream.metadata))
    if missing_metadata:
        raise ValueError(
            f"{stream.stream_id} missing metadata: {missing_metadata}"
        )
    if not isinstance(stream.rows, pd.DataFrame):
        raise TypeError(
            f"{stream.stream_id} rows must be a pandas DataFrame"
        )

    missing_columns = sorted(REQUIRED_COLUMNS - set(stream.rows.columns))
    if missing_columns:
        raise ValueError(
            f"{stream.stream_id} missing columns: {missing_columns}"
        )

    source_timestamps = np.asarray(stream.timestamps_ns, dtype=np.int64)
    if len(source_timestamps) != len(stream.rows):
        raise ValueError(
            f"{stream.stream_id} timestamp count "
            f"({len(source_timestamps)}) != row count ({len(stream.rows)})"
        )

    mask = (
        (source_timestamps >= source_start_ns)
        & (source_timestamps <= source_end_ns)
    )
    indices = np.flatnonzero(mask)
    if not len(indices):
        raise ValueError(
            f"{stream.stream_id} has no samples in "
            f"[{source_start_ns}, {source_end_ns}]"
        )

    clipped = stream.rows.iloc[indices].reset_index(drop=True)
    clipped_source_ts = source_timestamps[indices]
    source_frame_ids = clipped["source_frame_id"].fillna("").astype(str)
    quaternions = clipped[["qx", "qy", "qz", "qw"]].to_numpy(
        dtype=np.float64
    )
    continuity_group_ids, continuity_reasons, gap_threshold_ns = (
        assign_continuity_groups(
            clipped_source_ts,
            source_frame_ids.to_numpy(dtype=str),
            quaternions,
        )
    )

    result = pd.DataFrame(
        {
            "timestamp_ns": clipped_source_ts - source_start_ns,
            "source_timestamp_ns": clipped_source_ts,
            "log_time_ns": clipped["log_time_ns"].to_numpy(
                dtype=np.int64
            ),
            "publish_time_ns": clipped["publish_time_ns"].to_numpy(
                dtype=np.int64
            ),
            **{
                column: clipped[column].to_numpy(dtype=np.float64)
                for column in ("tx", "ty", "tz", "qx", "qy", "qz", "qw")
            },
            "parent_frame": source_frame_ids,
            "child_frame": str(stream.metadata["child_frame"]),
            "source_frame_id": source_frame_ids,
            "robot_id": str(stream.metadata["robot_id"]),
            "source_topic": str(stream.metadata["source_topic"]),
            "source_header_topic": (
                clipped["source_header_topic"].fillna("").astype(str)
            ),
            "translation_unit": str(
                stream.metadata["translation_unit"]
            ),
            "orientation_representation": str(
                stream.metadata["orientation_representation"]
            ),
            "semantic_status": str(stream.metadata["semantic_status"]),
            "continuity_group_id": continuity_group_ids,
            "continuity_start_reason": continuity_reasons,
        }
    )
    return result, gap_threshold_ns


def write_vio_pose_stream(
    stream: TimeSeriesStream,
    output_dir: str,
    source_start_ns: int,
    source_end_ns: int,
) -> dict:
    """Write one normalized UMI VIO pose stream and its contract."""
    frame, gap_threshold_ns = normalize_vio_pose(
        stream=stream,
        source_start_ns=source_start_ns,
        source_end_ns=source_end_ns,
    )
    relative_uri = f"data/poses/{stream.stream_id}.parquet"
    output_path = Path(output_dir) / relative_uri
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)

    return {
        "stream_id": stream.stream_id,
        "role": stream.role,
        "modality": stream.modality,
        "uri": relative_uri,
        "rows": len(frame),
        "rate_hz": stream.expected_rate_hz,
        "frame_id": stream.frame_id,
        "source_asset_id": stream.metadata.get(
            "source_asset_id",
            "raw_mcap",
        ),
        "source_topic": stream.metadata["source_topic"],
        "unit": stream.metadata["translation_unit"],
        "semantic_status": stream.metadata["semantic_status"],
        "source_field": stream.metadata.get("source_field", "pose"),
        "operation": "trim_preserve_raw_pose_no_interpolation",
        "fields": [
            {
                "name": "translation",
                "columns": ["tx", "ty", "tz"],
                "dtype": "float64",
                "unit": stream.metadata["translation_unit"],
            },
            {
                "name": "orientation",
                "columns": ["qx", "qy", "qz", "qw"],
                "dtype": "float64",
                "representation": "quaternion_xyzw",
            },
        ],
        "coordinate_contract": {
            "source_schema": stream.metadata.get(
                "source_schema",
                "foxglove.PoseInFrame",
            ),
            "source_frame_field": "frame_id",
            "parent_frame_column": "parent_frame",
            "child_frame": stream.metadata["child_frame"],
            "transform_direction": stream.metadata["transform_direction"],
            "translation_unit": stream.metadata["translation_unit"],
            "orientation_representation": stream.metadata[
                "orientation_representation"
            ],
            "source_topic_authority": stream.metadata.get(
                "source_topic_authority",
                "mcap_channel",
            ),
        },
        "continuity": {
            "group_column": "continuity_group_id",
            "start_reason_column": "continuity_start_reason",
            "gap_threshold_ns": gap_threshold_ns,
            "interpolation_across_groups": "forbidden",
            "explicit_reset_signal_available": False,
        },
        "continuity_group_count": int(
            frame["continuity_group_id"].nunique()
        ),
    }


__all__ = [
    "DEFAULT_CONTINUITY_GAP_NS",
    "QUATERNION_NORM_TOLERANCE",
    "assign_continuity_groups",
    "normalize_vio_pose",
    "write_vio_pose_stream",
]
