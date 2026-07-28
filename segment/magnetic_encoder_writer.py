"""Write UMI magnetic-encoder streams without assigning physical semantics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from zpds_prepare.readers.session_model import TimeSeriesStream

REQUIRED_METADATA = {
    "robot_id",
    "source_topic",
    "unit",
    "semantic_status",
}


def normalize_magnetic_encoder(
    stream: TimeSeriesStream,
    source_start_ns: int,
    source_end_ns: int,
) -> pd.DataFrame:
    """Trim one encoder stream while preserving source clocks and raw values."""
    if stream.modality != "magnetic_encoder":
        raise ValueError(
            f"{stream.stream_id} is not a magnetic_encoder stream"
        )

    missing_metadata = sorted(REQUIRED_METADATA - set(stream.metadata))
    if missing_metadata:
        raise ValueError(
            f"{stream.stream_id} missing metadata: {missing_metadata}"
        )
    if not isinstance(stream.rows, pd.DataFrame):
        raise TypeError(
            f"{stream.stream_id} rows must be a pandas DataFrame"
        )

    required_columns = {
        "log_time_ns",
        "publish_time_ns",
        "raw_value",
    }
    missing_columns = sorted(required_columns - set(stream.rows.columns))
    if missing_columns:
        raise ValueError(
            f"{stream.stream_id} missing columns: {missing_columns}"
        )

    source_timestamps = np.asarray(stream.timestamps_ns, dtype=np.int64)
    if len(source_timestamps) != len(stream.rows):
        raise ValueError(
            f"{stream.stream_id} timestamp count ({len(source_timestamps)}) "
            f"!= row count ({len(stream.rows)})"
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
    result = pd.DataFrame(
        {
            "timestamp_ns": clipped_source_ts - source_start_ns,
            "source_timestamp_ns": clipped_source_ts,
            "log_time_ns": clipped["log_time_ns"].to_numpy(
                dtype=np.int64,
            ),
            "publish_time_ns": clipped["publish_time_ns"].to_numpy(
                dtype=np.int64,
            ),
            "raw_value": clipped["raw_value"].to_numpy(dtype=np.float64),
            "robot_id": str(stream.metadata["robot_id"]),
            "source_topic": str(stream.metadata["source_topic"]),
            "unit": str(stream.metadata["unit"]),
            "semantic_status": str(stream.metadata["semantic_status"]),
        }
    )
    return result


def write_magnetic_encoder_stream(
    stream: TimeSeriesStream,
    output_dir: str,
    source_start_ns: int,
    source_end_ns: int,
) -> dict:
    """Write one normalized UMI encoder stream and return segment metadata."""
    frame = normalize_magnetic_encoder(
        stream=stream,
        source_start_ns=source_start_ns,
        source_end_ns=source_end_ns,
    )
    relative_uri = f"data/robot/{stream.stream_id}.parquet"
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
        "unit": stream.metadata["unit"],
        "semantic_status": stream.metadata["semantic_status"],
        "source_field": stream.metadata.get("source_field", "value"),
        "operation": "trim_preserve_raw_value",
        "fields": [
            {
                "name": "raw_value",
                "dtype": "float64",
                "unit": stream.metadata["unit"],
            }
        ],
    }


__all__ = [
    "normalize_magnetic_encoder",
    "write_magnetic_encoder_stream",
]
