"""Raw-level UMI magnetic-encoder checks with no gripper semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.detectors.umi.stream_timeline import analyze_stream_timeline
from zpds_prepare.readers.session_model import TimeSeriesStream


def _freeze_spans(values: np.ndarray, minimum_samples: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for index in range(1, len(values) + 1):
        closes = index == len(values) or not (
            np.isfinite(values[index])
            and np.isfinite(values[index - 1])
            and values[index] == values[index - 1]
        )
        if closes:
            if index - start >= minimum_samples:
                spans.append((start, index - 1))
            start = index
    return spans


def analyze_magnetic_encoder(
    stream: TimeSeriesStream,
    *,
    freeze_min_samples: int = 10,
    range_mad_factor: float = 10.0,
    minimum_gap_ns: int = 500_000_000,
) -> tuple[pd.DataFrame, dict, list[QualityIssue]]:
    """Check raw encoder integrity while preserving ``raw_unverified`` status."""
    if stream.modality != "magnetic_encoder":
        raise ValueError(f"{stream.stream_id} is not a magnetic_encoder stream")
    if freeze_min_samples < 2:
        raise ValueError("freeze_min_samples must be at least 2")
    if not isinstance(stream.rows, pd.DataFrame):
        raise TypeError("magnetic encoder rows must be a pandas DataFrame")
    if "raw_value" not in stream.rows:
        raise ValueError(f"{stream.stream_id} missing raw_value")

    timestamps = np.asarray(stream.timestamps_ns, dtype=np.int64)
    values = stream.rows["raw_value"].to_numpy(dtype=np.float64)
    if len(timestamps) != len(values):
        raise ValueError("encoder timestamp count must match row count")

    timeline, timeline_summary, timeline_issues = analyze_stream_timeline(
        stream.stream_id,
        timestamps,
        expected_rate_hz=stream.expected_rate_hz,
        minimum_gap_ns=minimum_gap_ns,
    )
    finite = np.isfinite(values)
    finite_values = values[finite]
    value_median = float(np.median(finite_values)) if len(finite_values) else None
    value_mad = (
        float(np.median(np.abs(finite_values - value_median)))
        if len(finite_values) and value_median is not None
        else None
    )
    range_candidate = np.zeros(len(values), dtype=bool)
    range_threshold = None
    if len(finite_values) >= 5 and value_mad and value_mad > 0:
        range_threshold = range_mad_factor * value_mad
        range_candidate = finite & (
            np.abs(values - value_median) > range_threshold
        )

    freeze_spans = _freeze_spans(values, freeze_min_samples)
    frozen = np.zeros(len(values), dtype=bool)
    for start, end in freeze_spans:
        frozen[start : end + 1] = True

    evidence = timeline.copy()
    evidence["raw_value"] = values
    evidence["value_finite"] = finite
    evidence["freeze_candidate"] = frozen
    evidence["range_candidate"] = range_candidate
    evidence["unit"] = str(stream.metadata.get("unit", "unknown"))
    evidence["semantic_status"] = str(
        stream.metadata.get("semantic_status", "raw_unverified")
    )

    issues = list(timeline_issues)
    for index in np.flatnonzero(~finite):
        issues.append(
            QualityIssue(
                issue_type="umi_magnetic_encoder_non_finite",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index]),
                end_ns=int(timestamps[index]),
                severity="error",
                decision="keep_with_flag",
                details={
                    "sample_index": int(index),
                    "semantic_status": "raw_unverified",
                },
            )
        )
    for start, end in freeze_spans:
        issues.append(
            QualityIssue(
                issue_type="umi_magnetic_encoder_freeze_candidate",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[start]),
                end_ns=int(timestamps[end]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "start_sample": start,
                    "end_sample": end,
                    "sample_count": end - start + 1,
                    "raw_value": float(values[start]),
                    "physical_interpretation": "unavailable",
                },
            )
        )
    for index in np.flatnonzero(range_candidate):
        issues.append(
            QualityIssue(
                issue_type="umi_magnetic_encoder_range_candidate",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index]),
                end_ns=int(timestamps[index]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "sample_index": int(index),
                    "raw_value": float(values[index]),
                    "method": "median_mad_candidate",
                    "physical_range": "unavailable",
                },
            )
        )

    semantic_status = str(
        stream.metadata.get("semantic_status", "raw_unverified")
    )
    summary = {
        **timeline_summary,
        "finite_ratio": round(float(finite.mean()), 6) if len(finite) else 0.0,
        "freeze_span_count": len(freeze_spans),
        "range_candidate_count": int(range_candidate.sum()),
        "raw_value_median": value_median,
        "raw_value_mad": value_mad,
        "raw_range_threshold": range_threshold,
        "unit": stream.metadata.get("unit", "unknown"),
        "semantic_status": semantic_status,
        "gripper_action_generated": False,
        "open_close_event_generated": False,
        "stall_generated": False,
        "physical_interpretation": (
            "unavailable" if semantic_status == "raw_unverified" else "external_contract"
        ),
    }
    return evidence, summary, issues


__all__ = ["analyze_magnetic_encoder"]
