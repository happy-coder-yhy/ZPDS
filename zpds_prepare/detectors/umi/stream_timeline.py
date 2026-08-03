"""Build reviewable continuity evidence for one UMI stream timeline."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue

DEFAULT_MINIMUM_GAP_NS = 500_000_000
DEFAULT_GAP_FACTOR = 10.0


def analyze_stream_timeline(
    stream_id: str,
    timestamps_ns: Sequence[int] | np.ndarray,
    *,
    expected_rate_hz: float | None = None,
    minimum_gap_ns: int = DEFAULT_MINIMUM_GAP_NS,
    gap_factor: float = DEFAULT_GAP_FACTOR,
) -> tuple[pd.DataFrame, dict, list[QualityIssue]]:
    """Analyze one Raw clock without sorting or repairing its samples.

    A new continuity group starts at a duplicate, rollback, or long gap.  The
    returned evidence preserves source order so downstream code cannot hide a
    clock reset by sorting timestamps.
    """
    timestamps = np.asarray(timestamps_ns, dtype=np.int64)
    if timestamps.ndim != 1:
        raise ValueError("timestamps_ns must be one-dimensional")
    if minimum_gap_ns <= 0:
        raise ValueError("minimum_gap_ns must be positive")
    if gap_factor <= 0:
        raise ValueError("gap_factor must be positive")

    sample_count = len(timestamps)
    deltas = np.zeros(sample_count, dtype=np.int64)
    if sample_count > 1:
        deltas[1:] = np.diff(timestamps)

    positive_deltas = deltas[deltas > 0]
    median_interval_ns = (
        int(np.median(positive_deltas)) if len(positive_deltas) else None
    )
    observed_rate_hz = (
        1_000_000_000 / median_interval_ns
        if median_interval_ns and median_interval_ns > 0
        else 0.0
    )
    cadence_gap_ns = (
        int(median_interval_ns * gap_factor) if median_interval_ns else 0
    )
    gap_threshold_ns = max(int(minimum_gap_ns), cadence_gap_ns)

    group_ids = np.zeros(sample_count, dtype=np.int64)
    start_reasons = [""] * sample_count
    if sample_count:
        start_reasons[0] = "stream_start"

    issues: list[QualityIssue] = []
    group_id = 0
    for index in range(1, sample_count):
        delta_ns = int(deltas[index])
        reason = ""
        if delta_ns < 0:
            reason = "timestamp_rollback"
            issues.append(
                QualityIssue(
                    issue_type="umi_timestamp_rollback",
                    stream_id=stream_id,
                    start_ns=int(timestamps[index - 1]),
                    end_ns=int(timestamps[index]),
                    severity="critical",
                    decision="split",
                    details={
                        "sample_index": index,
                        "delta_ns": delta_ns,
                        "mapping_across_boundary": "forbidden",
                    },
                )
            )
        elif delta_ns == 0:
            reason = "timestamp_duplicate"
            issues.append(
                QualityIssue(
                    issue_type="umi_timestamp_duplicate",
                    stream_id=stream_id,
                    start_ns=int(timestamps[index]),
                    end_ns=int(timestamps[index]),
                    severity="warning",
                    decision="keep_with_flag",
                    details={
                        "sample_index": index,
                        "mapping_across_boundary": "forbidden",
                    },
                )
            )
        elif delta_ns > gap_threshold_ns:
            reason = "timestamp_gap"
            issues.append(
                QualityIssue(
                    issue_type="umi_timestamp_gap",
                    stream_id=stream_id,
                    start_ns=int(timestamps[index - 1]),
                    end_ns=int(timestamps[index]),
                    severity="error",
                    decision="split",
                    details={
                        "sample_index": index,
                        "gap_ns": delta_ns,
                        "gap_threshold_ns": gap_threshold_ns,
                        "mapping_across_boundary": "forbidden",
                    },
                )
            )

        if reason:
            group_id += 1
            start_reasons[index] = reason
        group_ids[index] = group_id

    evidence = pd.DataFrame(
        {
            "sample_index": np.arange(sample_count, dtype=np.int64),
            "timestamp_ns": timestamps,
            "delta_ns": deltas,
            "continuity_group": group_ids,
            "continuity_start_reason": pd.Series(start_reasons, dtype="string"),
        }
    )
    summary = {
        "stream_id": stream_id,
        "sample_count": sample_count,
        "start_ns": int(timestamps[0]) if sample_count else None,
        "end_ns": int(timestamps[-1]) if sample_count else None,
        "expected_rate_hz": expected_rate_hz,
        "observed_rate_hz": round(observed_rate_hz, 6),
        "median_interval_ns": median_interval_ns,
        "gap_threshold_ns": gap_threshold_ns,
        "continuity_group_count": int(group_ids.max() + 1) if sample_count else 0,
        "duplicate_count": int(np.sum(deltas[1:] == 0)),
        "rollback_count": int(np.sum(deltas[1:] < 0)),
        "long_gap_count": int(np.sum(deltas[1:] > gap_threshold_ns)),
        "source_order_preserved": True,
        "interpolation_across_groups": "forbidden",
    }
    return evidence, summary, issues


__all__ = [
    "DEFAULT_GAP_FACTOR",
    "DEFAULT_MINIMUM_GAP_NS",
    "analyze_stream_timeline",
]
