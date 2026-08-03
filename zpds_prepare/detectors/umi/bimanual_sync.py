"""Estimate reviewable robot0/robot1 timestamp alignment for UMI."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def _as_int64(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.int64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    return array


def _default_tolerance_ns(left: np.ndarray, right: np.ndarray) -> int:
    medians: list[int] = []
    for timestamps in (left, right):
        diffs = np.diff(timestamps)
        positive = diffs[diffs > 0]
        if len(positive):
            medians.append(int(np.median(positive)))
    return max(medians, default=50_000_000) * 2


def _group_bounds(
    timestamps: np.ndarray,
    groups: np.ndarray,
) -> dict[int, tuple[int, int]]:
    return {
        int(group_id): (
            int(timestamps[groups == group_id].min()),
            int(timestamps[groups == group_id].max()),
        )
        for group_id in np.unique(groups)
    }


def build_dual_alignment(
    robot0_timestamps_ns: Sequence[int] | np.ndarray,
    robot1_timestamps_ns: Sequence[int] | np.ndarray,
    *,
    robot0_groups: Sequence[int] | np.ndarray | None = None,
    robot1_groups: Sequence[int] | np.ndarray | None = None,
    max_residual_ns: int | None = None,
    mapping_method: str = "inferred",
) -> tuple[pd.DataFrame, dict]:
    """Nearest-match robot0 to robot1 within overlapping continuity groups.

    The function never matches samples across non-overlapping continuity
    groups.  It does not interpolate, resample, or claim shared-clock authority.
    """
    if mapping_method not in {"direct", "inferred"}:
        raise ValueError("mapping_method must be 'direct' or 'inferred'")

    robot0 = _as_int64(robot0_timestamps_ns, "robot0_timestamps_ns")
    robot1 = _as_int64(robot1_timestamps_ns, "robot1_timestamps_ns")
    groups0 = (
        np.zeros(len(robot0), dtype=np.int64)
        if robot0_groups is None
        else _as_int64(robot0_groups, "robot0_groups")
    )
    groups1 = (
        np.zeros(len(robot1), dtype=np.int64)
        if robot1_groups is None
        else _as_int64(robot1_groups, "robot1_groups")
    )
    if len(groups0) != len(robot0) or len(groups1) != len(robot1):
        raise ValueError("continuity group count must match timestamp count")

    tolerance_ns = (
        _default_tolerance_ns(robot0, robot1)
        if max_residual_ns is None
        else int(max_residual_ns)
    )
    if tolerance_ns < 0:
        raise ValueError("max_residual_ns must be non-negative")

    bounds0 = _group_bounds(robot0, groups0) if len(robot0) else {}
    bounds1 = _group_bounds(robot1, groups1) if len(robot1) else {}
    rows: list[dict] = []

    for index0, timestamp0 in enumerate(robot0):
        group0 = int(groups0[index0])
        left0, right0 = bounds0[group0]
        candidate_indices: list[int] = []
        for group1, (left1, right1) in bounds1.items():
            if max(left0, left1) <= min(right0, right1):
                candidate_indices.extend(np.flatnonzero(groups1 == group1).tolist())

        if candidate_indices:
            candidates = np.asarray(candidate_indices, dtype=np.int64)
            residuals = np.abs(robot1[candidates] - timestamp0)
            best_position = int(np.argmin(residuals))
            index1 = int(candidates[best_position])
            signed_residual_ns = int(robot1[index1] - timestamp0)
            within_tolerance = abs(signed_residual_ns) <= tolerance_ns
        else:
            index1 = -1
            signed_residual_ns = 0
            within_tolerance = False

        rows.append(
            {
                "robot0_sample_index": index0,
                "robot0_timestamp_ns": int(timestamp0),
                "robot0_continuity_group": group0,
                "robot1_sample_index": index1 if within_tolerance else pd.NA,
                "robot1_timestamp_ns": (
                    int(robot1[index1]) if within_tolerance else pd.NA
                ),
                "robot1_continuity_group": (
                    int(groups1[index1]) if within_tolerance else pd.NA
                ),
                "residual_ns": signed_residual_ns if within_tolerance else pd.NA,
                "mapping_method": mapping_method if within_tolerance else "unavailable",
                "uncertainty_ns": tolerance_ns,
                "interpolated": False,
            }
        )

    alignment = pd.DataFrame(rows)
    for column in (
        "robot1_sample_index",
        "robot1_timestamp_ns",
        "robot1_continuity_group",
        "residual_ns",
    ):
        if column in alignment:
            alignment[column] = alignment[column].astype("Int64")

    available = alignment[alignment["mapping_method"] != "unavailable"]
    absolute_residuals = (
        available["residual_ns"].abs().to_numpy(dtype=np.int64)
        if len(available)
        else np.asarray([], dtype=np.int64)
    )
    summary = {
        "robot0_sample_count": len(robot0),
        "robot1_sample_count": len(robot1),
        "mapped_count": len(available),
        "unmapped_count": len(alignment) - len(available),
        "mapped_ratio": round(len(available) / len(alignment), 6)
        if len(alignment)
        else 0.0,
        "residual_p50_ns": int(np.percentile(absolute_residuals, 50))
        if len(absolute_residuals)
        else None,
        "residual_p95_ns": int(np.percentile(absolute_residuals, 95))
        if len(absolute_residuals)
        else None,
        "residual_max_ns": int(absolute_residuals.max())
        if len(absolute_residuals)
        else None,
        "max_residual_ns": tolerance_ns,
        "mapping_method": mapping_method,
        "interpolation_used": False,
        "cross_group_mapping": "forbidden",
    }
    return alignment, summary


__all__ = ["build_dual_alignment"]
