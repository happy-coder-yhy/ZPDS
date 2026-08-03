"""Raw-level UMI VIO checks without inventing coordinate semantics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from segment.vio_pose_writer import (
    DEFAULT_CONTINUITY_GAP_NS,
    QUATERNION_NORM_TOLERANCE,
    assign_continuity_groups,
)
from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.readers.session_model import TimeSeriesStream

POSE_COLUMNS = ("tx", "ty", "tz", "qx", "qy", "qz", "qw")


def analyze_vio_quality(
    stream: TimeSeriesStream,
    *,
    minimum_gap_ns: int = DEFAULT_CONTINUITY_GAP_NS,
    quaternion_norm_tolerance: float = QUATERNION_NORM_TOLERANCE,
    translation_step_mad_factor: float = 10.0,
) -> tuple[pd.DataFrame, dict, list[QualityIssue]]:
    """Return per-sample VIO evidence, summary, and reviewable issues.

    Translation-step outliers are statistical candidates only.  They never
    imply physical velocity or drift while translation units and frames remain
    unverified.
    """
    if stream.modality != "vio_pose":
        raise ValueError(f"{stream.stream_id} is not a vio_pose stream")
    if not isinstance(stream.rows, pd.DataFrame):
        raise TypeError("VIO rows must be a pandas DataFrame")
    required = set(POSE_COLUMNS) | {"source_frame_id"}
    missing = sorted(required - set(stream.rows.columns))
    if missing:
        raise ValueError(f"{stream.stream_id} missing VIO columns: {missing}")

    timestamps = np.asarray(stream.timestamps_ns, dtype=np.int64)
    if len(timestamps) != len(stream.rows):
        raise ValueError("VIO timestamp count must match row count")
    pose = stream.rows.loc[:, POSE_COLUMNS].to_numpy(dtype=np.float64)
    translations = pose[:, :3]
    quaternions = pose[:, 3:]
    source_frames = (
        stream.rows["source_frame_id"].fillna("").astype(str).to_numpy()
    )
    source_topic = str(stream.metadata.get("source_topic", ""))
    if "source_header_topic" in stream.rows:
        header_topics = (
            stream.rows["source_header_topic"].fillna("").astype(str).to_numpy()
        )
    else:
        header_topics = np.asarray([""] * len(timestamps), dtype=str)
    header_topic_available = header_topics != ""
    header_topic_mismatch = (
        header_topic_available & (header_topics != source_topic)
        if source_topic
        else np.zeros(len(timestamps), dtype=bool)
    )

    groups, reasons, gap_threshold_ns = assign_continuity_groups(
        timestamps,
        source_frames,
        quaternions,
        minimum_gap_ns=minimum_gap_ns,
    )
    finite_pose = np.isfinite(pose).all(axis=1)
    quaternion_norm = np.linalg.norm(quaternions, axis=1)
    quaternion_valid = np.isfinite(quaternion_norm) & (
        np.abs(quaternion_norm - 1.0) <= quaternion_norm_tolerance
    )

    translation_step = np.full(len(timestamps), np.nan, dtype=np.float64)
    same_group = np.zeros(len(timestamps), dtype=bool)
    if len(timestamps) > 1:
        same_group[1:] = groups[1:] == groups[:-1]
        deltas = np.linalg.norm(np.diff(translations, axis=0), axis=1)
        translation_step[1:] = deltas
        translation_step[~same_group] = np.nan

    finite_steps = translation_step[np.isfinite(translation_step)]
    step_median = float(np.median(finite_steps)) if len(finite_steps) else None
    step_mad = (
        float(np.median(np.abs(finite_steps - step_median)))
        if len(finite_steps) and step_median is not None
        else None
    )
    step_threshold = None
    step_outlier = np.zeros(len(timestamps), dtype=bool)
    if len(finite_steps) >= 4 and step_median is not None and step_mad is not None:
        robust_scale = max(step_mad, np.finfo(np.float64).eps)
        step_threshold = step_median + translation_step_mad_factor * robust_scale
        step_outlier = np.isfinite(translation_step) & (
            translation_step > step_threshold
        )

    evidence = pd.DataFrame(
        {
            "sample_index": np.arange(len(timestamps), dtype=np.int64),
            "timestamp_ns": timestamps,
            "continuity_group": groups,
            "continuity_start_reason": pd.Series(reasons, dtype="string"),
            "source_frame_id": pd.Series(source_frames, dtype="string"),
            "source_header_topic": pd.Series(header_topics, dtype="string"),
            "header_topic_matches_channel": pd.Series(
                np.where(
                    header_topic_available,
                    ~header_topic_mismatch,
                    pd.NA,
                ),
                dtype="boolean",
            ),
            "pose_finite": finite_pose,
            "quaternion_norm": quaternion_norm,
            "quaternion_valid": quaternion_valid,
            "translation_step_raw": translation_step,
            "translation_step_outlier": step_outlier,
        }
    )
    issues: list[QualityIssue] = []

    for index in np.flatnonzero(~finite_pose):
        issues.append(
            QualityIssue(
                issue_type="umi_vio_non_finite_pose",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index]),
                end_ns=int(timestamps[index]),
                severity="error",
                decision="quarantine",
                details={"sample_index": int(index), "vio_ready": False},
            )
        )
    for index in np.flatnonzero(~quaternion_valid):
        issues.append(
            QualityIssue(
                issue_type="umi_vio_invalid_quaternion",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index]),
                end_ns=int(timestamps[index]),
                severity="error",
                decision="quarantine",
                details={
                    "sample_index": int(index),
                    "quaternion_norm": float(quaternion_norm[index]),
                    "tolerance": quaternion_norm_tolerance,
                    "vio_ready": False,
                },
            )
        )
    for index, reason in enumerate(reasons):
        if index == 0 or not reason:
            continue
        decision = "split" if reason in {
            "timestamp_non_increasing",
            "timestamp_gap",
        } else "keep_with_flag"
        issues.append(
            QualityIssue(
                issue_type=f"umi_vio_{reason}",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index - 1]),
                end_ns=int(timestamps[index]),
                severity="error" if decision == "split" else "warning",
                decision=decision,
                details={
                    "sample_index": index,
                    "explicit_reset": False,
                    "mapping_method": "inferred",
                    "interpolation_across_boundary": "forbidden",
                },
            )
        )
    for index in np.flatnonzero(step_outlier):
        issues.append(
            QualityIssue(
                issue_type="umi_vio_translation_step_candidate",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[index - 1]),
                end_ns=int(timestamps[index]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "sample_index": int(index),
                    "raw_step": float(translation_step[index]),
                    "statistical_threshold": float(step_threshold),
                    "method": "median_mad_candidate",
                    "physical_semantics": "unavailable",
                },
            )
        )

    mismatch_indices = np.flatnonzero(header_topic_mismatch)
    if len(mismatch_indices):
        observed_topics = sorted(set(header_topics[mismatch_indices].tolist()))
        issues.append(
            QualityIssue(
                issue_type="umi_vio_header_topic_mismatch",
                stream_id=stream.stream_id,
                start_ns=int(timestamps[mismatch_indices[0]]),
                end_ns=int(timestamps[mismatch_indices[-1]]),
                severity="warning",
                decision="keep_with_flag",
                details={
                    "mismatch_count": len(mismatch_indices),
                    "channel_topic": source_topic,
                    "header_topics": observed_topics,
                    "source_metadata_integrity": "suspect",
                    "physical_pose_values_modified": False,
                },
            )
        )

    summary = {
        "stream_id": stream.stream_id,
        "sample_count": len(timestamps),
        "continuity_group_count": int(groups.max() + 1) if len(groups) else 0,
        "gap_threshold_ns": gap_threshold_ns,
        "non_finite_pose_count": int((~finite_pose).sum()),
        "invalid_quaternion_count": int((~quaternion_valid).sum()),
        "translation_step_candidate_count": int(step_outlier.sum()),
        "translation_step_median_raw": step_median,
        "translation_step_mad_raw": step_mad,
        "translation_step_threshold_raw": step_threshold,
        "header_topic_available_count": int(header_topic_available.sum()),
        "header_topic_mismatch_count": int(header_topic_mismatch.sum()),
        "translation_unit": stream.metadata.get("translation_unit", "unknown"),
        "semantic_status": stream.metadata.get("semantic_status", "unknown"),
        "explicit_reset_signal_available": False,
        "reset_inference": "observable_discontinuity_only",
        "interpolation_across_groups": "forbidden",
    }
    return evidence, summary, issues


__all__ = ["analyze_vio_quality"]
