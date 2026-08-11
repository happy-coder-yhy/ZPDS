"""ZPDS pre-clean quality report v1.9 contract constants.

The values live in code for validation, but are deliberately not embedded in
each report JSON.  Human-readable Chinese descriptions live in the companion
``quality_report_fields.final.v1.9.md`` document.
"""

from __future__ import annotations

from typing import Final

SCHEMA_VERSION: Final = "zpds.preclean_quality_report.v1.9"
LEGACY_SCHEMA_VERSIONS: Final = {
    "zpds.preclean_quality_report.v1.6",
    "zpds.preclean_quality_report.v1.7",
    "zpds.preclean_quality_report.v1.8",
}
SUPPORTED_REVIEW_SCHEMA_VERSIONS: Final = {SCHEMA_VERSION, *LEGACY_SCHEMA_VERSIONS}
ISSUE_SCHEMA_VERSION: Final = "0.2.0"

PROFILE_SOURCE_TYPE: Final = {
    "a2d": "hdf5+jpeg",
    "dunjia": "mcap",
    "epic": "mp4+pickle",
    "guida": "mkv+csv",
    "umi": "mcap",
}
ALLOWED_ASSET_FORMATS: Final = {
    "mcap",
    "mcap_topic",
    "hdf5",
    "hdf5_dataset",
    "jpeg",
    "png",
    "jpeg_sequence",
    "png_sequence",
    "mp4",
    "mkv",
    "csv",
    "json",
    "jsonl",
    "parquet",
    "pickle",
    "wav",
    "directory",
}
ALLOWED_MODALITIES: Final = {
    "rgb",
    "depth",
    "imu",
    "audio",
    "robot_state",
    "joint_state",
    "joint_command",
    "action",
    "force",
    "pose",
    "annotation",
    "hand_object_annotation",
    "mask",
    "magnetic_encoder",
    "vio_pose",
    "calibration",
    "container",
    "dataset_directory",
}
ALLOWED_STREAM_STATUSES: Final = {"available", "missing", "unreadable", "partial"}

ALLOWED_ACTIONS: Final = {
    "keep",
    "split",
    "quarantine",
    "reject",
    "manual_review",
}
LEGACY_RANGE_ACTIONS: Final = {"trim", "exclude_range"}
LEGACY_KEEP_ACTIONS: Final = {"keep_with_flag"}


def normalize_action(action: str) -> str:
    """Map legacy actions to the simplified v1.9 action set."""

    if action in LEGACY_RANGE_ACTIONS:
        return "split"
    if action in LEGACY_KEEP_ACTIONS:
        return "keep"
    return action
ALLOWED_SEVERITIES: Final = {"warning", "error", "critical"}
ALLOWED_CHECK_STATUSES: Final = {
    "passed",
    "warning",
    "failed",
    "not_run",
    "not_applicable",
    "unavailable",
}
ALLOWED_CHECK_APPLICABILITY: Final = {"applicable", "not_applicable"}
ALLOWED_REVIEW_STATUSES: Final = {"pending", "approved", "rejected", "modified"}
ALLOWED_REVIEW_DECISIONS: Final = {
    "accept_recommendation",
    "reject_issue",
    "modify_recommendation",
}
ALLOWED_REVIEW_REASON_CODES: Final = {
    "evidence_confirmed",
    "false_positive",
    "insufficient_evidence",
    "wrong_interval",
    "wrong_action",
    "impact_limited",
    "source_data_changed",
    "other",
}
ALLOWED_REPORT_REVIEW_STATUSES: Final = {"pending", "approved", "rejected", "returned"}
ALLOWED_REPORT_FINAL_RESULTS: Final = {
    "approved_for_cleaning",
    "returned_for_revision",
    "rejected",
}
ALLOWED_SEGMENT_STATUSES: Final = {"pending_review", "approved", "rejected", "modified"}
ALLOWED_SEGMENT_DISPOSITIONS: Final = {
    "keep",
    "quarantine",
    "reject",
}
ALLOWED_OVERALL_STATUSES: Final = {
    "passed",
    "passed_with_warnings",
    "review_required",
    "rejected",
    "incomplete",
}
ALLOWED_EVIDENCE_TYPES: Final = {"timestamp_point", "timestamp_range"}
ALLOWED_LOCATOR_METHODS: Final = {"nearest_timestamp", "range_overlap"}
ALLOWED_CLOCK_IDS: Final = {
    "source_device_clock",
    "unix_epoch",
    "ros_time",
    "monotonic_clock",
}
ALLOWED_CLEANING_STRATEGIES: Final = {
    "unified_master_timeline",
    "reviewed_issue_planning",
}
SYNCHRONIZATION_RULE: Final = "shared_master_timeline"

CHECK_IDS: Final = {
    "source_readability",
    "quality_detection",
    "qc_cascade",
    "scene_segmentation",
    "camera_completeness",
    "camera_robot_alignment",
    "action_finite",
    "joint_quality",
    "mcap_integrity",
    "rgb_frame_quality",
    "depth_coverage",
    "imu_continuity",
    "clock_alignment",
    "video_integrity",
    "frame_quality",
    "timestamp_continuity",
    "annotation_parse",
    "annotation_range",
    "rgb_imu_alignment",
    "privacy",
    "dual_camera_quality",
    "dual_camera_timestamps",
    "encoder_quality",
    "vio_pose_quality",
    "cross_stream_alignment",
    "calibration",
}

TOP_LEVEL_KEYS: Final = {
    "schema_version",
    "quality_issue_schema_version",
    "report_metadata",
    "dataset",
    "source_assets",
    "stream_inventory",
    "check_coverage",
    "issues",
    "evidence_index",
    "proposed_cleaning",
    "summary",
    "overall_result",
    "report_review",
    "integrity",
}
