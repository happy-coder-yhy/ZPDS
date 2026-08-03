"""In-memory orchestration for contract-independent UMI evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.detectors.umi.bimanual_sync import build_dual_alignment
from zpds_prepare.detectors.umi.magnetic_encoder_quality import (
    analyze_magnetic_encoder,
)
from zpds_prepare.detectors.umi.stream_timeline import analyze_stream_timeline
from zpds_prepare.detectors.umi.vio_quality import analyze_vio_quality
from zpds_prepare.readers.session_model import Session


@dataclass
class UmiEvidenceBundle:
    """All UMI detector output for one session, retained only in memory."""

    session_id: str
    timeline_evidence: dict[str, dict[str, pd.DataFrame]] = field(
        default_factory=lambda: {"video": {}, "imu": {}, "time_series": {}}
    )
    timeline_summaries: dict[str, dict[str, dict[str, Any]]] = field(
        default_factory=lambda: {"video": {}, "imu": {}, "time_series": {}}
    )
    timeline_issues: dict[str, dict[str, list[QualityIssue]]] = field(
        default_factory=lambda: {"video": {}, "imu": {}, "time_series": {}}
    )
    vio_evidence: dict[str, pd.DataFrame] = field(default_factory=dict)
    vio_summaries: dict[str, dict[str, Any]] = field(default_factory=dict)
    vio_issues: dict[str, list[QualityIssue]] = field(default_factory=dict)
    magnetic_encoder_evidence: dict[str, pd.DataFrame] = field(
        default_factory=dict
    )
    magnetic_encoder_summaries: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )
    magnetic_encoder_issues: dict[str, list[QualityIssue]] = field(
        default_factory=dict
    )
    dual_alignments: dict[str, pd.DataFrame] = field(default_factory=dict)
    dual_alignment_summaries: dict[str, dict[str, Any]] = field(
        default_factory=dict
    )

    @property
    def issues(self) -> list[QualityIssue]:
        """Return every issue without translating it into a quality decision."""
        result: list[QualityIssue] = []
        for category in self.timeline_issues.values():
            for stream_issues in category.values():
                result.extend(stream_issues)
        for collection in (self.vio_issues, self.magnetic_encoder_issues):
            for stream_issues in collection.values():
                result.extend(stream_issues)
        return result


def _robot_id(stream_id: str, metadata: dict[str, Any] | None = None) -> str | None:
    candidate = str((metadata or {}).get("robot_id", ""))
    if candidate in {"robot0", "robot1"}:
        return candidate
    for robot_id in ("robot0", "robot1"):
        if stream_id == robot_id or stream_id.startswith(f"{robot_id}_"):
            return robot_id
    return None


def _stream_suffix(stream_id: str, robot_id: str | None) -> str:
    prefix = f"{robot_id}_" if robot_id else ""
    return stream_id[len(prefix) :] if prefix and stream_id.startswith(prefix) else stream_id


def analyze_umi_session(
    session: Session,
    *,
    minimum_gap_ns: int = 500_000_000,
    alignment_max_residual_ns: int | None = None,
    alignment_mapping_method: str = "inferred",
    encoder_freeze_min_samples: int = 10,
) -> UmiEvidenceBundle:
    """Analyze all UMI stream clocks and compose reviewable evidence.

    The orchestration has no persistence, manifest, applicability, or quality-
    view behavior.  Magnetic-encoder results retain their source semantics and
    cannot become gripper actions here.
    """
    bundle = UmiEvidenceBundle(session_id=session.session_id)
    pair_candidates: dict[str, dict[str, tuple[str, pd.DataFrame]]] = {}

    def add_timeline(
        category: str,
        stream_id: str,
        timestamps_ns: Any,
        expected_rate_hz: float | None,
        pair_key: str,
        robot_id: str | None,
    ) -> None:
        evidence, summary, issues = analyze_stream_timeline(
            stream_id,
            timestamps_ns,
            expected_rate_hz=expected_rate_hz,
            minimum_gap_ns=minimum_gap_ns,
        )
        summary = {**summary, "stream_category": category}
        bundle.timeline_evidence[category][stream_id] = evidence
        bundle.timeline_summaries[category][stream_id] = summary
        bundle.timeline_issues[category][stream_id] = issues
        if robot_id is not None:
            pair_candidates.setdefault(pair_key, {})[robot_id] = (
                stream_id,
                evidence,
            )

    for stream_id, stream in session.video_streams.items():
        robot_id = _robot_id(stream_id)
        suffix = _stream_suffix(stream_id, robot_id)
        add_timeline(
            "video",
            stream_id,
            stream.timestamps_ns,
            stream.fps,
            f"video:{suffix}",
            robot_id,
        )

    for stream_id, stream in session.imu_streams.items():
        if "timestamp_ns" not in stream.dataframe:
            raise ValueError(f"{stream_id} IMU dataframe missing timestamp_ns")
        robot_id = _robot_id(stream_id)
        suffix = _stream_suffix(stream_id, robot_id)
        add_timeline(
            "imu",
            stream_id,
            stream.dataframe["timestamp_ns"].to_numpy(),
            stream.sample_rate_hz,
            f"imu:{suffix}",
            robot_id,
        )

    for stream_id, stream in session.time_series_streams.items():
        robot_id = _robot_id(stream_id, stream.metadata)
        add_timeline(
            "time_series",
            stream_id,
            stream.timestamps_ns,
            stream.expected_rate_hz,
            f"time_series:{stream.modality}",
            robot_id,
        )
        if stream.modality == "vio_pose":
            evidence, summary, issues = analyze_vio_quality(
                stream,
                minimum_gap_ns=minimum_gap_ns,
            )
            bundle.vio_evidence[stream_id] = evidence
            bundle.vio_summaries[stream_id] = summary
            bundle.vio_issues[stream_id] = issues
            if robot_id is not None:
                pair_candidates[f"time_series:{stream.modality}"][robot_id] = (
                    stream_id,
                    evidence,
                )
        elif stream.modality == "magnetic_encoder":
            evidence, summary, issues = analyze_magnetic_encoder(
                stream,
                freeze_min_samples=encoder_freeze_min_samples,
                minimum_gap_ns=minimum_gap_ns,
            )
            bundle.magnetic_encoder_evidence[stream_id] = evidence
            bundle.magnetic_encoder_summaries[stream_id] = summary
            bundle.magnetic_encoder_issues[stream_id] = issues

    for pair_key, streams in sorted(pair_candidates.items()):
        if set(streams) != {"robot0", "robot1"}:
            continue
        stream0_id, evidence0 = streams["robot0"]
        stream1_id, evidence1 = streams["robot1"]
        alignment, summary = build_dual_alignment(
            evidence0["timestamp_ns"].to_numpy(),
            evidence1["timestamp_ns"].to_numpy(),
            robot0_groups=evidence0["continuity_group"].to_numpy(),
            robot1_groups=evidence1["continuity_group"].to_numpy(),
            max_residual_ns=alignment_max_residual_ns,
            mapping_method=alignment_mapping_method,
        )
        alignment.insert(0, "modality", pair_key)
        alignment.insert(1, "robot0_stream_id", stream0_id)
        alignment.insert(2, "robot1_stream_id", stream1_id)
        bundle.dual_alignments[pair_key] = alignment
        bundle.dual_alignment_summaries[pair_key] = {
            **summary,
            "modality": pair_key,
            "robot0_stream_id": stream0_id,
            "robot1_stream_id": stream1_id,
        }

    return bundle


__all__ = ["UmiEvidenceBundle", "analyze_umi_session"]
