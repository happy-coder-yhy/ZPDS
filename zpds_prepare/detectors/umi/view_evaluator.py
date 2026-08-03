"""Candidate-only UMI quality views pending the shared dependency contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from zpds_prepare.detectors.umi.orchestrator import UmiEvidenceBundle

CANDIDATE_VIEW_VERSION = "umi-candidate-views-v1"


@dataclass(frozen=True)
class UmiCandidateView:
    """Non-formal view result that cannot automatically reject source data."""

    name: str
    status: str
    applicability: str
    disposition: str
    dependencies: tuple[str, ...]
    reasons: tuple[str, ...] = ()
    measurements: dict[str, Any] = field(default_factory=dict)
    version: str = CANDIDATE_VIEW_VERSION
    formal: bool = False
    automatic_reject: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _view(
    name: str,
    status: str,
    dependencies: tuple[str, ...],
    *,
    reasons: list[str] | None = None,
    measurements: dict[str, Any] | None = None,
) -> UmiCandidateView:
    if status not in {"candidate_pass", "review_required", "unavailable"}:
        raise ValueError(f"invalid candidate view status: {status}")
    return UmiCandidateView(
        name=name,
        status=status,
        applicability="unavailable" if status == "unavailable" else "applicable",
        disposition="keep" if status == "candidate_pass" else "keep_with_flag",
        dependencies=dependencies,
        reasons=tuple(reasons or []),
        measurements=dict(measurements or {}),
    )


def evaluate_candidate_views(
    bundle: UmiEvidenceBundle,
    *,
    require_vio_for_bimanual: bool = True,
) -> dict[str, UmiCandidateView]:
    """Evaluate isolated UMI candidates without calibrated reject thresholds."""
    video_summaries = bundle.timeline_summaries["video"]
    video_issues = [
        issue
        for issues in bundle.timeline_issues["video"].values()
        for issue in issues
    ]
    video_reasons = sorted(
        {
            issue.issue_type
            for issue in video_issues
            if issue.severity in {"error", "critical"}
        }
    )
    if not video_summaries:
        observation = _view(
            "robot_observation_ready_candidate",
            "unavailable",
            ("umi_video_timeline",),
            reasons=["no_video_stream"],
        )
    elif video_reasons:
        observation = _view(
            "robot_observation_ready_candidate",
            "review_required",
            ("umi_video_timeline",),
            reasons=video_reasons,
            measurements={"video_stream_count": len(video_summaries)},
        )
    else:
        observation = _view(
            "robot_observation_ready_candidate",
            "candidate_pass",
            ("umi_video_timeline",),
            measurements={"video_stream_count": len(video_summaries)},
        )

    vio_reasons: list[str] = []
    vio_measurements: dict[str, Any] = {
        "vio_stream_count": len(bundle.vio_summaries)
    }
    for stream_id, summary in bundle.vio_summaries.items():
        for field_name in (
            "non_finite_pose_count",
            "invalid_quaternion_count",
            "translation_step_candidate_count",
            "header_topic_mismatch_count",
        ):
            value = int(summary[field_name])
            vio_measurements[f"{stream_id}.{field_name}"] = value
            if value:
                vio_reasons.append(f"{stream_id}:{field_name}")
        if int(summary["continuity_group_count"]) > 1:
            vio_reasons.append(f"{stream_id}:multiple_continuity_groups")

    if not bundle.vio_summaries:
        vio = _view(
            "vio_ready_candidate",
            "unavailable",
            ("umi_vio_quality",),
            reasons=["no_vio_stream"],
            measurements=vio_measurements,
        )
    elif vio_reasons:
        vio = _view(
            "vio_ready_candidate",
            "review_required",
            ("umi_vio_quality",),
            reasons=sorted(set(vio_reasons)),
            measurements=vio_measurements,
        )
    else:
        vio = _view(
            "vio_ready_candidate",
            "candidate_pass",
            ("umi_vio_quality",),
            measurements=vio_measurements,
        )

    video_alignment_keys = sorted(
        key for key in bundle.dual_alignments if key.startswith("video:")
    )
    required_alignment_keys = list(video_alignment_keys)
    bimanual_reasons: list[str] = []
    if not video_alignment_keys:
        bimanual_reasons.append("no_dual_video_alignment")
    if require_vio_for_bimanual:
        required_alignment_keys.append("time_series:vio_pose")
        if "time_series:vio_pose" not in bundle.dual_alignments:
            bimanual_reasons.append("no_dual_vio_alignment")

    bimanual_measurements: dict[str, Any] = {
        "required_alignment_keys": required_alignment_keys,
        "residual_threshold_calibrated": False,
    }
    for key in required_alignment_keys:
        summary = bundle.dual_alignment_summaries.get(key)
        if summary is None:
            continue
        bimanual_measurements[f"{key}.mapped_ratio"] = summary["mapped_ratio"]
        bimanual_measurements[f"{key}.residual_p95_ns"] = summary[
            "residual_p95_ns"
        ]
        if float(summary["mapped_ratio"]) < 1.0:
            bimanual_reasons.append(f"{key}:partial_mapping")

    if require_vio_for_bimanual and vio.status == "review_required":
        bimanual_reasons.append("vio_review_required")

    missing_requirement = any(
        reason.startswith("no_dual_") for reason in bimanual_reasons
    )
    if missing_requirement:
        bimanual_status = "unavailable"
    elif bimanual_reasons:
        bimanual_status = "review_required"
    else:
        bimanual_status = "candidate_pass"
    bimanual = _view(
        "bimanual_umi_ready_candidate",
        bimanual_status,
        tuple(required_alignment_keys),
        reasons=sorted(set(bimanual_reasons)),
        measurements=bimanual_measurements,
    )

    return {
        observation.name: observation,
        vio.name: vio,
        bimanual.name: bimanual,
    }


__all__ = [
    "CANDIDATE_VIEW_VERSION",
    "UmiCandidateView",
    "evaluate_candidate_views",
]
