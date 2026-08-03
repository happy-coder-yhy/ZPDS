"""Formal integration boundary for UMI, Dunjia and A2D source QC reports.

The source detectors retain ownership of their algorithms.  This module only
normalizes their typed reports into the shared ZPDS contract and writes an
immutable-source revision manifest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.core.quality import QualityMetric, QualityReport, QualityView
from zpds.prepared.revision import (
    RevisionManifest,
    deterministic_config_hash,
    write_revision_manifest,
)
from zpds.segmentation.robot_spans import IdleCandidate, propose_edge_idle, propose_physical_spans


def _severity(value: str) -> Severity:
    return {
        "critical": Severity.FATAL,
        "fatal": Severity.FATAL,
        "error": Severity.ERROR,
        "warning": Severity.WARN,
        "warn": Severity.WARN,
    }.get(value, Severity.INFO)


def _disposition(value: str) -> Disposition:
    return {
        "pass": Disposition.KEEP,
        "keep": Disposition.KEEP,
        "keep_with_flag": Disposition.KEEP_WITH_FLAG,
        "quarantine": Disposition.QUARANTINE,
        "trim": Disposition.TRIM,
        "split": Disposition.SPLIT,
        "reject": Disposition.REJECT,
        "reject_view": Disposition.REJECT,
        "unavailable": Disposition.KEEP_WITH_FLAG,
    }.get(value, Disposition.KEEP_WITH_FLAG)


def _view_name(name: str) -> str:
    return {
        "robot_observation_ready_candidate": "robot_observation_ready",
        "vio_ready_candidate": "vio_ready",
        "bimanual_umi_ready_candidate": "bimanual_umi_ready",
    }.get(name, name)


@dataclass(frozen=True)
class RobotQCDelivery:
    """Result of source-report adaptation; semantic VLM is intentionally not run."""

    report: QualityReport
    manifest: RevisionManifest
    physical_spans: tuple[dict[str, Any], ...]
    idle_candidates: tuple[dict[str, Any], ...]


class FormalRobotQualityAdapter:
    """Implementation of Person A's ``UmiSharedContractAdapter`` protocol."""

    def __init__(self, *, producer: str = "zpds.robot_qc", version: str = "v1") -> None:
        self.producer = producer
        self.version = version

    def adapt_metric(self, metric: Any) -> QualityMetric:
        """Preserve all provisional fields without inventing physical semantics."""
        return QualityMetric(
            name=metric.metric_name,
            value=metric.value,
            threshold=None,
            comparison="none",
            pass_=None,
            unit=metric.unit,
            applicability=metric.applicability,
            severity=_severity(metric.severity),
            disposition=_disposition(metric.disposition),
            reason_code=metric.reason_code,
            start_ns=metric.start_ns,
            end_ns=metric.end_ns,
            evidence_uri=metric.evidence_uri,
            producer=metric.producer,
            version=metric.version,
            config_hash=metric.config_hash,
            details={
                "stream_id": metric.stream_id,
                **metric.details,
                "evaluation_status": "not_evaluated",
            },
        )

    def adapt_view(self, view: Any) -> QualityView:
        status = getattr(view, "status", None)
        return QualityView(
            name=_view_name(view.name),
            ready=status == "candidate_pass" if status is not None else bool(view.ready),
            applicability=view.applicability,
            disposition=_disposition(view.disposition),
            reasons=tuple(view.reasons),
            dependencies=tuple(getattr(view, "dependencies", getattr(view, "depends_on", ()))),
            evidence_uris=tuple(getattr(view, "evidence_uris", ())),
            producer=self.producer,
            version=getattr(view, "version", self.version),
        )

    def adapt_revision(self, revision: dict[str, Any]) -> RevisionManifest:
        modalities = {
            name: value["applicability"] if isinstance(value, dict) else str(value)
            for name, value in revision["modalities"].items()
        }
        return RevisionManifest(
            revision_id="r0001",
            source_session_id=revision["source_session_id"],
            profile=revision["profile"],
            source_assets=list(revision["source_assets"]),
            modalities=modalities,
            quality_views={},
            evidence_index=dict(revision.get("evidence_artifacts", {})),
            producer=str(revision.get("producer", self.producer)),
            version=str(revision.get("producer_version", self.version)),
            config_hash=str(revision["config_hash"]),
        )


def adapt_source_views(
    views_report: Any,
    *,
    producer: str,
    version: str,
    config_hash: str,
) -> dict[str, QualityView]:
    """Adapt Dunjia/A2D source views without coupling to their dataclasses."""
    adapted: dict[str, QualityView] = {}
    for name, view in views_report.views.items():
        disposition = _disposition(view.disposition)
        adapted[name] = QualityView(
            name=name,
            ready=bool(view.ready),
            applicability="unavailable" if view.disposition == "unavailable" else "applicable",
            disposition=disposition,
            reasons=tuple(view.reasons),
            dependencies=tuple(view.depends_on),
            evidence_uris=tuple(view.evidence_uris),
            producer=producer,
            version=version,
            config_hash=config_hash,
        )
    return adapted


def source_views_to_decisions(views: dict[str, QualityView]) -> list[Decision]:
    """Create only view-level decisions; source reason text remains in ``detail``."""
    decisions: list[Decision] = []
    for view in views.values():
        if view.disposition is Disposition.KEEP and view.ready:
            continue
        severity = Severity.ERROR if view.disposition is Disposition.REJECT else Severity.WARN
        decisions.append(
            Decision(
                stage=12,
                reason=ReasonCode.SOURCE_QUALITY_FLAG,
                severity=severity,
                message=f"Quality view {view.name}: {view.disposition.value}",
                disposition=view.disposition,
                detail={
                    "view": view.name,
                    "applicability": view.applicability,
                    "reasons": list(view.reasons),
                    "evidence_uris": list(view.evidence_uris),
                },
            )
        )
    return decisions


def build_robot_qc_delivery(
    *,
    session_id: str,
    profile: str,
    source_assets: list[dict[str, Any]],
    modalities: dict[str, str],
    views: dict[str, QualityView],
    metrics: list[QualityMetric] | None = None,
    evidence_index: dict[str, str] | None = None,
    stream_ranges: dict[str, tuple[int, int]] | None = None,
    idle_timestamps_ns: list[int] | None = None,
    robot_motion_energy: list[float] | None = None,
    gripper_event_energy: list[float] | None = None,
    visual_change_energy: list[float] | None = None,
    idle_thresholds: dict[str, float] | None = None,
    effective_config: dict[str, Any] | None = None,
    producer: str = "zpds.robot_qc",
    version: str = "v1",
) -> RobotQCDelivery:
    """Build a formal, source-immutable QC delivery with semantic review skipped."""
    config = effective_config or {}
    config_hash = deterministic_config_hash(config)
    decisions = source_views_to_decisions(views)
    physical = propose_physical_spans(stream_ranges or {}, decisions)
    thresholds = idle_thresholds or {}
    idle: list[IdleCandidate] = []
    if idle_timestamps_ns:
        idle = propose_edge_idle(
            idle_timestamps_ns,
            robot_motion_energy,
            gripper_event_energy,
            visual_change_energy,
            motion_max=float(thresholds.get("robot_motion_max", 0.0)),
            gripper_max=float(thresholds.get("gripper_event_max", 0.0)),
            visual_change_max=float(thresholds.get("visual_change_max", 0.0)),
            min_samples=int(thresholds.get("min_samples", 1)),
        )
    report = QualityReport(
        session_id=session_id,
        decisions=decisions,
        metrics=list(metrics or []),
        quality_views=views,
        overall_pass=not any(d.severity in {Severity.FATAL, Severity.ERROR} for d in decisions),
    )
    manifest = RevisionManifest(
        revision_id="r0001",
        source_session_id=session_id,
        profile=profile,
        source_assets=source_assets,
        modalities=modalities,
        quality_views=views,
        metrics=report.metrics,
        decisions=[
            {
                **asdict(decision),
                "reason": decision.reason.value,
                "reason_code": decision.reason.value,
                "severity": decision.severity.value,
                "disposition": decision.disposition.value if decision.disposition else None,
                "start_ns": decision.timestamp_ns,
                "end_ns": decision.end_timestamp_ns,
            }
            for decision in decisions
        ],
        physical_spans=[asdict(span) for span in physical],
        idle_candidates=[
            {**asdict(candidate), "disposition": candidate.disposition.value}
            for candidate in idle
        ],
        evidence_index=dict(evidence_index or {}),
        outcome={"value": "unknown", "status": "not_run", "reason": "vlm_semantic_skipped"},
        producer=producer,
        version=version,
        config_hash=config_hash,
    )
    return RobotQCDelivery(
        report=report,
        manifest=manifest,
        physical_spans=tuple(manifest.physical_spans),
        idle_candidates=tuple(manifest.idle_candidates),
    )


def run_umi_formal_session(
    session: Any,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    producer_version: str = "v1",
) -> RobotQCDelivery:
    """Run Person A's detector then adapt its candidate delivery to the formal schema."""
    from zpds_prepare.detectors.umi.provisional_pipeline import run_umi_provisional_session

    provisional = run_umi_provisional_session(
        session, output_dir, config=config, producer_version=producer_version
    )
    adapter = FormalRobotQualityAdapter(version=producer_version)
    views = {
        adapted.name: adapted
        for view in provisional.candidate_views.values()
        for adapted in (adapter.adapt_view(view),)
    }
    source_assets = list(provisional.revision_candidate["source_assets"])
    modalities = {
        name: value["applicability"]
        for name, value in provisional.revision_candidate["modalities"].items()
    }
    ranges = {
        stream_id: (int(min(stream.timestamps_ns)), int(max(stream.timestamps_ns)))
        for stream_id, stream in session.video_streams.items()
        if stream.timestamps_ns
    }
    delivery = build_robot_qc_delivery(
        session_id=session.session_id,
        profile="jianzhi_umi",
        source_assets=source_assets,
        modalities=modalities,
        views=views,
        metrics=[adapter.adapt_metric(metric) for metric in provisional.evidence_index.metrics],
        evidence_index=provisional.evidence_index.artifacts,
        stream_ranges=ranges,
        effective_config=provisional.effective_config,
        producer="zpds.robot_qc",
        version=producer_version,
    )
    manifest_path = Path(output_dir) / "revision.json"
    write_revision_manifest(manifest_path, delivery.manifest)
    return delivery


def run_dunjia_formal_session(
    session: Any,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    producer_version: str = "v1",
) -> RobotQCDelivery:
    """Run Person B's Dunjia B1-B5 detectors then adapt to the formal schema."""
    from zpds_prepare.detectors.dunjia import (
        check_dunjia_completeness,
        check_dunjia_coverage,
        check_dunjia_imu,
        check_dunjia_rgbd,
        aggregate_dunjia_quality_views,
    )

    cfg = config or {}
    require_depth = bool(cfg.get("dunjia", {}).get("depth", {}).get("required", True))
    max_pairing_ns = int(cfg.get("dunjia", {}).get("rgbd", {}).get("max_pairing_offset_ns", 50_000_000))
    spike_factor = float(cfg.get("dunjia", {}).get("imu", {}).get("spike_std_factor", 6.0))

    # B1-B4: run detectors
    completeness = check_dunjia_completeness(session, require_depth=require_depth)
    rgbd = check_dunjia_rgbd(session, max_pairing_offset_ns=max_pairing_ns)
    imu = check_dunjia_imu(session, spike_std_factor=spike_factor)
    coverage = check_dunjia_coverage(session)

    # B5: aggregate views
    views_report = aggregate_dunjia_quality_views(
        completeness=completeness,
        rgbd=rgbd,
        imu=imu,
        coverage=coverage,
        session_id=session.session_id,
        source_path=str(session.source_path),
    )

    config_hash = deterministic_config_hash(cfg)
    views = adapt_source_views(
        views_report,
        producer="person-b.dunjia",
        version=producer_version,
        config_hash=config_hash,
    )

    source_assets: list[dict[str, Any]] = [
        {"asset_type": "mcap", "uri": str(session.source_path)}
    ]

    modalities: dict[str, str] = {
        "human_hand": "not_applicable",
        "end_effector": "applicable",
    }

    stream_ranges: dict[str, tuple[int, int]] = {}
    for stream_id, vs in session.video_streams.items():
        if vs.timestamps_ns:
            stream_ranges[stream_id] = (int(min(vs.timestamps_ns)), int(max(vs.timestamps_ns)))

    delivery = build_robot_qc_delivery(
        session_id=session.session_id,
        profile="dunjia_ego",
        source_assets=source_assets,
        modalities=modalities,
        views=views,
        stream_ranges=stream_ranges,
        effective_config=cfg,
        producer="person-b.dunjia",
        version=producer_version,
    )
    manifest_path = Path(output_dir) / "revision.json"
    write_revision_manifest(manifest_path, delivery.manifest)
    return delivery


def run_a2d_formal_session(
    session: Any,
    episode_root: str | Path,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    producer_version: str = "v1",
) -> RobotQCDelivery:
    """Run Person B's A2D B6-B9 detectors then adapt to the formal schema."""
    from zpds_prepare.detectors.a2d import (
        check_a2d_alignment,
        check_a2d_completeness,
        check_a2d_robot_quality,
        aggregate_a2d_quality_views,
    )

    cfg = config or {}
    freeze_min_s = float(cfg.get("a2d", {}).get("robot", {}).get("freeze_min_duration_s", 2.0))
    gap_factor = float(cfg.get("a2d", {}).get("robot", {}).get("gap_factor", 3.0))

    # B6: completeness (takes path, not session)
    completeness = check_a2d_completeness(episode_root)

    # B7-B8: alignment + robot quality (take session)
    alignment = check_a2d_alignment(session)
    robot_quality = check_a2d_robot_quality(
        session, freeze_min_duration_s=freeze_min_s, gap_factor=gap_factor
    )

    # B9: aggregate views
    views_report = aggregate_a2d_quality_views(
        completeness=completeness,
        alignment=alignment,
        robot_quality=robot_quality,
        episode_id=session.session_id,
        source_path=str(episode_root),
    )

    config_hash = deterministic_config_hash(cfg)
    views = adapt_source_views(
        views_report,
        producer="person-b.a2d",
        version=producer_version,
        config_hash=config_hash,
    )

    source_assets: list[dict[str, Any]] = [
        {"asset_type": "episode_directory", "uri": str(episode_root)}
    ]

    modalities: dict[str, str] = {
        "human_hand": "not_applicable",
        "end_effector": "applicable",
    }

    stream_ranges: dict[str, tuple[int, int]] = {}
    for stream_id, vs in session.video_streams.items():
        if vs.timestamps_ns:
            stream_ranges[stream_id] = (int(min(vs.timestamps_ns)), int(max(vs.timestamps_ns)))

    delivery = build_robot_qc_delivery(
        session_id=session.session_id,
        profile="a2d_robot",
        source_assets=source_assets,
        modalities=modalities,
        views=views,
        stream_ranges=stream_ranges,
        effective_config=cfg,
        producer="person-b.a2d",
        version=producer_version,
    )
    manifest_path = Path(output_dir) / "revision.json"
    write_revision_manifest(manifest_path, delivery.manifest)
    return delivery


__all__ = [
    "FormalRobotQualityAdapter", "RobotQCDelivery", "adapt_source_views",
    "build_robot_qc_delivery", "run_a2d_formal_session",
    "run_dunjia_formal_session", "run_umi_formal_session",
    "source_views_to_decisions",
]
