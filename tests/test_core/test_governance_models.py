import pytest

from zpds.core.decisions import (
    Decision,
    DecisionType,
    Evidence,
    ReasonCode,
    Severity,
)
from zpds.core.provenance import OriginKind, ValueOrigin
from zpds.core.quality import MetricDirection, QualityMetric, QualityReport
from zpds.core.types import BoundaryKind, SpanProposal, TimeRange


def test_time_range_uses_half_open_interval() -> None:
    time_range = TimeRange(start_ns=10, end_ns=25)

    assert time_range.duration_ns == 15
    with pytest.raises(ValueError, match="greater"):
        TimeRange(start_ns=10, end_ns=10)


def test_span_proposal_distinguishes_physical_and_semantic_boundaries() -> None:
    physical = SpanProposal(0, 10, boundary_kind=BoundaryKind.PHYSICAL)
    scene = SpanProposal(0, 10, boundary_kind=BoundaryKind.SCENE)

    assert physical.boundary_kind is BoundaryKind.PHYSICAL
    assert scene.boundary_kind is BoundaryKind.SCENE


def test_evidence_and_decision_reject_invalid_ranges() -> None:
    with pytest.raises(ValueError, match="start_ns is required"):
        Evidence(uri="report://one", kind="log", end_ns=10)
    with pytest.raises(ValueError, match="span_start_ns is required"):
        Decision(
            stage=2,
            reason=ReasonCode.TIMESTAMP_GAP,
            severity=Severity.ERROR,
            message="gap at end of stream",
            span_end_ns=10,
            evidence=[Evidence(uri="report://one", kind="timeline")],
        )


def test_quality_metric_supports_all_threshold_directions() -> None:
    high = QualityMetric("coverage", 0.9, threshold=0.8)
    low = QualityMetric(
        "error_ns",
        5,
        unit="ns",
        direction=MetricDirection.LOWER_IS_BETTER,
        threshold=10,
    )
    bounded = QualityMetric(
        "rate",
        30,
        unit="Hz",
        direction=MetricDirection.IN_RANGE,
        threshold=None,
        lower_bound=29,
        upper_bound=31,
    )

    assert high.pass_
    assert low.pass_
    assert bounded.pass_


def test_quality_report_is_recomputed_after_decision_changes() -> None:
    report = QualityReport(
        session_id="session",
        metrics=[QualityMetric("coverage", 0.9, threshold=0.8)],
    )
    assert report.overall_pass

    report.decisions.append(
        Decision(
            stage=2,
            reason=ReasonCode.CLOCK_RESET,
            severity=Severity.FATAL,
            decision=DecisionType.QUARANTINE,
            message="device clock reset",
            evidence=[Evidence(uri="report://clock/reset", kind="timeline")],
        )
    )
    assert not report.overall_pass


def test_non_keep_decision_requires_explanation_and_evidence() -> None:
    with pytest.raises(ValueError, match="message is required"):
        Decision(
            stage=2,
            reason=ReasonCode.CLOCK_RESET,
            severity=Severity.FATAL,
            decision=DecisionType.QUARANTINE,
        )


def test_model_origin_requires_model_identity_and_source() -> None:
    with pytest.raises(ValueError, match="source_refs"):
        ValueOrigin(
            kind=OriginKind.MODEL_ESTIMATED,
            producer_id="hand_pose",
            model_name="pose-model",
            model_version="1.0.0",
        )

    origin = ValueOrigin(
        kind=OriginKind.MODEL_ESTIMATED,
        producer_id="hand_pose",
        source_refs=["stream://segment/rgb"],
        model_name="pose-model",
        model_version="1.0.0",
    )
    assert origin.kind is OriginKind.MODEL_ESTIMATED
