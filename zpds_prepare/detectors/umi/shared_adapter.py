"""Adapter boundary that Person C can implement without changing UMI logic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar, runtime_checkable

from zpds_prepare.detectors.umi.provisional_contract import (
    UmiEvidenceIndex,
    UmiProvisionalMetric,
)
from zpds_prepare.detectors.umi.view_evaluator import UmiCandidateView

MetricT = TypeVar("MetricT")
ViewT = TypeVar("ViewT")
RevisionT = TypeVar("RevisionT")


@runtime_checkable
class UmiSharedContractAdapter(Protocol[MetricT, ViewT, RevisionT]):
    """Minimal interface required from the future shared QC contract."""

    def adapt_metric(self, metric: UmiProvisionalMetric) -> MetricT: ...

    def adapt_view(self, view: UmiCandidateView) -> ViewT: ...

    def adapt_revision(self, revision: dict[str, Any]) -> RevisionT: ...


@dataclass(frozen=True)
class AdaptedUmiDelivery(Generic[MetricT, ViewT, RevisionT]):
    metrics: tuple[MetricT, ...]
    views: dict[str, ViewT]
    revision: RevisionT


def adapt_umi_delivery(
    evidence_index: UmiEvidenceIndex,
    candidate_views: dict[str, UmiCandidateView],
    revision_candidate: dict[str, Any],
    adapter: UmiSharedContractAdapter[MetricT, ViewT, RevisionT],
) -> AdaptedUmiDelivery[MetricT, ViewT, RevisionT]:
    """Convert only at the boundary; detectors remain contract-independent."""
    return AdaptedUmiDelivery(
        metrics=tuple(
            adapter.adapt_metric(metric) for metric in evidence_index.metrics
        ),
        views={
            name: adapter.adapt_view(view)
            for name, view in candidate_views.items()
        },
        revision=adapter.adapt_revision(revision_candidate),
    )


__all__ = [
    "AdaptedUmiDelivery",
    "UmiSharedContractAdapter",
    "adapt_umi_delivery",
]
