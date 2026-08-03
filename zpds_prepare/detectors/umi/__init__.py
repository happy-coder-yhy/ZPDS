"""UMI-specific, contract-independent quality detectors."""

from zpds_prepare.detectors.umi.bimanual_sync import build_dual_alignment
from zpds_prepare.detectors.umi.evidence_writer import write_umi_evidence_bundle
from zpds_prepare.detectors.umi.gold_tooling import (
    UmiEpisodeCandidate,
    UmiGoldAnnotation,
    UmiLabeledSpan,
    binary_classification_metrics,
    compare_independent_reviews,
    evaluate_threshold_candidates,
    stratified_sample_episodes,
)
from zpds_prepare.detectors.umi.hand_applicability import (
    UmiHandApplicabilityResult,
    evaluate_umi_hand_applicability,
)
from zpds_prepare.detectors.umi.magnetic_encoder_quality import (
    analyze_magnetic_encoder,
)
from zpds_prepare.detectors.umi.orchestrator import (
    UmiEvidenceBundle,
    analyze_umi_session,
)
from zpds_prepare.detectors.umi.provisional_contract import (
    UmiEvidenceIndex,
    UmiProvisionalMetric,
    deterministic_config_hash,
)
from zpds_prepare.detectors.umi.provisional_pipeline import (
    UmiProvisionalRunResult,
    run_umi_provisional_dataset,
    run_umi_provisional_session,
)
from zpds_prepare.detectors.umi.revision_candidate import (
    build_umi_revision_candidate,
)
from zpds_prepare.detectors.umi.shared_adapter import (
    AdaptedUmiDelivery,
    UmiSharedContractAdapter,
    adapt_umi_delivery,
)
from zpds_prepare.detectors.umi.source_contract import (
    UmiSourceContractCandidate,
    build_umi_source_contract_candidate,
)
from zpds_prepare.detectors.umi.stream_timeline import analyze_stream_timeline
from zpds_prepare.detectors.umi.view_evaluator import (
    UmiCandidateView,
    evaluate_candidate_views,
)
from zpds_prepare.detectors.umi.vio_quality import analyze_vio_quality

__all__ = [
    "AdaptedUmiDelivery",
    "UmiCandidateView",
    "UmiEpisodeCandidate",
    "UmiEvidenceBundle",
    "UmiEvidenceIndex",
    "UmiGoldAnnotation",
    "UmiHandApplicabilityResult",
    "UmiLabeledSpan",
    "UmiProvisionalMetric",
    "UmiProvisionalRunResult",
    "UmiSharedContractAdapter",
    "UmiSourceContractCandidate",
    "adapt_umi_delivery",
    "analyze_magnetic_encoder",
    "analyze_stream_timeline",
    "analyze_umi_session",
    "analyze_vio_quality",
    "binary_classification_metrics",
    "build_dual_alignment",
    "build_umi_revision_candidate",
    "build_umi_source_contract_candidate",
    "compare_independent_reviews",
    "deterministic_config_hash",
    "evaluate_candidate_views",
    "evaluate_threshold_candidates",
    "evaluate_umi_hand_applicability",
    "run_umi_provisional_dataset",
    "run_umi_provisional_session",
    "stratified_sample_episodes",
    "write_umi_evidence_bundle",
]
