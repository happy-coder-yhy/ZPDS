"""Build a non-formal UMI revision payload for later manifest adaptation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from zpds_prepare.detectors.umi.hand_applicability import (
    UmiHandApplicabilityResult,
)
from zpds_prepare.detectors.umi.provisional_contract import UmiEvidenceIndex
from zpds_prepare.detectors.umi.source_contract import (
    UmiSourceContractCandidate,
)
from zpds_prepare.detectors.umi.view_evaluator import UmiCandidateView

REVISION_CANDIDATE_VERSION = "umi-revision-candidate-v1"


def build_umi_revision_candidate(
    *,
    source_session_id: str,
    source_path: str | Path,
    source_sha256: str | None,
    source_contract: UmiSourceContractCandidate,
    hand_applicability: UmiHandApplicabilityResult,
    evidence_index: UmiEvidenceIndex,
    candidate_views: dict[str, UmiCandidateView],
    effective_config: dict[str, Any],
) -> dict[str, Any]:
    """Compose the field superset; do not claim formal manifest validity."""
    return {
        "schema_version": REVISION_CANDIDATE_VERSION,
        "formal_manifest": False,
        "source_session_id": source_session_id,
        "profile": source_contract.profile_name,
        "source_assets": [
            {
                "source_asset_id": "raw_mcap",
                "uri": str(source_path),
                "sha256": source_sha256,
                "immutable": True,
            }
        ],
        "modalities": {
            name: {
                "applicability": value.applicability,
                "reason": value.reason,
            }
            for name, value in source_contract.modalities.items()
        },
        "stage9_hand_applicability": hand_applicability.to_dict(),
        "quality_views": {
            name: view.to_dict() for name, view in candidate_views.items()
        },
        "metric_count": len(evidence_index.metrics),
        "metrics": [metric.to_dict() for metric in evidence_index.metrics],
        "evidence_artifacts": dict(evidence_index.artifacts),
        "producer": evidence_index.producer,
        "producer_version": evidence_index.version,
        "config_hash": evidence_index.config_hash,
        "effective_config": dict(effective_config),
        "outcome": {
            "value": "unknown",
            "applicability": "unavailable",
            "reason": "semantic_review_not_run_by_person_a",
        },
        "raw_mutation": False,
        "automatic_reject": False,
    }


__all__ = [
    "REVISION_CANDIDATE_VERSION",
    "build_umi_revision_candidate",
]
