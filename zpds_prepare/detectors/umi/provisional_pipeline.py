"""Standalone UMI provisional pipeline, isolated from the shared main entry."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
from typing import Any

from zpds_prepare.detectors.umi.evidence_writer import (
    EVIDENCE_DIRECTORY,
    write_umi_evidence_bundle,
)
from zpds_prepare.detectors.umi.hand_applicability import (
    UmiHandApplicabilityResult,
    evaluate_umi_hand_applicability,
)
from zpds_prepare.detectors.umi.orchestrator import (
    UmiEvidenceBundle,
    analyze_umi_session,
)
from zpds_prepare.detectors.umi.provisional_contract import UmiEvidenceIndex
from zpds_prepare.detectors.umi.revision_candidate import (
    build_umi_revision_candidate,
)
from zpds_prepare.detectors.umi.source_contract import (
    UmiSourceContractCandidate,
    build_umi_source_contract_candidate,
)
from zpds_prepare.detectors.umi.view_evaluator import (
    UmiCandidateView,
    evaluate_candidate_views,
)
from zpds_prepare.readers.session_model import Session

DEFAULT_PROVISIONAL_CONFIG: dict[str, Any] = {
    "minimum_gap_ns": 500_000_000,
    "alignment_max_residual_ns": None,
    "alignment_mapping_method": "inferred",
    "encoder_freeze_min_samples": 10,
    "require_vio_for_bimanual": True,
}


@dataclass(frozen=True)
class UmiProvisionalRunResult:
    session_id: str
    source_contract: UmiSourceContractCandidate
    hand_applicability: UmiHandApplicabilityResult
    bundle: UmiEvidenceBundle
    evidence_index: UmiEvidenceIndex
    candidate_views: dict[str, UmiCandidateView]
    revision_candidate: dict[str, Any]
    effective_config: dict[str, Any]
    source_sha256_before: str | None
    source_sha256_after: str | None
    formal_manifest_written: bool = False
    human_hand_model_invoked: bool = False


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def run_umi_provisional_session(
    session: Session,
    output_dir: str | Path,
    *,
    config: dict[str, Any] | None = None,
    producer_version: str = "dev",
) -> UmiProvisionalRunResult:
    """Run UMI analysis and write only provisional evidence artifacts."""
    effective_config = {**DEFAULT_PROVISIONAL_CONFIG, **(config or {})}
    source_contract = build_umi_source_contract_candidate()
    hand_applicability = evaluate_umi_hand_applicability(source_contract)
    if hand_applicability.run_human_hand_model:
        raise RuntimeError("UMI hand applicability guard must skip inference")

    source_path = Path(session.source_path)
    source_hash_before = _file_sha256(source_path)
    bundle = analyze_umi_session(
        session,
        minimum_gap_ns=int(effective_config["minimum_gap_ns"]),
        alignment_max_residual_ns=effective_config[
            "alignment_max_residual_ns"
        ],
        alignment_mapping_method=str(
            effective_config["alignment_mapping_method"]
        ),
        encoder_freeze_min_samples=int(
            effective_config["encoder_freeze_min_samples"]
        ),
    )
    candidate_views = evaluate_candidate_views(
        bundle,
        require_vio_for_bimanual=bool(
            effective_config["require_vio_for_bimanual"]
        ),
    )
    output_root = Path(output_dir)
    index = write_umi_evidence_bundle(
        bundle,
        output_root,
        version=producer_version,
        effective_config=effective_config,
    )

    evidence_root = output_root / EVIDENCE_DIRECTORY
    contract_path = evidence_root / "source_contract_candidate.json"
    hand_path = evidence_root / "stage9_hand_applicability_candidate.json"
    views_path = evidence_root / "candidate_views.json"
    _write_json(contract_path, source_contract.to_dict())
    _write_json(hand_path, hand_applicability.to_dict())
    _write_json(
        views_path,
        {name: view.to_dict() for name, view in candidate_views.items()},
    )
    extra_artifacts = {
        "source_contract_candidate": contract_path.relative_to(
            output_root
        ).as_posix(),
        "stage9_hand_applicability_candidate": hand_path.relative_to(
            output_root
        ).as_posix(),
        "candidate_views": views_path.relative_to(output_root).as_posix(),
    }
    index = replace(index, artifacts={**index.artifacts, **extra_artifacts})
    index_path = evidence_root / "evidence_index.json"
    _write_json(index_path, index.to_dict())

    revision_candidate = build_umi_revision_candidate(
        source_session_id=session.session_id,
        source_path=session.source_path,
        source_sha256=source_hash_before,
        source_contract=source_contract,
        hand_applicability=hand_applicability,
        evidence_index=index,
        candidate_views=candidate_views,
        effective_config=effective_config,
    )
    revision_path = evidence_root / "umi_revision_candidate.json"
    _write_json(revision_path, revision_candidate)
    index = replace(
        index,
        artifacts={
            **index.artifacts,
            "umi_revision_candidate": revision_path.relative_to(
                output_root
            ).as_posix(),
        },
    )
    _write_json(index_path, index.to_dict())

    source_hash_after = _file_sha256(source_path)
    if source_hash_before != source_hash_after:
        raise RuntimeError("Raw UMI source changed during provisional analysis")

    run_summary = {
        "session_id": session.session_id,
        "formal_manifest_written": False,
        "human_hand_model_invoked": False,
        "source_sha256_before": source_hash_before,
        "source_sha256_after": source_hash_after,
        "evidence_index_uri": index_path.relative_to(output_root).as_posix(),
        "candidate_view_status": {
            name: view.status for name, view in candidate_views.items()
        },
    }
    summary_path = evidence_root / "provisional_run_summary.json"
    _write_json(summary_path, run_summary)
    index = replace(
        index,
        artifacts={
            **index.artifacts,
            "provisional_run_summary": summary_path.relative_to(
                output_root
            ).as_posix(),
        },
    )
    _write_json(index_path, index.to_dict())

    return UmiProvisionalRunResult(
        session_id=session.session_id,
        source_contract=source_contract,
        hand_applicability=hand_applicability,
        bundle=bundle,
        evidence_index=index,
        candidate_views=candidate_views,
        revision_candidate=revision_candidate,
        effective_config=effective_config,
        source_sha256_before=source_hash_before,
        source_sha256_after=source_hash_after,
    )


def run_umi_provisional_dataset(
    dataset_path: str | Path,
    output_dir: str | Path,
    *,
    cache_dir: str | Path,
    config: dict[str, Any] | None = None,
    producer_version: str = "dev",
) -> UmiProvisionalRunResult:
    """Read one UMI MCAP and run the isolated provisional pipeline."""
    from zpds_prepare.readers.umi_reader import read_session

    session = read_session(str(dataset_path), cache_dir=Path(cache_dir))
    return run_umi_provisional_session(
        session,
        output_dir,
        config=config,
        producer_version=producer_version,
    )


__all__ = [
    "DEFAULT_PROVISIONAL_CONFIG",
    "UmiProvisionalRunResult",
    "run_umi_provisional_dataset",
    "run_umi_provisional_session",
]
