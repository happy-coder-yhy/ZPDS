"""Pure Stage-9 guard for UMI, ready for the shared stage to call later."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from zpds_prepare.detectors.umi.source_contract import (
    UmiSourceContractCandidate,
)


@dataclass(frozen=True)
class UmiHandApplicabilityResult:
    applicability: str
    reason: str
    run_human_hand_model: bool
    severity: str
    disposition: str
    emitted_reason_codes: tuple[str, ...]
    forbidden_reason_codes: tuple[str, ...]
    formal: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


def evaluate_umi_hand_applicability(
    contract: UmiSourceContractCandidate,
) -> UmiHandApplicabilityResult:
    """Return a deterministic skip result before any human-hand inference."""
    human_hand = contract.modalities["human_hand"]
    if human_hand.applicability != "not_applicable":
        raise ValueError("UMI hand guard requires not_applicable applicability")
    if contract.human_hand_model_action != "skip":
        raise ValueError("UMI hand guard requires model action 'skip'")
    return UmiHandApplicabilityResult(
        applicability="not_applicable",
        reason=human_hand.reason,
        run_human_hand_model=False,
        severity="info",
        disposition="keep",
        emitted_reason_codes=(),
        forbidden_reason_codes=contract.forbidden_hand_reason_codes,
    )


__all__ = [
    "UmiHandApplicabilityResult",
    "evaluate_umi_hand_applicability",
]
