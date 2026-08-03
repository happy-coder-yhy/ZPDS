"""UMI source-contract candidate pending the shared Profile schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass

SOURCE_CONTRACT_VERSION = "umi-source-candidate-v1"


@dataclass(frozen=True)
class UmiModalityCandidate:
    applicability: str
    reason: str


@dataclass(frozen=True)
class UmiSourceContractCandidate:
    """Non-formal source declaration used to guard provisional UMI runs."""

    profile_name: str
    modalities: dict[str, UmiModalityCandidate]
    candidate_quality_views: tuple[str, ...]
    human_hand_model_action: str
    forbidden_hand_reason_codes: tuple[str, ...]
    contract_version: str = SOURCE_CONTRACT_VERSION
    formal: bool = False

    def __post_init__(self) -> None:
        human_hand = self.modalities.get("human_hand")
        if human_hand is None or human_hand.applicability != "not_applicable":
            raise ValueError("UMI human_hand must be declared not_applicable")
        end_effector = self.modalities.get("end_effector")
        if end_effector is None or end_effector.applicability != "applicable":
            raise ValueError("UMI end_effector must be declared applicable")
        if self.human_hand_model_action != "skip":
            raise ValueError("UMI provisional runs must skip the human-hand model")

    def to_dict(self) -> dict:
        return asdict(self)


def build_umi_source_contract_candidate() -> UmiSourceContractCandidate:
    return UmiSourceContractCandidate(
        profile_name="jianzhi_umi",
        modalities={
            "human_hand": UmiModalityCandidate(
                applicability="not_applicable",
                reason="robot_end_effector_observation",
            ),
            "end_effector": UmiModalityCandidate(
                applicability="applicable",
                reason="dual_robot_gripper_observation",
            ),
        },
        candidate_quality_views=(
            "robot_observation_ready_candidate",
            "vio_ready_candidate",
            "bimanual_umi_ready_candidate",
        ),
        human_hand_model_action="skip",
        forbidden_hand_reason_codes=("HAND_ABSENT",),
    )


__all__ = [
    "SOURCE_CONTRACT_VERSION",
    "UmiModalityCandidate",
    "UmiSourceContractCandidate",
    "build_umi_source_contract_candidate",
]
