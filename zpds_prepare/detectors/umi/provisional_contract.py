"""Replaceable UMI-only contract used before the shared QC schema is frozen."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue

CONTRACT_VERSION = "umi-provisional-v1"
APPLICABILITY_VALUES = {"applicable", "not_applicable", "unavailable"}
SEVERITY_VALUES = {"info", "warning", "error", "critical"}
DISPOSITION_VALUES = {
    "keep",
    "keep_with_flag",
    "quarantine",
    "trim",
    "split",
    "reject_view",
}


def deterministic_config_hash(config: dict[str, Any]) -> str:
    """Hash the effective provisional configuration deterministically."""
    payload = json.dumps(
        config,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


@dataclass(frozen=True)
class UmiProvisionalMetric:
    """Field superset that can later be adapted to the shared QC contract."""

    metric_name: str
    value: Any
    unit: str
    applicability: str
    severity: str
    disposition: str
    reason_code: str
    start_ns: int | None
    end_ns: int | None
    evidence_uri: str
    producer: str
    version: str
    config_hash: str
    stream_id: str
    source_session_id: str
    contract_version: str = CONTRACT_VERSION
    details: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.metric_name:
            raise ValueError("metric_name must not be empty")
        if self.applicability not in APPLICABILITY_VALUES:
            raise ValueError(f"invalid applicability: {self.applicability}")
        if self.severity not in SEVERITY_VALUES:
            raise ValueError(f"invalid severity: {self.severity}")
        if self.disposition not in DISPOSITION_VALUES:
            raise ValueError(f"invalid disposition: {self.disposition}")
        if (
            self.start_ns is not None
            and self.end_ns is not None
            and self.end_ns < self.start_ns
        ):
            raise ValueError("end_ns must not be earlier than start_ns")
        if not self.producer or not self.version:
            raise ValueError("producer and version must not be empty")
        if len(self.config_hash) != 64:
            raise ValueError("config_hash must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_quality_issue(
        cls,
        issue: QualityIssue,
        *,
        evidence_uri: str,
        producer: str,
        version: str,
        config_hash: str,
        source_session_id: str,
    ) -> UmiProvisionalMetric:
        """Preserve a detector issue without claiming a formal shared decision."""
        severity = issue.severity if issue.severity in SEVERITY_VALUES else "warning"
        disposition = (
            issue.decision
            if issue.decision in DISPOSITION_VALUES
            else "keep_with_flag"
        )
        return cls(
            metric_name=issue.issue_type,
            value=1,
            unit="event",
            applicability="applicable",
            severity=severity,
            disposition=disposition,
            reason_code=issue.issue_type,
            start_ns=issue.start_ns,
            end_ns=issue.end_ns,
            evidence_uri=evidence_uri,
            producer=producer,
            version=version,
            config_hash=config_hash,
            stream_id=issue.stream_id,
            source_session_id=source_session_id,
            details=dict(issue.details),
        )


@dataclass(frozen=True)
class UmiEvidenceIndex:
    """Index of provisional evidence; deliberately not a formal manifest."""

    source_session_id: str
    producer: str
    version: str
    config_hash: str
    artifacts: dict[str, str]
    metrics: tuple[UmiProvisionalMetric, ...]
    contract_version: str = CONTRACT_VERSION
    formal_manifest: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "formal_manifest": self.formal_manifest,
            "source_session_id": self.source_session_id,
            "producer": self.producer,
            "version": self.version,
            "config_hash": self.config_hash,
            "artifacts": dict(self.artifacts),
            "metrics": [metric.to_dict() for metric in self.metrics],
        }


__all__ = [
    "APPLICABILITY_VALUES",
    "CONTRACT_VERSION",
    "DISPOSITION_VALUES",
    "SEVERITY_VALUES",
    "UmiEvidenceIndex",
    "UmiProvisionalMetric",
    "deterministic_config_hash",
]
