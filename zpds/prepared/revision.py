"""Formal revision manifest management for QC deliveries."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from zpds.core.quality import QualityMetric, QualityView

_REVISION_RE = re.compile(r"^r(\d{4,})$")
REVISION_SCHEMA_V1 = "zpds.revision_manifest.v1"
REVISION_SCHEMA_V2 = "zpds.revision_manifest.v2"
SUPPORTED_REVISION_SCHEMAS = frozenset({REVISION_SCHEMA_V1, REVISION_SCHEMA_V2})


def deterministic_config_hash(config: dict[str, Any]) -> str:
    """Produce the same SHA-256 for semantically identical configuration maps."""
    payload = json.dumps(
        config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass
class RevisionManifest:
    """Small formal manifest shared by robot-source QC integrations."""

    revision_id: str
    source_session_id: str
    profile: str
    source_assets: list[dict[str, Any]]
    modalities: dict[str, str]
    quality_views: dict[str, QualityView]
    metrics: list[QualityMetric] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    physical_spans: list[dict[str, Any]] = field(default_factory=list)
    idle_candidates: list[dict[str, Any]] = field(default_factory=list)
    evidence_index: dict[str, str] = field(default_factory=dict)
    outcome: dict[str, str] = field(
        default_factory=lambda: {"value": "unknown", "status": "not_run"}
    )
    producer: str = "zpds.robot_qc"
    version: str = "v1"
    config_hash: str = ""
    schema_version: str = REVISION_SCHEMA_V2
    raw_mutation: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "quality_views": {
                name: {
                    **asdict(view),
                    "disposition": view.disposition.value,
                }
                for name, view in self.quality_views.items()
            },
            "metrics": [
                {
                    **asdict(metric),
                    "metric_name": metric.metric_name,
                    "severity": metric.severity.value,
                    "disposition": metric.disposition.value,
                }
                for metric in self.metrics
            ],
        }


def write_revision_manifest(path: str | Path, manifest: RevisionManifest) -> Path:
    """Atomically write a formal manifest without touching source assets."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(f"{target.suffix}.tmp")
    temporary.write_text(json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_revision_manifest(path: str | Path) -> dict[str, Any]:
    """Read and validate required formal-manifest fields for round-trip use."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    required = {"schema_version", "revision_id", "source_session_id", "profile", "source_assets",
                "modalities", "quality_views", "metrics", "config_hash", "raw_mutation"}
    missing = sorted(required - document.keys())
    if missing or document.get("schema_version") not in SUPPORTED_REVISION_SCHEMAS:
        raise ValueError(f"invalid revision manifest, missing={missing}")
    if document["raw_mutation"] is not False:
        raise ValueError("revision manifest must declare raw_mutation=false")
    return document


class RevisionManager:
    """修订版本管理器。"""

    def __init__(self, root: str | Path = ".") -> None:
        self.root = Path(root)

    def create(self, notes: str = "") -> str:
        """创建新 revision，返回 revision_id（如 r0002）。"""
        latest = self.latest()
        number = int(latest[1:]) + 1 if latest else 1
        revision_id = f"r{number:04d}"
        directory = self.root / revision_id
        directory.mkdir(parents=True, exist_ok=False)
        if notes:
            (directory / "notes.txt").write_text(notes + "\n", encoding="utf-8")
        return revision_id

    def latest(self) -> str:
        """获取最新 revision_id。"""
        if not self.root.is_dir():
            return ""
        revisions = [entry.name for entry in self.root.iterdir() if entry.is_dir() and _REVISION_RE.match(entry.name)]
        return max(revisions, default="", key=lambda item: int(item[1:]))


__all__ = [
    "REVISION_SCHEMA_V1",
    "REVISION_SCHEMA_V2",
    "SUPPORTED_REVISION_SCHEMAS",
    "RevisionManager",
    "RevisionManifest",
    "deterministic_config_hash",
    "read_revision_manifest",
    "write_revision_manifest",
]
