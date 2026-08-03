"""Persist provisional UMI evidence without writing the formal segment manifest."""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from zpds_prepare.detectors.umi.orchestrator import UmiEvidenceBundle
from zpds_prepare.detectors.umi.provisional_contract import (
    UmiEvidenceIndex,
    UmiProvisionalMetric,
    deterministic_config_hash,
)

EVIDENCE_DIRECTORY = "umi_provisional_evidence"


def _safe_name(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return safe or "unnamed"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
            default=_json_default,
        )
        + "\n",
        encoding="utf-8",
    )


def _measurement_metric(
    *,
    name: str,
    value: Any,
    unit: str,
    stream_id: str,
    session_id: str,
    evidence_uri: str,
    producer: str,
    version: str,
    config_hash: str,
) -> UmiProvisionalMetric:
    return UmiProvisionalMetric(
        metric_name=name,
        value=value,
        unit=unit,
        applicability="applicable",
        severity="info",
        disposition="keep",
        reason_code="measurement_only_uncalibrated",
        start_ns=None,
        end_ns=None,
        evidence_uri=evidence_uri,
        producer=producer,
        version=version,
        config_hash=config_hash,
        stream_id=stream_id,
        source_session_id=session_id,
        details={"automatic_reject": False, "formal_contract": False},
    )


def write_umi_evidence_bundle(
    bundle: UmiEvidenceBundle,
    output_dir: str | Path,
    *,
    producer: str = "zpds_prepare.detectors.umi",
    version: str = "dev",
    effective_config: dict[str, Any] | None = None,
) -> UmiEvidenceIndex:
    """Write versioned evidence and return a provisional, non-formal index."""
    config = effective_config or {}
    config_hash = deterministic_config_hash(config)
    output_root = Path(output_dir)
    evidence_root = output_root / EVIDENCE_DIRECTORY
    evidence_root.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    stream_evidence_uri: dict[str, str] = {}

    def write_frame(key: str, relative: Path, frame: Any) -> str:
        path = evidence_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(path, index=False)
        uri = path.relative_to(output_root).as_posix()
        artifacts[key] = uri
        return uri

    for category, streams in bundle.timeline_evidence.items():
        for stream_id, frame in streams.items():
            uri = write_frame(
                f"timeline:{category}:{stream_id}",
                Path("timelines") / category / f"{_safe_name(stream_id)}.parquet",
                frame,
            )
            stream_evidence_uri.setdefault(stream_id, uri)

    for stream_id, frame in bundle.vio_evidence.items():
        uri = write_frame(
            f"vio:{stream_id}",
            Path("vio") / f"{_safe_name(stream_id)}.parquet",
            frame,
        )
        stream_evidence_uri[stream_id] = uri

    for stream_id, frame in bundle.magnetic_encoder_evidence.items():
        uri = write_frame(
            f"magnetic_encoder:{stream_id}",
            Path("magnetic_encoder") / f"{_safe_name(stream_id)}.parquet",
            frame,
        )
        stream_evidence_uri[stream_id] = uri

    alignment_uris: dict[str, str] = {}
    for pair_key, frame in bundle.dual_alignments.items():
        alignment_uris[pair_key] = write_frame(
            f"dual_alignment:{pair_key}",
            Path("dual_alignment") / f"{_safe_name(pair_key)}.parquet",
            frame,
        )

    summaries = {
        "timeline": bundle.timeline_summaries,
        "vio": bundle.vio_summaries,
        "magnetic_encoder": bundle.magnetic_encoder_summaries,
        "dual_alignment": bundle.dual_alignment_summaries,
    }
    summaries_path = evidence_root / "summaries.json"
    _write_json(summaries_path, summaries)
    artifacts["summaries"] = summaries_path.relative_to(output_root).as_posix()

    issues_path = evidence_root / "issues.json"
    _write_json(issues_path, [issue.to_dict() for issue in bundle.issues])
    issues_uri = issues_path.relative_to(output_root).as_posix()
    artifacts["issues"] = issues_uri

    metrics: list[UmiProvisionalMetric] = [
        UmiProvisionalMetric.from_quality_issue(
            issue,
            evidence_uri=stream_evidence_uri.get(issue.stream_id, issues_uri),
            producer=producer,
            version=version,
            config_hash=config_hash,
            source_session_id=bundle.session_id,
        )
        for issue in bundle.issues
    ]

    for pair_key, summary in bundle.dual_alignment_summaries.items():
        uri = alignment_uris[pair_key]
        for field_name, unit in (
            ("mapped_ratio", "ratio"),
            ("residual_p50_ns", "ns"),
            ("residual_p95_ns", "ns"),
            ("residual_max_ns", "ns"),
        ):
            value = summary.get(field_name)
            if value is not None:
                metrics.append(
                    _measurement_metric(
                        name=f"umi_alignment.{pair_key}.{field_name}",
                        value=value,
                        unit=unit,
                        stream_id=pair_key,
                        session_id=bundle.session_id,
                        evidence_uri=uri,
                        producer=producer,
                        version=version,
                        config_hash=config_hash,
                    )
                )

    for stream_id, summary in bundle.vio_summaries.items():
        for field_name in (
            "non_finite_pose_count",
            "invalid_quaternion_count",
            "translation_step_candidate_count",
            "header_topic_mismatch_count",
        ):
            metrics.append(
                _measurement_metric(
                    name=f"umi_vio.{field_name}",
                    value=summary[field_name],
                    unit="count",
                    stream_id=stream_id,
                    session_id=bundle.session_id,
                    evidence_uri=stream_evidence_uri[stream_id],
                    producer=producer,
                    version=version,
                    config_hash=config_hash,
                )
            )

    for stream_id, summary in bundle.magnetic_encoder_summaries.items():
        for field_name, unit in (
            ("finite_ratio", "ratio"),
            ("freeze_span_count", "count"),
            ("range_candidate_count", "count"),
        ):
            metrics.append(
                _measurement_metric(
                    name=f"umi_magnetic_encoder.{field_name}",
                    value=summary[field_name],
                    unit=unit,
                    stream_id=stream_id,
                    session_id=bundle.session_id,
                    evidence_uri=stream_evidence_uri[stream_id],
                    producer=producer,
                    version=version,
                    config_hash=config_hash,
                )
            )

    metrics_path = evidence_root / "metrics.json"
    _write_json(metrics_path, [metric.to_dict() for metric in metrics])
    artifacts["metrics"] = metrics_path.relative_to(output_root).as_posix()

    index = UmiEvidenceIndex(
        source_session_id=bundle.session_id,
        producer=producer,
        version=version,
        config_hash=config_hash,
        artifacts=dict(artifacts),
        metrics=tuple(metrics),
    )
    index_path = evidence_root / "evidence_index.json"
    _write_json(index_path, index.to_dict())
    return index


__all__ = ["EVIDENCE_DIRECTORY", "write_umi_evidence_bundle"]
