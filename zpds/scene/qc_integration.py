"""人员 B：SceneProposal 与 VLM 复核结果转成 QC 指标与决策。"""

from __future__ import annotations

from collections.abc import Sequence

from zpds.core.decisions import (
    Decision,
    Disposition,
    ReasonCode,
    Severity,
)
from zpds.core.quality import QualityMetric
from zpds.scene.config import SceneConfig
from zpds.scene.schemas import SceneProposal, VLMReviewResult

QC_PRODUCER = "zpds.scene.qc"
QC_VERSION = "v1"


def build_scene_metrics(
    scenes: Sequence[SceneProposal],
    vlm_results: Sequence[VLMReviewResult],
    *,
    config_hash: str,
) -> list[QualityMetric]:
    """产生 scene_count 与边界/复核覆盖率指标。"""

    confidences = [scene.confidence for scene in scenes]
    consistent = sum(
        1 for result in vlm_results if result.decision == "consistent"
    )
    metrics: list[QualityMetric] = [
        QualityMetric(
            name="scene_count",
            value=len(scenes),
            threshold=None,
            comparison="none",
            unit="count",
            reason_code="measurement",
            producer=QC_PRODUCER,
            version=QC_VERSION,
            config_hash=config_hash,
        ),
        QualityMetric(
            name="boundary_confidence_min",
            value=min(confidences) if confidences else None,
            threshold=None,
            comparison="none",
            unit="confidence",
            reason_code="measurement",
            producer=QC_PRODUCER,
            version=QC_VERSION,
            config_hash=config_hash,
        ),
        QualityMetric(
            name="boundary_confidence_mean",
            value=(
                round(sum(confidences) / len(confidences), 6)
                if confidences
                else None
            ),
            threshold=None,
            comparison="none",
            unit="confidence",
            reason_code="measurement",
            producer=QC_PRODUCER,
            version=QC_VERSION,
            config_hash=config_hash,
        ),
        QualityMetric(
            name="vlm_consistent_ratio",
            value=round(consistent / len(vlm_results), 6)
            if vlm_results
            else None,
            threshold=None,
            comparison="none",
            unit="ratio",
            reason_code="measurement",
            producer=QC_PRODUCER,
            version=QC_VERSION,
            config_hash=config_hash,
        ),
        QualityMetric(
            name="vlm_reviewed_ratio",
            value=round(len(vlm_results) / len(scenes), 6)
            if scenes
            else None,
            threshold=None,
            comparison="none",
            unit="ratio",
            reason_code="measurement",
            producer=QC_PRODUCER,
            version=QC_VERSION,
            config_hash=config_hash,
        ),
    ]
    return metrics


def _decision_detail(
    scene: SceneProposal | None,
    *,
    extra: dict[str, object] | None = None,
) -> dict[str, object]:
    detail: dict[str, object] = {
        "producer": QC_PRODUCER,
        "version": QC_VERSION,
    }
    if scene is not None:
        detail.update(
            {
                "scene_id": scene.scene_id,
                "evidence_uris": list(scene.evidence_uris),
                "config_hash": scene.config_hash,
            }
        )
    if extra:
        detail.update(extra)
    return detail


def build_scene_decisions(
    scenes: Sequence[SceneProposal],
    vlm_results: Sequence[VLMReviewResult],
    *,
    config: SceneConfig,
    vlm_enabled: bool,
) -> list[Decision]:
    """VLM 不一致与低置信度边界转成可追溯决策。"""

    decisions: list[Decision] = []
    for scene in scenes:
        if scene.confidence < config.fusion.low_confidence_threshold:
            decisions.append(
                Decision(
                    stage=10,
                    reason=ReasonCode.SCENE_BOUNDARY_LOW_CONFIDENCE,
                    severity=Severity.WARN,
                    message=(
                        f"场景 {scene.scene_id} 边界置信度 "
                        f"{scene.confidence:.3f} 低于阈值 "
                        f"{config.fusion.low_confidence_threshold:.3f}，"
                        "建议人工复核"
                    ),
                    frame_idx=None,
                    timestamp_ns=scene.start_ns,
                    end_timestamp_ns=scene.end_ns,
                    disposition=Disposition.QUARANTINE,
                    detail=_decision_detail(
                        scene,
                        extra={"confidence": scene.confidence},
                    ),
                )
            )
    for result in vlm_results:
        if result.decision == "inconsistent":
            matched_scene = next(
                (
                    item
                    for item in scenes
                    if item.scene_id == result.scene_id
                ),
                None,
            )
            detail = _decision_detail(
                matched_scene,
                extra={
                    "vlm_decision": result.decision,
                    "vlm_confidence": result.confidence,
                    "vlm_reasons": result.reasons,
                },
            )
            if matched_scene is None:
                detail["scene_id"] = result.scene_id
                detail["config_hash"] = result.config_hash
            decisions.append(
                Decision(
                    stage=10,
                    reason=ReasonCode.SEMANTIC_INCONSISTENCY,
                    severity=Severity.WARN,
                    message=(
                        f"VLM 判定场景 {result.scene_id} 的 "
                        f"scene={result.scene_label!r} 与 "
                        f"task={result.task_label!r} 不一致: {result.reasons}"
                    ),
                    timestamp_ns=(
                        matched_scene.start_ns
                        if matched_scene is not None
                        else None
                    ),
                    disposition=Disposition.QUARANTINE,
                    detail=detail,
                )
            )
    if vlm_enabled and scenes and not vlm_results:
        decisions.append(
            Decision(
                stage=10,
                reason=ReasonCode.SEMANTIC_NOT_RUN,
                severity=Severity.INFO,
                message="VLM 复核已启用但未产出任何复核结果",
                disposition=Disposition.KEEP,
                detail={
                    "producer": QC_PRODUCER,
                    "version": QC_VERSION,
                    "config_hash": config.config_hash,
                },
            )
        )
    return decisions


__all__ = [
    "QC_PRODUCER",
    "QC_VERSION",
    "build_scene_decisions",
    "build_scene_metrics",
]
