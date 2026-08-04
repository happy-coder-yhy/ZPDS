"""Stage 0: 隐私脱敏门 — 脱敏结果 → QC Decision + QualityView。

挂在级联最前端。LLM 不可用时产生 ERROR 决策并阻断产出；
覆盖不足等软阈值只出 WARN/quarantine，不自动 reject。
"""

from __future__ import annotations

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.core.quality import QualityView
from zpds.qc.cascade import register_stage


def check(
    *,
    manifest: dict | None = None,
    stage_config: dict | None = None,
) -> list[Decision]:
    """输入 PrivacyRunManifest（dict 形式），输出 QC Decision 列表。

    Args:
        manifest: PrivacyRunManifest.to_dict() 风格字典，含 stats、llm_available 等。
        stage_config: 级联配置（来自 YAML，可用于开关）。

    Returns:
        Decision 列表。
    """
    cfg = stage_config or {}
    if not cfg.get("enabled", True):
        return []

    decisions: list[Decision] = []

    if manifest is None:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_COVERAGE_LOW,
            severity=Severity.WARN,
            message="Privacy manifest 缺失，无法判断脱敏覆盖率",
            disposition=Disposition.QUARANTINE,
        ))
        return decisions

    stats = manifest.get("stats", manifest)
    llm_available = bool(manifest.get("llm_available", False))
    total_frames = int(stats.get("total_frames", 0))

    # ---- LLM 不可用 → ERROR ----
    if not llm_available:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_LLM_UNAVAILABLE,
            severity=Severity.ERROR,
            message="LLM PII 分类后端不可用，脱敏流程失败，不产出脱敏产物",
            disposition=Disposition.REJECT,
        ))
        return decisions

    # ---- 人脸脱敏记录 ----
    face_frames = int(stats.get("frames_with_faces", 0))
    face_regions = int(stats.get("total_face_regions", 0))
    if face_regions > 0:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_FACE_MASKED,
            severity=Severity.INFO,
            message=f"人脸已脱敏: {face_regions} 区域, {face_frames} 帧",
            disposition=Disposition.KEEP,
            detail={
                "face_regions": face_regions,
                "face_frames": face_frames,
            },
        ))

    # ---- PII 文本脱敏记录 ----
    pii_masked = int(stats.get("total_pii_masked", 0))
    text_regions = int(stats.get("total_text_regions", 0))
    pii_categories = list(stats.get("pii_categories_found", []))
    if pii_masked > 0:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_PII_MASKED,
            severity=Severity.INFO,
            message=f"PII 文本已脱敏: {pii_masked} 区域, 类别={pii_categories}",
            disposition=Disposition.KEEP,
            detail={
                "pii_masked": pii_masked,
                "text_regions": text_regions,
                "pii_categories": pii_categories,
            },
        ))

    # ---- 覆盖率异常检测 ----
    face_applicable = cfg.get("face", {}).get("applicability", "applicable")
    text_applicable = cfg.get("text", {}).get("applicability", "applicable")

    if face_applicable == "applicable" and total_frames > 0 and face_frames == 0:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_COVERAGE_LOW,
            severity=Severity.WARN,
            message=f"人脸脱敏覆盖异常: {total_frames} 帧中 0 帧检测到人脸（可能模型/配置问题）",
            disposition=Disposition.QUARANTINE,
            detail={"total_frames": total_frames, "face_frames": 0},
        ))

    if text_applicable == "applicable" and total_frames > 0 and text_regions == 0:
        decisions.append(Decision(
            stage=0,
            reason=ReasonCode.PRIVACY_COVERAGE_LOW,
            severity=Severity.WARN,
            message=f"文本脱敏覆盖异常: {total_frames} 帧中 0 个文本区域（可能无文本或检测器失效）",
            disposition=Disposition.KEEP_WITH_FLAG,
            detail={"total_frames": total_frames, "text_regions": 0},
        ))

    return decisions


def build_privacy_view(decisions: list[Decision]) -> QualityView:
    """从 Stage 0 decisions 构建 privacy_ready QualityView。"""
    has_error = any(
        d.severity in (Severity.FATAL, Severity.ERROR) for d in decisions
    )
    has_warn = any(d.severity == Severity.WARN for d in decisions)
    reasons = [d.reason.value for d in decisions]

    if has_error:
        return QualityView(
            name="privacy_ready",
            ready=False,
            applicability="applicable",
            disposition=Disposition.REJECT,
            reasons=tuple(reasons),
            dependencies=(),
            evidence_uris=(),
        )

    if has_warn:
        return QualityView(
            name="privacy_ready",
            ready=True,
            applicability="applicable",
            disposition=Disposition.KEEP_WITH_FLAG,
            reasons=tuple(reasons),
            dependencies=(),
            evidence_uris=(),
        )

    return QualityView(
        name="privacy_ready",
        ready=True,
        applicability="applicable",
        disposition=Disposition.KEEP,
        reasons=tuple(reasons),
        dependencies=(),
        evidence_uris=(),
    )


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(0)
def _check_stage0(context: dict) -> list[Decision]:
    """Stage 0 QCCascade 入口：从 context dict 提取 manifest 并检查。"""
    manifest = context.get("privacy_manifest")
    stage_config = context.get("stage_config", {})
    return check(manifest=manifest, stage_config=stage_config)


__all__ = [
    "build_privacy_view",
    "check",
]
