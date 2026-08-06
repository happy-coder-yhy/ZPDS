"""Stage 7 附属：无手来源的机器人观测质量视图。

``human_hand=applicable`` 的来源（墨现等）由 Stage 9 手部链路负责，
本评估跳过；``human_hand=not_applicable`` 的来源（遁甲/UMI/A2D）
在这里评估各自的机器人观测质量视图：

- 遁甲（B1–B5）：``robot_observation_ready`` / ``end_effector_visible``
- A2D（B6–B9）：``robot_observation_ready`` / ``robot_bc_ready`` /
  ``geometry_ready`` / ``failure_recovery``
- UMI：``robot_observation_ready_candidate`` 等 candidate views

注册入口在 :mod:`zpds.qc.stage7_robot` 的统一检查器中。
"""

from __future__ import annotations

from pathlib import Path

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity

_PROFILE_REGISTRY_NAMES = {
    "guida": "guida_ego",
    "dunjia": "dunjia_ego",
    "umi": "jianzhi_umi",
    "epic": "epic100",
    "a2d": "a2d_robot",
}


def _profile_registry_name(profile: str) -> str:
    return _PROFILE_REGISTRY_NAMES.get(profile, profile)


def _extract_view_state(view) -> tuple[bool, str, list[str]]:
    """从 dunjia/a2d QualityView 或 UMI UmiCandidateView 提取统一状态。"""
    if hasattr(view, "status"):
        # UMI candidate view：candidate_pass / review_required / unavailable
        ready = view.status == "candidate_pass"
        disposition = str(getattr(view, "disposition", "keep"))
        reasons = list(getattr(view, "reasons", ()))
        return ready, disposition, reasons
    ready = bool(getattr(view, "ready", False))
    disposition = str(getattr(view, "disposition", "pass"))
    reasons = list(getattr(view, "reasons", ()))
    return ready, disposition, reasons


def _reason_pair(name: str) -> tuple[ReasonCode, ReasonCode]:
    if name.startswith("robot_observation_ready"):
        return (
            ReasonCode.ROBOT_OBSERVATION_READY,
            ReasonCode.ROBOT_OBSERVATION_NOT_READY,
        )
    if name == "end_effector_visible":
        return (
            ReasonCode.END_EFFECTOR_VISIBLE,
            ReasonCode.END_EFFECTOR_NOT_VISIBLE,
        )
    return ReasonCode.ROBOT_VIEW_PASS, ReasonCode.ROBOT_VIEW_FAIL


def views_to_decisions(
    views_report,
    *,
    session_id: str,
    config_hash: str = "",
) -> list[Decision]:
    """把无手质量视图聚合报告（dunjia/a2d/umi）转成 Stage 7 Decision。"""

    decisions: list[Decision] = []
    view_items = (
        views_report.views.items()
        if hasattr(views_report, "views")
        else views_report.items()
    )
    for name, view in view_items:
        ready, disposition, reasons = _extract_view_state(view)
        ready_reason, fail_reason = _reason_pair(name)
        detail = {
            "session_id": session_id,
            "view": name,
            "reasons": reasons,
            "config_hash": config_hash,
        }
        if ready:
            decisions.append(
                Decision(
                    stage=7,
                    reason=ready_reason,
                    severity=Severity.INFO,
                    message=f"{name}: 通过",
                    disposition=Disposition.KEEP,
                    detail=detail,
                )
            )
            continue
        if fail_reason == ReasonCode.ROBOT_OBSERVATION_NOT_READY:
            if disposition == "reject":
                decisions.append(
                    Decision(
                        stage=7,
                        reason=fail_reason,
                        severity=Severity.ERROR,
                        message="主视角观测不可用: " + "; ".join(reasons),
                        disposition=Disposition.REJECT,
                        detail=detail,
                    )
                )
            elif disposition == "keep_with_flag":
                decisions.append(
                    Decision(
                        stage=7,
                        reason=fail_reason,
                        severity=Severity.WARN,
                        message="主视角观测可用但带告警: " + "; ".join(reasons),
                        disposition=Disposition.QUARANTINE,
                        detail=detail,
                    )
                )
            else:
                decisions.append(
                    Decision(
                        stage=7,
                        reason=fail_reason,
                        severity=Severity.WARN,
                        message="主视角观测未评估: " + "; ".join(reasons),
                        disposition=Disposition.KEEP_WITH_FLAG,
                        detail=detail,
                    )
                )
        elif fail_reason == ReasonCode.END_EFFECTOR_NOT_VISIBLE:
            decisions.append(
                Decision(
                    stage=7,
                    reason=fail_reason,
                    severity=Severity.WARN,
                    message=(
                        "末端执行器不可见/未确认: " + "; ".join(reasons)
                    ),
                    disposition=(
                        Disposition.QUARANTINE
                        if disposition == "reject"
                        else Disposition.KEEP_WITH_FLAG
                    ),
                    detail=detail,
                )
            )
        else:
            decisions.append(
                Decision(
                    stage=7,
                    reason=fail_reason,
                    severity=Severity.WARN,
                    message=f"{name}: 未通过: " + "; ".join(reasons),
                    disposition=(
                        Disposition.QUARANTINE
                        if disposition in {"reject", "review_required"}
                        else Disposition.KEEP_WITH_FLAG
                    ),
                    detail=detail,
                )
            )
    return decisions


def _evaluate_dunjia_views(
    session,
    cfg: dict,
) -> list[Decision]:
    from zpds_prepare.detectors.dunjia import (
        aggregate_dunjia_quality_views,
        check_dunjia_completeness,
        check_dunjia_coverage,
        check_dunjia_imu,
        check_dunjia_rgbd,
    )

    require_depth = bool(
        cfg.get("dunjia", {}).get("depth", {}).get("required", True)
    )
    max_pairing_ns = int(
        cfg.get("dunjia", {})
        .get("rgbd", {})
        .get("max_pairing_offset_ns", 50_000_000)
    )
    spike_factor = float(
        cfg.get("dunjia", {}).get("imu", {}).get("spike_std_factor", 6.0)
    )
    completeness = check_dunjia_completeness(
        session, require_depth=require_depth
    )
    rgbd = check_dunjia_rgbd(session, max_pairing_offset_ns=max_pairing_ns)
    imu = check_dunjia_imu(session, spike_std_factor=spike_factor)
    coverage = check_dunjia_coverage(session)
    views = aggregate_dunjia_quality_views(
        completeness=completeness,
        rgbd=rgbd,
        imu=imu,
        coverage=coverage,
        session_id=session.session_id,
        source_path=str(getattr(session, "source_path", "")),
    )
    return views_to_decisions(
        views,
        session_id=session.session_id,
        config_hash="",
    )


def _evaluate_a2d_views(
    session,
    cfg: dict,
    episode_root: str | None,
) -> list[Decision]:
    from zpds_prepare.detectors.a2d import (
        aggregate_a2d_quality_views,
        check_a2d_alignment,
        check_a2d_completeness,
        check_a2d_robot_quality,
    )

    if episode_root is None:
        source_path = str(getattr(session, "source_path", ""))
        episode_root = source_path if Path(source_path).is_dir() else None
    if episode_root is None:
        return []
    completeness = check_a2d_completeness(episode_root)
    alignment = check_a2d_alignment(session)
    robot_quality = check_a2d_robot_quality(
        session,
        freeze_min_duration_s=float(
            cfg.get("a2d", {})
            .get("robot", {})
            .get("freeze_min_duration_s", 2.0)
        ),
        gap_factor=float(
            cfg.get("a2d", {}).get("robot", {}).get("gap_factor", 3.0)
        ),
    )
    views = aggregate_a2d_quality_views(
        completeness=completeness,
        alignment=alignment,
        robot_quality=robot_quality,
        episode_id=session.session_id,
        source_path=str(episode_root),
    )
    return views_to_decisions(
        views,
        session_id=session.session_id,
        config_hash="",
    )


def _evaluate_umi_views(
    session,
    cfg: dict,
) -> list[Decision]:
    from zpds_prepare.detectors.umi.orchestrator import analyze_umi_session
    from zpds_prepare.detectors.umi.view_evaluator import (
        evaluate_candidate_views,
    )

    bundle = analyze_umi_session(
        session,
        minimum_gap_ns=int(
            cfg.get("umi", {}).get("minimum_gap_ns", 500_000_000)
        ),
    )
    views = evaluate_candidate_views(
        bundle,
        require_vio_for_bimanual=bool(
            cfg.get("umi", {}).get("require_vio_for_bimanual", True)
        ),
    )
    return views_to_decisions(
        views,
        session_id=session.session_id,
        config_hash="",
    )


def evaluate_no_hand_observation_views(
    context: dict,
) -> list[Decision]:
    """无手来源的机器人观测质量检查（每个 session 只跑一次）。"""

    if context.get("robot_observation_checked"):
        return []
    profile = context.get("profile")
    if not profile:
        return []
    from zpds.profiles.registry import get as _get_profile

    registered = _get_profile(_profile_registry_name(str(profile)))
    if registered is not None and (
        registered.modalities.get("human_hand") == "applicable"
    ):
        return []
    session = context.get("session")
    if session is None:
        return []

    cfg = context.get("cfg", {})
    if profile == "dunjia":
        return _evaluate_dunjia_views(session, cfg)
    if profile == "a2d":
        return _evaluate_a2d_views(
            session,
            cfg,
            context.get("episode_root"),
        )
    if profile == "umi":
        return _evaluate_umi_views(session, cfg)
    return []


__all__ = ["evaluate_no_hand_observation_views", "views_to_decisions"]
