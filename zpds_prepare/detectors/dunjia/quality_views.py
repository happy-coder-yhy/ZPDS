"""B5: 遁甲质量视图聚合。

将 B1–B4 的检测结果聚合成独立的、可独立判定的质量视图。

质量视图:
  - ``robot_observation_ready`` — 视觉观测可用（RGB 可解码 + 时间连续）
  - ``end_effector_visible`` — 末端执行器在主视角中可见（或可被侧视角互补）

原则:
  - 各视图独立判定，不相互否定
  - 末端不可见 ≠ 任务失败
  - Depth/IMU/几何子视图失败不得连带拒绝可用 RGB
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from zpds_prepare.detectors.dunjia.completeness import DunjiaCompletenessReport
from zpds_prepare.detectors.dunjia.coverage import DunjiaCoverageReport
from zpds_prepare.detectors.dunjia.imu_quality import DunjiaIMUReport
from zpds_prepare.detectors.dunjia.rgbd_quality import DunjiaRGBDReport


@dataclass
class QualityView:
    """单个质量视图。"""

    name: str
    ready: bool
    disposition: str  # "pass" | "keep_with_flag" | "reject"
    reasons: list[str] = field(default_factory=list)
    depends_on: list[str] = field(default_factory=list)  # 依赖的其他视图名
    evidence_uris: list[str] = field(default_factory=list)


@dataclass
class DunjiaQualityViewsReport:
    """遁甲质量视图聚合报告。"""

    session_id: str
    source_path: str
    schema_version: str = "zpds.dunjia_quality_views.v1"

    views: dict[str, QualityView] = field(default_factory=dict)
    issues: list[str] = field(default_factory=list)


def aggregate_dunjia_quality_views(
    completeness: DunjiaCompletenessReport | None = None,
    rgbd: DunjiaRGBDReport | None = None,
    imu: DunjiaIMUReport | None = None,
    coverage: DunjiaCoverageReport | None = None,
    session_id: str = "",
    source_path: str = "",
) -> DunjiaQualityViewsReport:
    """聚合遁甲 B1–B4 检测结果为质量视图。

    各视图只依赖其相关检测器，独立判定。
    Depth/IMU 视图失败不牵连 RGB 观测视图。

    Args:
        completeness: B1 完整性报告
        rgbd: B2 RGB-D 质量报告
        imu: B3 IMU 质量报告
        coverage: B4 覆盖与可见性报告
        session_id: session 标识
        source_path: 源数据路径

    Returns:
        DunjiaQualityViewsReport 含各质量视图的独立判定。
    """
    report = DunjiaQualityViewsReport(
        session_id=session_id,
        source_path=source_path,
    )

    # ---- robot_observation_ready ----
    obs_view = _evaluate_robot_observation(completeness, rgbd, coverage, imu)
    report.views["robot_observation_ready"] = obs_view

    # ---- end_effector_visible ----
    ee_view = _evaluate_end_effector_visible(completeness, coverage)
    report.views["end_effector_visible"] = ee_view

    # 汇总 issues
    for view in report.views.values():
        for reason in view.reasons:
            report.issues.append(f"[{view.name}] {reason}")

    return report


def _evaluate_robot_observation(
    completeness: DunjiaCompletenessReport | None,
    rgbd: DunjiaRGBDReport | None,
    coverage: DunjiaCoverageReport | None,
    imu: DunjiaIMUReport | None = None,
) -> QualityView:
    """评估 robot_observation_ready。

    要求: 主视角 camera0 可解码 + 时间连续，深度/IMU 可以有告警但不阻断。
    """
    view = QualityView(
        name="robot_observation_ready",
        ready=False,
        disposition="pass",
        depends_on=["camera0_rgb", "depth", "imu"],
    )

    # B1: 主视角存在
    if completeness is not None:
        c0 = completeness.streams.get("camera0")
        if c0 is None or not c0.present:
            view.reasons.append("camera0 不存在 (B1)")
            view.ready = False
            view.disposition = "reject"
            return view

    # B4: 主视角可解码
    if coverage is not None:
        c0_cov = coverage.cameras.get("camera0")
        if c0_cov is not None:
            if c0_cov.decode_status == "unavailable":
                view.reasons.append("camera0 不可用 (B4)")
                view.ready = False
                view.disposition = "reject"
                return view
            if c0_cov.decode_status == "undecodable":
                view.reasons.append("camera0 无法解码 (B4)")
                view.ready = False
                view.disposition = "reject"
                return view

    # 深度/IMU 仅告警
    if rgbd is not None and rgbd.overall_disposition != "pass":
        view.reasons.append(f"RGB-D 质量有告警: {rgbd.issues}")
    if imu is not None and imu.overall_disposition != "pass":
        view.reasons.append(f"IMU 质量有告警: {imu.issues}")

    view.ready = True
    if view.reasons:
        view.disposition = "keep_with_flag"
    return view


def _evaluate_end_effector_visible(
    completeness: DunjiaCompletenessReport | None,
    coverage: DunjiaCoverageReport | None,
) -> QualityView:
    """评估 end_effector_visible。

    遁甲无机器人 kinematics，仅基于相机覆盖做判定。
    """
    view = QualityView(
        name="end_effector_visible",
        ready=False,
        disposition="pass",
        depends_on=["camera0_rgb", "camera1_rgb", "camera2_rgb"],
    )

    if coverage is not None:
        ee = coverage.end_effector_visibility
        if ee.status == "not_visible":
            view.reasons.append(f"末端不可见: {ee.notes}")
            view.ready = False
            view.disposition = "reject"
            return view
        if ee.status == "unassessed":
            view.reasons.append(f"末端可见性未评估: {ee.notes}")
            view.ready = True
            view.disposition = "keep_with_flag"
            return view

    # 无 coverage 报告时
    if completeness is not None:
        c0 = completeness.streams.get("camera0")
        if c0 is not None and c0.present:
            view.reasons.append("主视角可用，末端可见性待 VLM/人工复核")
            view.ready = True
            view.disposition = "keep_with_flag"
        else:
            view.reasons.append("主视角不可用，末端不可见")
            view.ready = False
            view.disposition = "reject"
    else:
        view.reasons.append("缺少覆盖报告，无法判定")
        view.ready = False
        view.disposition = "reject"

    return view


__all__ = [
    "DunjiaQualityViewsReport",
    "QualityView",
    "aggregate_dunjia_quality_views",
]
