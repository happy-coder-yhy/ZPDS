"""Stage 7 无手来源机器人观测质量视图测试。"""

from __future__ import annotations

from types import SimpleNamespace

from zpds.core.decisions import Disposition, ReasonCode
from zpds.qc.cascade import get_stage_checker
from zpds.qc.stage7_robot_observation import (
    evaluate_no_hand_observation_views,
    views_to_decisions,
)
from zpds_prepare.detectors.dunjia.quality_views import (
    DunjiaQualityViewsReport,
    QualityView,
)


class TestViewsToDecisions:
    def test_without_detector_reports(self) -> None:
        views = DunjiaQualityViewsReport(
            session_id="dunjia_1",
            source_path="",
        )
        views.views["robot_observation_ready"] = QualityView(
            name="robot_observation_ready",
            ready=False,
            disposition="reject",
            reasons=["camera0 缺失"],
        )
        views.views["end_effector_visible"] = QualityView(
            name="end_effector_visible",
            ready=False,
            disposition="reject",
            reasons=["末端不可见"],
        )
        decisions = views_to_decisions(views, session_id="dunjia_1")
        reasons = [decision.reason for decision in decisions]

        assert ReasonCode.ROBOT_OBSERVATION_NOT_READY in reasons
        assert ReasonCode.END_EFFECTOR_NOT_VISIBLE in reasons
        robot = next(
            decision
            for decision in decisions
            if decision.reason == ReasonCode.ROBOT_OBSERVATION_NOT_READY
        )
        assert robot.disposition is Disposition.REJECT
        end_effector = next(
            decision
            for decision in decisions
            if decision.reason == ReasonCode.END_EFFECTOR_NOT_VISIBLE
        )
        assert end_effector.disposition is Disposition.QUARANTINE
        assert end_effector.detail["session_id"] == "dunjia_1"


class TestEvaluateNoHandViews:
    def test_skips_when_human_hand_applicable(self) -> None:
        assert evaluate_no_hand_observation_views({"profile": "guida"}) == []

    def test_skips_after_checked_once(self) -> None:
        assert (
            evaluate_no_hand_observation_views(
                {"profile": "dunjia", "robot_observation_checked": True}
            )
            == []
        )

    def test_empty_without_session(self) -> None:
        assert evaluate_no_hand_observation_views({"profile": "dunjia"}) == []

    def test_umi_returns_empty_without_session(self) -> None:
        assert evaluate_no_hand_observation_views({"profile": "umi"}) == []

    def test_a2d_returns_empty_without_session(self) -> None:
        assert evaluate_no_hand_observation_views({"profile": "a2d"}) == []


class TestUmiCandidateViewMapping:
    def test_candidate_pass_and_review_required(self) -> None:
        from zpds_prepare.detectors.umi.view_evaluator import UmiCandidateView

        report = SimpleNamespace(
            views={
                "robot_observation_ready_candidate": UmiCandidateView(
                    name="robot_observation_ready_candidate",
                    status="candidate_pass",
                    applicability="applicable",
                    disposition="keep",
                    dependencies=(),
                ),
                "bimanual_umi_ready_candidate": UmiCandidateView(
                    name="bimanual_umi_ready_candidate",
                    status="review_required",
                    applicability="applicable",
                    disposition="keep_with_flag",
                    dependencies=(),
                    reasons=("vio_unavailable",),
                ),
            }
        )
        decisions = views_to_decisions(report, session_id="umi_1")
        by_reason = {decision.reason: decision for decision in decisions}

        assert ReasonCode.ROBOT_OBSERVATION_READY in by_reason
        assert by_reason[ReasonCode.ROBOT_OBSERVATION_READY].disposition is (
            Disposition.KEEP
        )
        assert ReasonCode.ROBOT_VIEW_FAIL in by_reason
        assert by_reason[ReasonCode.ROBOT_VIEW_FAIL].disposition is (
            Disposition.KEEP_WITH_FLAG
        )


class TestA2dViewMapping:
    def test_generic_view_failure(self) -> None:
        report = SimpleNamespace(
            views={
                "robot_observation_ready": SimpleNamespace(
                    ready=True,
                    disposition="pass",
                    reasons=[],
                ),
                "robot_bc_ready": SimpleNamespace(
                    ready=False,
                    disposition="reject",
                    reasons=["state_action_lag"],
                ),
            }
        )
        decisions = views_to_decisions(report, session_id="a2d_1")
        by_reason = {decision.reason: decision for decision in decisions}

        assert ReasonCode.ROBOT_OBSERVATION_READY in by_reason
        assert ReasonCode.ROBOT_VIEW_FAIL in by_reason
        assert by_reason[ReasonCode.ROBOT_VIEW_FAIL].disposition is (
            Disposition.QUARANTINE
        )


def test_stage7_registered_from_robot_module() -> None:
    checker = get_stage_checker(7)
    assert checker is not None
    assert checker.__module__ == "zpds.qc.stage7_robot"
