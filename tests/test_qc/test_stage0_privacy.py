"""测试 Stage 0 隐私门 QC 集成。"""

import pytest

from zpds.core.decisions import Disposition, ReasonCode, Severity
from zpds.qc.stage0_privacy import build_privacy_view, check


class TestStage0Check:
    def test_empty_manifest(self):
        decisions = check(manifest=None)
        assert len(decisions) == 1
        assert decisions[0].reason == ReasonCode.PRIVACY_COVERAGE_LOW
        assert decisions[0].severity == Severity.WARN

    def test_llm_unavailable(self):
        manifest = {"llm_available": False, "stats": {"total_frames": 100}}
        decisions = check(manifest=manifest)
        assert any(d.reason == ReasonCode.PRIVACY_LLM_UNAVAILABLE for d in decisions)
        assert any(d.severity == Severity.ERROR for d in decisions)

    def test_face_masked(self):
        manifest = {
            "llm_available": True,
            "stats": {
                "total_frames": 100,
                "frames_with_faces": 10,
                "total_face_regions": 25,
                "frames_with_text": 5,
                "total_text_regions": 5,
                "total_pii_masked": 3,
                "pii_categories_found": ["phone"],
            },
        }
        decisions = check(
            manifest=manifest,
            stage_config={"face": {"applicability": "applicable"}, "text": {"applicability": "applicable"}},
        )
        reasons = [d.reason for d in decisions]
        assert ReasonCode.PRIVACY_FACE_MASKED in reasons
        assert ReasonCode.PRIVACY_PII_MASKED in reasons
        # 有检测到人脸/文本，不应有 COVERAGE_LOW
        assert ReasonCode.PRIVACY_COVERAGE_LOW not in reasons

    def test_coverage_low_when_face_applicable_but_no_faces(self):
        manifest = {
            "llm_available": True,
            "stats": {
                "total_frames": 100,
                "frames_with_faces": 0,
                "total_face_regions": 0,
                "frames_with_text": 5,
                "total_text_regions": 5,
                "total_pii_masked": 0,
                "pii_categories_found": [],
            },
        }
        decisions = check(
            manifest=manifest,
            stage_config={"face": {"applicability": "applicable"}, "text": {"applicability": "applicable"}},
        )
        coverage = [d for d in decisions if d.reason == ReasonCode.PRIVACY_COVERAGE_LOW]
        assert len(coverage) == 1  # only face coverage, text has regions
        assert coverage[0].severity == Severity.WARN

    def test_disabled_stage_returns_empty(self):
        decisions = check(stage_config={"enabled": False})
        assert decisions == []

    def test_not_applicable_no_coverage_warning(self):
        manifest = {
            "llm_available": True,
            "stats": {
                "total_frames": 100,
                "frames_with_faces": 0,
                "total_face_regions": 0,
                "frames_with_text": 0,
                "total_text_regions": 0,
                "total_pii_masked": 0,
                "pii_categories_found": [],
            },
        }
        decisions = check(
            manifest=manifest,
            stage_config={
                "face": {"applicability": "not_applicable"},
                "text": {"applicability": "not_applicable"},
            },
        )
        # not_applicable 时不产生 COVERAGE_LOW
        coverage = [d for d in decisions if d.reason == ReasonCode.PRIVACY_COVERAGE_LOW]
        assert len(coverage) == 0


class TestBuildPrivacyView:
    def test_all_clean(self):
        from zpds.core.decisions import Decision
        decisions = [
            Decision(0, ReasonCode.PRIVACY_FACE_MASKED, Severity.INFO, "ok", disposition=Disposition.KEEP),
        ]
        view = build_privacy_view(decisions)
        assert view.ready is True
        assert view.disposition == Disposition.KEEP

    def test_error_rejects(self):
        from zpds.core.decisions import Decision
        decisions = [
            Decision(0, ReasonCode.PRIVACY_LLM_UNAVAILABLE, Severity.ERROR, "fail", disposition=Disposition.REJECT),
        ]
        view = build_privacy_view(decisions)
        assert view.ready is False
        assert view.disposition == Disposition.REJECT

    def test_warn_flags(self):
        from zpds.core.decisions import Decision
        decisions = [
            Decision(0, ReasonCode.PRIVACY_COVERAGE_LOW, Severity.WARN, "low", disposition=Disposition.QUARANTINE),
        ]
        view = build_privacy_view(decisions)
        assert view.ready is True
        assert view.disposition == Disposition.KEEP_WITH_FLAG


class TestReasonCodes:
    def test_new_reason_codes_exist(self):
        assert ReasonCode.PRIVACY_FACE_MASKED == "privacy_face_masked"
        assert ReasonCode.PRIVACY_PII_MASKED == "privacy_pii_masked"
        assert ReasonCode.PRIVACY_LLM_UNAVAILABLE == "privacy_llm_unavailable"
        assert ReasonCode.PRIVACY_COVERAGE_LOW == "privacy_coverage_low"

    def test_existing_codes_unchanged(self):
        # 确保旧 ReasonCode 未受损
        assert ReasonCode.OVEREXPOSED == "overexposed"
        assert ReasonCode.BLACK_FRAME == "black_frame"
        assert ReasonCode.HAND_ABSENT == "hand_absent"
        assert ReasonCode.CHECK_NOT_APPLICABLE == "check_not_applicable"


class TestStage0Registered:
    def test_stage0_is_registered(self):
        from zpds.qc.cascade import get_stage_checker
        checker = get_stage_checker(0)
        assert checker is not None, "Stage 0 未在级联中注册"

    def test_stage0_returns_decisions(self):
        from zpds.qc.cascade import get_stage_checker
        checker = get_stage_checker(0)
        assert checker is not None
        decisions = checker({"privacy_manifest": {"llm_available": True, "stats": {"total_frames": 0}}})
        assert isinstance(decisions, list)
