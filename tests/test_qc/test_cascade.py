"""QCCascade 调度器测试。"""

import cv2
import numpy as np

from zpds.core.decisions import Decision, ReasonCode, Severity
from zpds.core.quality import QualityReport
from zpds.qc.cascade import (
    _STAGE_REGISTRY,
    CascadeConfig,
    CascadeDistribution,
    QCCascade,
    get_stage_checker,
    register_stage,
)

# ---------------------------------------------------------------------------
# 基础功能
# ---------------------------------------------------------------------------


class TestStageRegistry:
    def test_register_single(self):
        """手动注册一个测试 stage。"""
        @register_stage(99)
        def _test_checker(context: dict) -> list[Decision]:
            return [Decision(stage=99, reason=ReasonCode.DELIVERY_CHECK_FAIL, severity=Severity.INFO, message="test")]

        checker = get_stage_checker(99)
        assert checker is not None
        result = checker({})
        assert len(result) == 1
        assert result[0].stage == 99

        # clean up
        _STAGE_REGISTRY.pop(99, None)

    def test_registered_stages_on_import(self):
        """导入后应有 4 个 stage 已注册。"""
        assert 3 in _STAGE_REGISTRY
        assert 5 in _STAGE_REGISTRY
        assert 6 in _STAGE_REGISTRY
        assert 11 in _STAGE_REGISTRY


# ---------------------------------------------------------------------------
# CascadeConfig
# ---------------------------------------------------------------------------


class TestCascadeConfig:
    def test_defaults(self):
        cfg = CascadeConfig()
        assert len(cfg.enabled_stages) == 13
        assert Severity.FATAL in cfg.stop_on_severity

    def test_custom_stages(self):
        cfg = CascadeConfig(enabled_stages=[3, 5, 6])
        assert cfg.enabled_stages == [3, 5, 6]


# ---------------------------------------------------------------------------
# CascadeDistribution
# ---------------------------------------------------------------------------


class TestCascadeDistribution:
    def test_record(self):
        dist = CascadeDistribution()
        d = Decision(stage=3, reason=ReasonCode.BLACK_FRAME, severity=Severity.WARN, message="test")
        dist.record(d)
        assert dist.total_decisions == 1
        assert dist.severity_counts.get("warn") == 1
        assert dist.reason_counts.get("black_frame") == 1

    def test_to_dict(self):
        dist = CascadeDistribution()
        d = Decision(stage=5, reason=ReasonCode.DEPTH_INVALID_RATIO, severity=Severity.WARN, message="test")
        dist.record(d)
        result = dist.to_dict()
        assert result["total_decisions"] == 1
        assert "by_severity" in result
        assert "by_reason" in result
        assert "by_stage" in result


# ---------------------------------------------------------------------------
# QCCascade 核心逻辑
# ---------------------------------------------------------------------------


class TestQCCascade:
    def test_run_all_stages(self):
        """运行全部 4 个已注册 stage（使用 dummy context）。"""
        cascade = QCCascade(
            config=CascadeConfig(enabled_stages=[3, 5, 6, 11]),
        )
        report = cascade.run(
            context={
                "session_id": "test_session",
                "video_path": "",
                "imu_timestamps_ns": [0, 5_000_000, 10_000_000],
                "imu_values": np.random.randn(3, 6),
            }
        )
        assert isinstance(report, QualityReport)
        assert report.session_id == "test_session"

    def test_overall_pass_fatal(self):
        """FATAL 应导致 overall_pass=False。"""
        @register_stage(50)
        def _fatal_checker(context: dict) -> list[Decision]:
            return [Decision(stage=50, reason=ReasonCode.DELIVERY_CHECK_FAIL, severity=Severity.FATAL, message="fatal")]

        cascade = QCCascade(config=CascadeConfig(enabled_stages=[50]))
        report = cascade.run(context={"session_id": "test"})
        assert report.overall_pass is False

        _STAGE_REGISTRY.pop(50, None)

    def test_overall_pass_clean(self):
        """全 INFO 应通过。"""
        @register_stage(51)
        def _clean_checker(context: dict) -> list[Decision]:
            return [Decision(stage=51, reason=ReasonCode.NEAR_DUPLICATE, severity=Severity.INFO, message="clean")]

        cascade = QCCascade(config=CascadeConfig(enabled_stages=[51]))
        report = cascade.run(context={"session_id": "test"})
        assert report.overall_pass is True

        _STAGE_REGISTRY.pop(51, None)

    def test_stop_on_fatal(self):
        """FATAL 后应提前终止级联。"""
        call_count = {"count": 0}

        @register_stage(60)
        def _fatal_first(context: dict) -> list[Decision]:
            call_count["count"] += 1
            return [Decision(stage=60, reason=ReasonCode.DELIVERY_CHECK_FAIL, severity=Severity.FATAL)]

        @register_stage(61)
        def _never_called(context: dict) -> list[Decision]:
            call_count["count"] += 1
            return []

        cascade = QCCascade(
            config=CascadeConfig(enabled_stages=[60, 61]),
        )
        cascade.run(context={"session_id": "test"})
        assert call_count["count"] == 1  # stage 61 should not be called

        _STAGE_REGISTRY.pop(60, None)
        _STAGE_REGISTRY.pop(61, None)

    def test_skips_unregistered_stages(self):
        """未注册 stage 应跳过。"""
        cascade = QCCascade(config=CascadeConfig(enabled_stages=[0, 1, 2]))
        report = cascade.run(context={"session_id": "test"})
        assert isinstance(report, QualityReport)
        assert report.session_id == "test"

    def test_distribution_after_run(self):
        """运行后应填充分布统计。"""
        @register_stage(70)
        def _warn_checker(context: dict) -> list[Decision]:
            return [
                Decision(stage=70, reason=ReasonCode.BLACK_FRAME, severity=Severity.WARN, message="w1"),
                Decision(stage=70, reason=ReasonCode.BLUR_DETECTED, severity=Severity.INFO, message="i1"),
            ]

        cascade = QCCascade(config=CascadeConfig(enabled_stages=[70]))
        cascade.run(context={"session_id": "test"})
        dist = cascade.distribution
        assert dist.total_decisions == 2
        assert dist.severity_counts.get("warn") == 1
        assert dist.severity_counts.get("info") == 1
        assert dist.reason_counts.get("black_frame") == 1

        _STAGE_REGISTRY.pop(70, None)

    def test_exception_in_stage(self):
        """Stage 抛出异常应被捕获并生成 ERROR Decision。"""
        @register_stage(80)
        def _crashing_checker(context: dict) -> list[Decision]:
            raise RuntimeError("simulated failure")

        cascade = QCCascade(config=CascadeConfig(enabled_stages=[80]))
        report = cascade.run(context={"session_id": "test"})
        errors = [d for d in report.decisions if d.severity == Severity.ERROR]
        assert len(errors) >= 1
        assert "unhandled exception" in errors[0].message.lower()

        _STAGE_REGISTRY.pop(80, None)


# ---------------------------------------------------------------------------
# from_profile
# ---------------------------------------------------------------------------


class TestFromProfile:
    def test_guida_profile(self):
        """从 Guida profile 加载应成功。"""
        cascade = QCCascade.from_profile("guida_ego")
        assert cascade is not None
        assert len(cascade.config.enabled_stages) == 13

    def test_unknown_profile(self):
        """未知 profile 应使用默认配置。"""
        cascade = QCCascade.from_profile("nonexistent_profile")
        assert cascade is not None

    def test_guida_alias_loads_canonical_thresholds(self):
        cascade = QCCascade.from_profile("guida")

        assert cascade._stage_configs[6]["max_gap_s"] == 0.06


# ---------------------------------------------------------------------------
# 端到端：真实视频
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def test_full_cascade_with_video(self, tmp_path):
        """用真实视频运行完整级联。"""
        # 创建测试视频
        frames = [np.random.randint(0, 255, (120, 160, 3), dtype=np.uint8) for _ in range(10)]
        vpath = str(tmp_path / "test.mp4")
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(vpath, fourcc, 30.0, (w, h))
        for f in frames:
            writer.write(f)
        writer.release()

        # 创建 IMU 数据
        n = 100
        ts = list(range(0, n * 5_000_000, 5_000_000))
        rng = np.random.RandomState(42)
        imu_vals = rng.normal(0, 0.1, (n, 6))
        imu_vals[:, :3] += 9.81

        cascade = QCCascade(config=CascadeConfig(enabled_stages=[3, 5, 6, 11]))
        report = cascade.run(
            context={
                "session_id": "test_e2e",
                "video_path": vpath,
                "depth_frames": [np.full((60, 80), 500, dtype=np.uint16) for _ in range(5)],
                "imu_timestamps_ns": ts,
                "imu_values": imu_vals,
                "file_paths": [vpath],
            }
        )

        assert isinstance(report, QualityReport)
        assert report.session_id == "test_e2e"
        # decisions 应包含多个 stage 的结果
        stages_found = {d.stage for d in report.decisions}
        assert len(stages_found) >= 1  # at least one stage produced output

        # 分布统计
        dist = cascade.distribution_dict()
        assert "total_decisions" in dist
