"""B5 + B9 质量视图聚合单元测试。"""

from __future__ import annotations

import pytest

from zpds_prepare.detectors.a2d.alignment import (
    A2DAlignmentReport,
    StreamAlignmentSummary,
)
from zpds_prepare.detectors.a2d.completeness import (
    A2DAssetStatus,
    A2DCompletenessReport,
)
from zpds_prepare.detectors.a2d.quality_views import (
    aggregate_a2d_quality_views,
)
from zpds_prepare.detectors.a2d.robot_quality import (
    A2DRobotQualityReport,
    GripperResponse,
    TimeSeriesQuality,
)
from zpds_prepare.detectors.dunjia.completeness import (
    DunjiaCompletenessReport,
    StreamCompleteness,
)
from zpds_prepare.detectors.dunjia.coverage import (
    CameraCoverageRecord,
    DunjiaCoverageReport,
    EndEffectorVisibility,
)
from zpds_prepare.detectors.dunjia.imu_quality import DunjiaIMUReport
from zpds_prepare.detectors.dunjia.quality_views import (
    aggregate_dunjia_quality_views,
)
from zpds_prepare.detectors.dunjia.rgbd_quality import DunjiaRGBDReport


# ===================================================================
# B5: 遁甲质量视图
# ===================================================================


class TestDunjiaQualityViews:
    def test_all_pass(self):
        """所有检测通过 → robot_observation_ready=true。"""
        comp = DunjiaCompletenessReport(
            session_id="test", source_path="/t",
            source_sha256="abc",
            required_present=5, required_total=5,
        )
        comp.streams["camera0"] = StreamCompleteness(
            stream_id="camera0", stream_type="video",
            required=True, present=True, frame_count=351,
            disposition="pass",
        )

        cov = DunjiaCoverageReport(session_id="test", source_path="/t")
        cov.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=351,
            start_timestamp_ns=0, end_timestamp_ns=14_000_000_000,
            duration_s=14.0, decode_status="decodable",
        )

        report = aggregate_dunjia_quality_views(
            completeness=comp, coverage=cov,
        )

        assert report.views["robot_observation_ready"].ready
        assert report.views["robot_observation_ready"].disposition == "pass"

    def test_camera0_missing(self):
        """主视角缺失 → robot_observation_ready=false。"""
        comp = DunjiaCompletenessReport(
            session_id="test", source_path="/t",
            source_sha256="abc",
        )
        comp.streams["camera0"] = StreamCompleteness(
            stream_id="camera0", stream_type="video",
            required=True, present=False, frame_count=0,
            disposition="reject",
        )

        report = aggregate_dunjia_quality_views(completeness=comp)

        obs = report.views["robot_observation_ready"]
        assert not obs.ready
        assert obs.disposition == "reject"

    def test_end_effector_unassessed(self):
        """无几何 → end_effector_visible=unassessed + keep_with_flag。"""
        cov = DunjiaCoverageReport(session_id="test", source_path="/t")
        cov.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=351,
            start_timestamp_ns=0, end_timestamp_ns=14e9,
            duration_s=14.0, decode_status="decodable",
        )
        cov.end_effector_visibility = EndEffectorVisibility(
            status="unassessed",
            assessment_method="camera_coverage_only",
            notes="需 VLM 复核",
        )

        report = aggregate_dunjia_quality_views(coverage=cov)

        ee = report.views["end_effector_visible"]
        assert ee.ready
        assert ee.disposition == "keep_with_flag"

    def test_rgbd_warning_does_not_block_rgb(self):
        """RGB-D 质量告警不阻断 robot_observation_ready。"""
        comp = DunjiaCompletenessReport(
            session_id="test", source_path="/t",
            source_sha256="abc",
            required_present=5, required_total=5,
        )
        comp.streams["camera0"] = StreamCompleteness(
            stream_id="camera0", stream_type="video",
            required=True, present=True, frame_count=351,
            disposition="pass",
        )

        cov = DunjiaCoverageReport(session_id="test", source_path="/t")
        cov.cameras["camera0"] = CameraCoverageRecord(
            camera_id="camera0", role="primary",
            role_source="test", frame_id="cam0",
            frame_count=351,
            start_timestamp_ns=0, end_timestamp_ns=14e9,
            duration_s=14.0, decode_status="decodable",
        )

        rgbd = DunjiaRGBDReport(
            session_id="test", source_path="/t",
            issues=["深度零值比例过高"],
            overall_disposition="keep_with_flag",
        )

        report = aggregate_dunjia_quality_views(
            completeness=comp, coverage=cov, rgbd=rgbd,
        )

        obs = report.views["robot_observation_ready"]
        assert obs.ready  # RGB 仍然可用
        assert obs.disposition == "keep_with_flag"  # 但有告警


# ===================================================================
# B9: A2D 质量视图
# ===================================================================


class TestA2DQualityViews:
    def test_observation_ready_pass(self):
        """head_rgb 完整 → robot_observation_ready=true。"""
        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
            required_present=6, required_total=6,
        )
        comp.assets["head_rgb"] = A2DAssetStatus(
            asset_id="head_rgb", asset_type="camera_rgb",
            required=True, present=True, frame_count=166,
        )

        report = aggregate_a2d_quality_views(completeness=comp)

        obs = report.views["robot_observation_ready"]
        assert obs.ready

    def test_no_head_rgb(self):
        """head_rgb 缺失 → robot_observation_ready=false。"""
        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
        )
        comp.assets["head_rgb"] = A2DAssetStatus(
            asset_id="head_rgb", asset_type="camera_rgb",
            required=True, present=False, frame_count=0,
        )

        report = aggregate_a2d_quality_views(completeness=comp)

        obs = report.views["robot_observation_ready"]
        assert not obs.ready
        assert obs.disposition == "reject"

    def test_bc_ready_pass(self):
        """对齐 + 机器人质量全通过 → robot_bc_ready=true。"""
        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
        )
        comp.assets["aligned_joints.h5"] = A2DAssetStatus(
            asset_id="aligned_joints.h5", asset_type="hdf5",
            required=True, present=True,
        )
        comp.hdf5_sample_count = 1630

        align = A2DAlignmentReport(
            episode_id="8032", source_path="/t",
        )
        align.robot_bc_ready = True
        align.streams["head_rgb"] = StreamAlignmentSummary(
            stream_id="head_rgb",
            total_camera_frames=166, mapped_frames=166,
            unmapped_frames=0, continuity_groups=1,
        )

        rq = A2DRobotQualityReport(
            episode_id="8032", source_path="/t",
        )
        rq.robot_bc_ready = True
        rq.robot_state_quality = TimeSeriesQuality(
            stream_id="robot_state",
            sample_count=1630, field_count=18, joint_count=18,
            nan_count=0, inf_count=0, finite_ratio=1.0,
        )

        report = aggregate_a2d_quality_views(
            completeness=comp, alignment=align, robot_quality=rq,
        )

        bc = report.views["robot_bc_ready"]
        assert bc.ready

    def test_bc_ready_false_when_alignment_fails(self):
        """对齐失败 → robot_bc_ready=false。"""
        align = A2DAlignmentReport(
            episode_id="8032", source_path="/t",
        )
        align.robot_bc_ready = False
        align.issues.append("映射覆盖率不足")

        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
        )
        comp.assets["aligned_joints.h5"] = A2DAssetStatus(
            asset_id="aligned_joints.h5", asset_type="hdf5",
            required=True, present=True,
        )
        comp.hdf5_sample_count = 1630

        rq = A2DRobotQualityReport(
            episode_id="8032", source_path="/t",
        )
        rq.robot_bc_ready = True
        rq.robot_state_quality = TimeSeriesQuality(
            stream_id="robot_state",
            sample_count=1630, field_count=18, joint_count=18,
            nan_count=0, inf_count=0, finite_ratio=1.0,
        )

        report = aggregate_a2d_quality_views(
            completeness=comp, alignment=align, robot_quality=rq,
        )

        bc = report.views["robot_bc_ready"]
        assert not bc.ready
        assert bc.disposition == "reject"

    def test_observation_ready_independent_of_bc(self):
        """robot_bc_ready=false 不牵连 robot_observation_ready。"""
        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
            required_present=6, required_total=6,
        )
        comp.assets["head_rgb"] = A2DAssetStatus(
            asset_id="head_rgb", asset_type="camera_rgb",
            required=True, present=True, frame_count=166,
        )
        comp.assets["aligned_joints.h5"] = A2DAssetStatus(
            asset_id="aligned_joints.h5", asset_type="hdf5",
            required=True, present=False,
        )
        comp.hdf5_sample_count = 0

        report = aggregate_a2d_quality_views(completeness=comp)

        # RGB 可用
        assert report.views["robot_observation_ready"].ready
        # BC 不可用（无 HDF5）
        assert not report.views["robot_bc_ready"].ready

    def test_geometry_unavailable(self):
        """外参缺失 → geometry_ready=keep_with_flag。"""
        comp = A2DCompletenessReport(
            episode_id="8032", source_path="/t",
            source_sha256="abc",
        )
        comp.assets["camera_calibration"] = A2DAssetStatus(
            asset_id="camera_calibration", asset_type="calibration",
            required=True, present=True,
            details={
                "present_calibrations": ["head_rgb", "hand_left_rgb", "hand_right_rgb"],
                "missing_calibrations": [],
            },
        )

        report = aggregate_a2d_quality_views(completeness=comp)

        geo = report.views["geometry_ready"]
        assert geo.ready
        assert geo.disposition == "keep_with_flag"  # 外参 unavailable

    def test_gripper_stall_flagged(self):
        """夹爪失速 → failure_recovery flagged。"""
        rq = A2DRobotQualityReport(
            episode_id="8032", source_path="/t",
        )
        rq.gripper_response = GripperResponse(
            command_count=50, response_count=45,
            stall_count=5, no_op_count=50,
        )

        report = aggregate_a2d_quality_views(robot_quality=rq)

        fr = report.views["failure_recovery"]
        assert fr.ready
        assert fr.disposition == "keep_with_flag"
        assert any("失速" in r for r in fr.reasons)
