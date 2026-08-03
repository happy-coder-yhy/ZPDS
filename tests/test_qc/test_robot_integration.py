from __future__ import annotations

from types import SimpleNamespace

from zpds.core.decisions import Disposition, ReasonCode
from zpds.core.quality import QualityMetric, QualityView
from zpds.prepared.revision import read_revision_manifest, write_revision_manifest
from zpds.qc import get_stage_checker
from zpds.qc.robot_integration import (
    FormalRobotQualityAdapter,
    adapt_source_views,
    build_robot_qc_delivery,
)


def test_robot_profiles_skip_human_hand_stage9() -> None:
    checker = get_stage_checker(9)
    assert checker is not None
    for profile in ("jianzhi_umi", "dunjia_ego", "a2d_robot"):
        decisions = checker({"profile": profile})
        assert len(decisions) == 1
        assert decisions[0].reason is ReasonCode.CHECK_NOT_APPLICABLE
        assert decisions[0].detail["run_human_hand_model"] is False
        assert all(d.reason is not ReasonCode.HAND_ABSENT for d in decisions)


def test_umi_adapter_preserves_review_without_rejecting_rgb() -> None:
    adapter = FormalRobotQualityAdapter(version="test")
    rgb = adapter.adapt_view(
        SimpleNamespace(
            name="robot_observation_ready_candidate", status="candidate_pass",
            applicability="applicable", disposition="keep", reasons=(), dependencies=("video",),
            evidence_uris=(), version="candidate-v1",
        )
    )
    vio = adapter.adapt_view(
        SimpleNamespace(
            name="vio_ready_candidate", status="review_required", applicability="applicable",
            disposition="keep_with_flag", reasons=("header_topic_mismatch",),
            dependencies=("vio",), evidence_uris=(), version="candidate-v1",
        )
    )
    assert rgb.name == "robot_observation_ready"
    assert rgb.ready is True
    assert vio.name == "vio_ready"
    assert vio.disposition.value == "keep_with_flag"


def test_a2d_bc_reject_does_not_mutate_observation_view() -> None:
    source = SimpleNamespace(
        views={
            "robot_observation_ready": SimpleNamespace(
                ready=True, disposition="pass", reasons=[], depends_on=["head_rgb"], evidence_uris=[]
            ),
            "robot_bc_ready": SimpleNamespace(
                ready=False, disposition="reject", reasons=["alignment unreliable"],
                depends_on=["alignment"], evidence_uris=[]
            ),
        }
    )
    views = adapt_source_views(source, producer="person-b", version="v1", config_hash="a" * 64)
    assert views["robot_observation_ready"].ready is True
    assert views["robot_observation_ready"].disposition.value == "keep"
    assert views["robot_bc_ready"].ready is False
    assert views["robot_bc_ready"].disposition.value == "reject"


def test_delivery_round_trip_keeps_raw_and_skips_vlm(tmp_path) -> None:
    source = tmp_path / "raw.mcap"
    source.write_bytes(b"immutable")
    delivery = build_robot_qc_delivery(
        session_id="umi-1",
        profile="jianzhi_umi",
        source_assets=[{"uri": str(source), "sha256": "abc", "immutable": True}],
        modalities={"human_hand": "not_applicable", "end_effector": "applicable"},
        views={
            "robot_observation_ready": QualityView("robot_observation_ready", True),
            "vio_ready": QualityView("vio_ready", False, disposition=Disposition.KEEP_WITH_FLAG),
        },
        metrics=[QualityMetric("vio.invalid_count", 1, unit="event")],
        evidence_index={"metrics": "evidence/metrics.json"},
        stream_ranges={"robot0_camera": (0, 100), "robot1_camera": (10, 90)},
        idle_timestamps_ns=[10, 20, 30, 40, 50, 60, 70, 80, 90],
        robot_motion_energy=[0, 0, 1, 1, 1, 1, 1, 0, 0],
        gripper_event_energy=[0, 0, 1, 1, 1, 1, 1, 0, 0],
        visual_change_energy=[0, 0, 1, 1, 1, 1, 1, 0, 0],
        idle_thresholds={"robot_motion_max": 0, "gripper_event_max": 0, "visual_change_max": 0},
        effective_config={"idle": {"enabled": True}},
    )
    path = write_revision_manifest(tmp_path / "revision.json", delivery.manifest)
    loaded = read_revision_manifest(path)

    assert source.read_bytes() == b"immutable"
    assert loaded["raw_mutation"] is False
    assert loaded["outcome"]["status"] == "not_run"
    assert loaded["metrics"][0]["metric_name"] == "vio.invalid_count"
    assert len(loaded["physical_spans"]) == 1
    assert {candidate["edge"] for candidate in loaded["idle_candidates"]} == {"head", "tail"}
