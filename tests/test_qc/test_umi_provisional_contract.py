from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from zpds_prepare.detectors.umi import (
    UmiProvisionalMetric,
    analyze_umi_session,
    build_umi_source_contract_candidate,
    deterministic_config_hash,
    evaluate_candidate_views,
    run_umi_provisional_session,
    write_umi_evidence_bundle,
)
from zpds_prepare.readers.session_model import (
    ImuStream,
    Session,
    TimeSeriesStream,
    VideoStream,
)


def _video(robot_id: str, timestamps: list[int]) -> VideoStream:
    return VideoStream(
        stream_id=f"{robot_id}_camera0",
        timestamps_ns=timestamps,
        index_frames=[],
        video_path="unused.mp4",
        fps=30.0,
        frame_count=len(timestamps),
    )


def _imu(robot_id: str, timestamps: list[int]) -> ImuStream:
    return ImuStream(
        stream_id=f"{robot_id}_imu",
        dataframe=pd.DataFrame({"timestamp_ns": timestamps}),
        sample_rate_hz=100.0,
    )


def _vio(
    tmp_path: Path,
    robot_id: str,
    timestamps: list[int],
) -> TimeSeriesStream:
    count = len(timestamps)
    return TimeSeriesStream(
        stream_id=f"{robot_id}_vio_pose",
        modality="vio_pose",
        role="state",
        source_path=tmp_path / "source.mcap",
        timestamps_ns=timestamps,
        rows=pd.DataFrame(
            {
                "tx": list(range(count)),
                "ty": [0.0] * count,
                "tz": [0.0] * count,
                "qx": [0.0] * count,
                "qy": [0.0] * count,
                "qz": [0.0] * count,
                "qw": [1.0] * count,
                "source_frame_id": pd.Series(["world"] * count, dtype="string"),
                "source_header_topic": pd.Series(
                    [f"/{robot_id}/vio/eef_pose"] * count,
                    dtype="string",
                ),
            }
        ),
        fields=[],
        expected_rate_hz=100.0,
        metadata={
            "robot_id": robot_id,
            "source_topic": f"/{robot_id}/vio/eef_pose",
            "translation_unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def _encoder(
    tmp_path: Path,
    robot_id: str,
    timestamps: list[int],
) -> TimeSeriesStream:
    return TimeSeriesStream(
        stream_id=f"{robot_id}_magnetic_encoder",
        modality="magnetic_encoder",
        role="sensor",
        source_path=tmp_path / "source.mcap",
        timestamps_ns=timestamps,
        rows=pd.DataFrame({"raw_value": [0.25] * len(timestamps)}),
        fields=[],
        expected_rate_hz=100.0,
        metadata={
            "robot_id": robot_id,
            "unit": "unknown",
            "semantic_status": "raw_unverified",
        },
    )


def _session(tmp_path: Path) -> Session:
    timestamps0 = [0, 10, 20, 30]
    timestamps1 = [1, 11, 21, 31]
    time_series = [
        _vio(tmp_path, "robot0", timestamps0),
        _vio(tmp_path, "robot1", timestamps1),
        _encoder(tmp_path, "robot0", timestamps0),
        _encoder(tmp_path, "robot1", timestamps1),
    ]
    return Session(
        session_id="umi-provisional-test",
        source_path=str(tmp_path / "source.mcap"),
        meta={},
        video_streams={
            "robot0_camera0": _video("robot0", timestamps0),
            "robot1_camera0": _video("robot1", timestamps1),
        },
        imu_streams={
            "robot0_imu": _imu("robot0", timestamps0),
            "robot1_imu": _imu("robot1", timestamps1),
        },
        time_series_streams={stream.stream_id: stream for stream in time_series},
    )


def test_config_hash_is_stable_and_sensitive_to_effective_config() -> None:
    config_a = {"gap_ns": 100, "nested": {"enabled": True, "name": "UMI"}}
    config_b = {"nested": {"name": "UMI", "enabled": True}, "gap_ns": 100}

    assert deterministic_config_hash(config_a) == deterministic_config_hash(config_b)
    assert deterministic_config_hash(config_a) != deterministic_config_hash(
        {**config_a, "gap_ns": 101}
    )


def test_source_contract_candidate_skips_human_hand_model() -> None:
    contract = build_umi_source_contract_candidate()

    assert contract.formal is False
    assert contract.modalities["human_hand"].applicability == "not_applicable"
    assert contract.modalities["end_effector"].applicability == "applicable"
    assert contract.human_hand_model_action == "skip"
    assert contract.forbidden_hand_reason_codes == ("HAND_ABSENT",)


def test_provisional_metric_validates_fields() -> None:
    config_hash = deterministic_config_hash({})
    metric = UmiProvisionalMetric(
        metric_name="umi_alignment.video.mapped_ratio",
        value=1.0,
        unit="ratio",
        applicability="applicable",
        severity="info",
        disposition="keep",
        reason_code="measurement_only_uncalibrated",
        start_ns=None,
        end_ns=None,
        evidence_uri="umi_provisional_evidence/alignment.parquet",
        producer="test",
        version="1",
        config_hash=config_hash,
        stream_id="video:camera0",
        source_session_id="umi-test",
    )

    assert metric.to_dict()["contract_version"] == "umi-provisional-v1"
    with pytest.raises(ValueError, match="invalid applicability"):
        UmiProvisionalMetric(
            **{**metric.to_dict(), "applicability": "unknown_value"}
        )


def test_evidence_writer_is_versioned_and_not_a_formal_manifest(
    tmp_path: Path,
) -> None:
    bundle = analyze_umi_session(
        _session(tmp_path),
        minimum_gap_ns=100,
        alignment_max_residual_ns=5,
        encoder_freeze_min_samples=3,
    )
    output = tmp_path / "output"
    index = write_umi_evidence_bundle(
        bundle,
        output,
        version="test-v1",
        effective_config={"minimum_gap_ns": 100, "max_residual_ns": 5},
    )

    evidence_root = output / "umi_provisional_evidence"
    payload = json.loads((evidence_root / "evidence_index.json").read_text("utf-8"))
    assert index.formal_manifest is False
    assert payload["formal_manifest"] is False
    assert payload["contract_version"] == "umi-provisional-v1"
    assert not (output / "manifest.json").exists()
    assert any(path.endswith(".parquet") for path in payload["artifacts"].values())
    assert all(
        (output / uri).exists() for uri in payload["artifacts"].values()
    )
    assert all(metric.config_hash == index.config_hash for metric in index.metrics)
    assert any(
        metric.reason_code == "measurement_only_uncalibrated"
        for metric in index.metrics
    )


def test_candidate_views_isolate_vio_from_rgb_observation(tmp_path: Path) -> None:
    session = _session(tmp_path)
    session.time_series_streams["robot1_vio_pose"].rows[
        "source_header_topic"
    ] = "/robot0/vio/eef_pose"
    bundle = analyze_umi_session(
        session,
        minimum_gap_ns=100,
        alignment_max_residual_ns=5,
    )

    views = evaluate_candidate_views(bundle)

    observation = views["robot_observation_ready_candidate"]
    vio = views["vio_ready_candidate"]
    bimanual = views["bimanual_umi_ready_candidate"]
    assert observation.status == "candidate_pass"
    assert observation.disposition == "keep"
    assert vio.status == "review_required"
    assert bimanual.status == "review_required"
    assert all(not view.formal for view in views.values())
    assert all(not view.automatic_reject for view in views.values())


def test_candidate_bimanual_view_is_unavailable_without_second_camera(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    session.video_streams.pop("robot1_camera0")
    views = evaluate_candidate_views(analyze_umi_session(session))

    assert views["robot_observation_ready_candidate"].status == "candidate_pass"
    bimanual = views["bimanual_umi_ready_candidate"]
    assert bimanual.status == "unavailable"
    assert "no_dual_video_alignment" in bimanual.reasons


def test_provisional_pipeline_writes_candidate_delivery_and_preserves_raw(
    tmp_path: Path,
) -> None:
    session = _session(tmp_path)
    raw_path = Path(session.source_path)
    raw_path.write_bytes(b"immutable-umi-raw-fixture")
    output = tmp_path / "delivery"

    result = run_umi_provisional_session(
        session,
        output,
        config={
            "minimum_gap_ns": 100,
            "alignment_max_residual_ns": 5,
            "encoder_freeze_min_samples": 3,
        },
        producer_version="test-v1",
    )

    assert raw_path.read_bytes() == b"immutable-umi-raw-fixture"
    assert result.source_sha256_before == result.source_sha256_after
    assert result.formal_manifest_written is False
    assert result.human_hand_model_invoked is False
    assert not (output / "manifest.json").exists()
    evidence_root = output / "umi_provisional_evidence"
    assert (evidence_root / "source_contract_candidate.json").exists()
    assert (evidence_root / "stage9_hand_applicability_candidate.json").exists()
    assert (evidence_root / "candidate_views.json").exists()
    assert (evidence_root / "provisional_run_summary.json").exists()
    assert (evidence_root / "umi_revision_candidate.json").exists()
    index_payload = json.loads(
        (evidence_root / "evidence_index.json").read_text("utf-8")
    )
    assert index_payload["formal_manifest"] is False
    assert "candidate_views" in index_payload["artifacts"]
    assert "source_contract_candidate" in index_payload["artifacts"]
    assert "provisional_run_summary" in index_payload["artifacts"]
    assert "umi_revision_candidate" in index_payload["artifacts"]
    reason_codes = {metric.reason_code for metric in result.evidence_index.metrics}
    assert "HAND_ABSENT" not in reason_codes
    assert result.hand_applicability.run_human_hand_model is False
    assert result.revision_candidate["formal_manifest"] is False
    assert result.revision_candidate["automatic_reject"] is False
    assert result.revision_candidate["outcome"]["value"] == "unknown"
