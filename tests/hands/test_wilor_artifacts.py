"""WiLoR Writer/Validator artifact contracts without optional model dependencies."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from zpds.hands.schemas import HandObservation
from zpds.hands.validator import validate_wilor_hands
from zpds.hands.writer import wilor_provenance, write_hand_observations


@dataclass
class _ModelInfo:
    model_version: str = "wilor_cvpr2025"
    checkpoint_sha256: str = "checkpoint-sha"
    device: str = "cuda:0"


@dataclass
class _FrameStats:
    total_frames: int = 10
    detected: int = 9
    no_hand: int = 0
    failed: int = 1
    skipped_invalid_input: int = 0
    fallback_attempted: int = 1
    fallback_used: int = 1
    total_inference_ms: float = 20.0


class _RunReport:
    def to_dict(self) -> dict:
        return {
            "model": {
                "name": "wilor",
                "version": "wilor_cvpr2025",
                "checkpoint_sha256": "checkpoint-sha",
                "device": "cuda:0",
            },
            "coverage": {"decoded_frames": 10, "failed_frames": 1},
            "errors": [{"failure_reason": "synthetic WiLoR frame failure"}],
        }


class _Estimator:
    model_info = _ModelInfo()
    frame_stats = _FrameStats()

    def build_run_report(self) -> _RunReport:
        return _RunReport()


def _observation(*, fallback: bool = False) -> HandObservation:
    return HandObservation(
        segment_id="seg_000001",
        video_stream_id="ego_rgb",
        output_frame_index=0,
        timestamp_ns=0,
        source_frame_index=0,
        source_timestamp_ns=0,
        detection_id=0,
        handedness="right",
        handedness_score=0.9,
        bbox_xyxy=(100.0, 100.0, 300.0, 350.0),
        keypoints_2d=[(110.0 + index * 4, 120.0 + index * 4) for index in range(21)],
        keypoints_z_relative=[0.0] * 21,
        model_name="mediapipe" if fallback else "wilor",
        model_version="hand_landmarker_v1" if fallback else "wilor_cvpr2025",
        backend_requested="wilor",
        backend_active="mediapipe" if fallback else "wilor",
        backend_fallback_used=fallback,
        backend_fallback_reason="WiLoR frame failure" if fallback else "",
    )


def _run_report(*, failed: int = 0, total: int = 10) -> dict:
    return {
        "model": {
            "name": "wilor",
            "version": "wilor_cvpr2025",
            "checkpoint_sha256": "checkpoint-sha",
            "device": "cuda:0",
        },
        "coverage": {"decoded_frames": total, "failed_frames": failed},
    }


def test_wilor_provenance_serializes_model_stats_and_fallback_reason() -> None:
    provenance, report = wilor_provenance(_Estimator(), {"hands": {"backend": "wilor"}})

    assert provenance["model_name"] == "wilor"
    assert provenance["checkpoint_sha256"] == "checkpoint-sha"
    assert provenance["backend_fallback_used"] is True
    assert provenance["backend_fallback_reason"] == "synthetic WiLoR frame failure"
    assert report["session_statistics"]["fallback_used"] == 1


def test_wilor_validator_accepts_pixel_provenance_attribution(tmp_path: Path) -> None:
    parquet_path = write_hand_observations(
        [_observation(), _observation()],
        tmp_path / "hands_2d.parquet",
        checkpoint_sha256="checkpoint-sha",
        config_sha256="config-sha",
    )
    report_path = tmp_path / "hands_run.json"
    report_path.write_text(json.dumps(_run_report()), encoding="utf-8")

    result = validate_wilor_hands(
        parquet_path,
        hands_run_path=str(report_path),
        image_width=640,
        image_height=480,
    )

    assert result["status"] == "pass", result
    assert result["checks"]["pixel_inverse_transform"] == "pass"
    assert result["checks"]["reprojection_consistency"] == "pass"
    assert result["checks"]["wilor_provenance"] == "pass"
    assert result["checks"]["backend_attribution"] == "pass"


def test_wilor_validator_rejects_out_of_bounds_points_and_missing_fallback_reason(
    tmp_path: Path,
) -> None:
    parquet_path = write_hand_observations(
        [_observation(fallback=True)],
        tmp_path / "hands_2d.parquet",
        checkpoint_sha256="checkpoint-sha",
        config_sha256="config-sha",
    )
    frame = pd.read_parquet(parquet_path)
    frame.at[0, "keypoints_2d"] = [[700.0, 120.0]] * 21
    frame.at[0, "backend_fallback_reason"] = ""
    frame.to_parquet(parquet_path, index=False)
    report_path = tmp_path / "hands_run.json"
    report_path.write_text(json.dumps(_run_report()), encoding="utf-8")

    result = validate_wilor_hands(
        parquet_path,
        hands_run_path=str(report_path),
        image_width=640,
        image_height=480,
    )

    assert result["status"] == "fail"
    assert result["checks"]["pixel_inverse_transform"] == "fail"
    assert result["checks"]["backend_attribution"] == "fail"


def test_wilor_validator_enforces_failure_ratio(tmp_path: Path) -> None:
    parquet_path = write_hand_observations(
        [_observation()],
        tmp_path / "hands_2d.parquet",
        checkpoint_sha256="checkpoint-sha",
        config_sha256="config-sha",
    )
    report_path = tmp_path / "hands_run.json"
    report_path.write_text(json.dumps(_run_report(failed=1, total=10)), encoding="utf-8")

    result = validate_wilor_hands(
        parquet_path,
        hands_run_path=str(report_path),
        image_width=640,
        image_height=480,
    )

    assert result["checks"]["failure_records"] == "fail"
