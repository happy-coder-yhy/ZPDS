import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from zpds.hands.backend_router import HandsBackendRouter
from zpds.hands.config import HandsPipelineConfig
from zpds.hands.contracts import (
    FrameInferenceRecord,
    HandEstimator,
    RunFrameStatistics,
)
from zpds.hands.schemas import (
    HandBBox,
    HandKeypoints,
    PreparedFrame,
    RawHandResult,
)
from zpds.hands.testing import (
    FakeBBoxWriter,
    FakeFrameStatusWriter,
    FakeHandEstimator,
)


def _frame() -> PreparedFrame:
    return PreparedFrame(
        frame_rgb=np.zeros((24, 32, 3), dtype=np.uint8),
        output_frame_index=0,
        timestamp_ns=0,
        source_frame_index=10,
        source_timestamp_ns=1_000,
    )


def _hand() -> RawHandResult:
    return RawHandResult(
        handedness="Left",
        handedness_score=0.9,
        keypoints=HandKeypoints(
            normalized=[(0.1, 0.1, 0.0)] * 21,
            pixel=[(3.0, 4.0)] * 21,
        ),
        bbox=HandBBox(1.0, 2.0, 10.0, 12.0),
        detection_score=0.8,
    )


def _parallel_config() -> dict:
    return {
        "hands": {
            "ego_bbox_backend": "wilor",
            "non_ego_bbox_backend": "mediapipe",
            "fallback_2d_backend": "mediapipe",
            "mediapipe": {
                "backend": "solutions_hands",
                "fallback_backend": "solutions_hands",
                "num_hands": 2,
                "tasks": {
                    "model_path": "models/hand_landmarker.task",
                    "delegate": "cpu",
                },
                "solutions": {
                    "model_complexity": 1,
                    "input_mirrored": False,
                },
            },
            "wilor": {
                "enabled": True,
                "ego_bbox_every_frame": True,
                "bbox_fps": 30.0,
                "write_frame_status": True,
                "checkpoint_path": "models/wilor/model.pt",
                "device": "cuda:0",
                "precision": "fp16",
            },
        }
    }


def _write_config(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def test_parallel_config_routes_ego_and_non_ego(tmp_path: Path) -> None:
    config = HandsPipelineConfig.load(
        _write_config(tmp_path, _parallel_config())
    )
    router = HandsBackendRouter(config.backend_policy)

    assert router.select_backend(is_ego=True) == "wilor"
    assert router.select_backend(is_ego=False) == "mediapipe"
    assert router.fallback_2d_backend == "mediapipe"
    assert config.estimator.backend == "solutions_hands"
    assert config.wilor.checkpoint_path == str(
        (tmp_path / "models" / "wilor" / "model.pt").resolve()
    )


def test_backend_override_updates_nested_mediapipe_config_hash(
    tmp_path: Path,
) -> None:
    document = _parallel_config()
    config = HandsPipelineConfig.load(
        _write_config(tmp_path, document),
        backend_override="tasks_hand_landmarker",
    )
    expected = copy.deepcopy(document)
    expected["hands"]["mediapipe"]["backend"] = "tasks_hand_landmarker"

    assert config.estimator.backend == "tasks_hand_landmarker"
    assert config.document == expected


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("ego_bbox_every_frame", False, "ego_bbox_every_frame"),
        ("write_frame_status", False, "write_frame_status"),
    ],
)
def test_wilor_production_guards(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    document = _parallel_config()
    document["hands"]["wilor"][field] = value

    with pytest.raises(ValueError, match=message):
        HandsPipelineConfig.load(_write_config(tmp_path, document))


def test_frame_record_enforces_status_semantics() -> None:
    detected = FrameInferenceRecord(
        frame=_frame(),
        inference_status="detected",
        raw_hands=(_hand(),),
        active_backend="wilor",
        inference_ms=4.2,
    )
    no_hand = FrameInferenceRecord(
        frame=_frame(),
        inference_status="no_hand",
        active_backend="wilor",
    )
    failed = FrameInferenceRecord(
        frame=_frame(),
        inference_status="failed",
        failure_reason="CUDA OOM",
        active_backend="wilor",
    )

    assert len(detected.raw_hands) == 1
    assert no_hand.raw_hands == ()
    assert failed.failure_reason == "CUDA OOM"

    with pytest.raises(ValueError, match="至少包含一只手"):
        FrameInferenceRecord(
            frame=_frame(),
            inference_status="detected",
            active_backend="wilor",
        )
    with pytest.raises(ValueError, match="failure_reason"):
        FrameInferenceRecord(
            frame=_frame(),
            inference_status="failed",
            active_backend="wilor",
        )


def test_fake_estimator_and_writers_cover_stage_one_states() -> None:
    hand = _hand()
    estimator = FakeHandEstimator(
        [[hand], [], RuntimeError("model frame failed")]
    )

    assert isinstance(estimator, HandEstimator)
    assert estimator.estimate(_frame().frame_rgb, 0) == [hand]
    assert estimator.estimate(_frame().frame_rgb, 33) == []
    with pytest.raises(RuntimeError, match="model frame failed"):
        estimator.estimate(_frame().frame_rgb, 66)

    records = [
        FrameInferenceRecord(
            frame=_frame(),
            inference_status="detected",
            raw_hands=(hand,),
            active_backend="wilor",
        ),
        FrameInferenceRecord(
            frame=_frame(),
            inference_status="no_hand",
            active_backend="wilor",
        ),
        FrameInferenceRecord(
            frame=_frame(),
            inference_status="failed",
            failure_reason="model frame failed",
            active_backend="wilor",
        ),
    ]
    status_writer = FakeFrameStatusWriter()
    bbox_writer = FakeBBoxWriter()
    statistics = RunFrameStatistics()
    for record in records:
        status_writer.write(record)
        bbox_writer.write(record)
        statistics.add(record)

    assert len(status_writer.records) == 3
    assert bbox_writer.bbox_count == 1
    assert statistics.to_manifest() == {
        "requested": 3,
        "detected": 1,
        "no_hand": 1,
        "failed": 1,
        "skipped_invalid_input": 0,
    }
    assert statistics.is_complete

    estimator.close()
    status_writer.close()
    bbox_writer.close()
    assert estimator.closed
    assert status_writer.closed
    assert bbox_writer.closed


def test_importing_stage_one_contracts_does_not_import_heavy_models() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zpds.hands.contracts; "
                "assert 'torch' not in sys.modules; "
                "assert 'wilor' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
