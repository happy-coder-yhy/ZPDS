"""G16 相机内参去畸变测试。"""

from __future__ import annotations

import numpy as np
import pytest

from segment.image_undistorter import (
    UndistortionCalibrationError,
    build_frame_undistorter,
    mark_prepared_undistorted,
    plan_undistortion,
)


def _standard_camera() -> dict:
    return {
        "stream_id": "head_rgb",
        "distortion_model": "brown_conrady",
        "intrinsics": {"fx": 70.0, "fy": 71.0, "cx": 32.0, "cy": 24.0},
        "distortion": {"k1": 0.10, "k2": -0.02, "k3": 0.003, "p1": 0.01, "p2": -0.01},
        "resolution": {"width": 64, "height": 48},
    }


def test_standard_undistorter_reorders_a2d_coefficients_and_preserves_shape() -> None:
    undistorter = build_frame_undistorter(_standard_camera())
    frame = np.zeros((48, 64, 3), dtype=np.uint8)
    frame[:, 32:] = 255

    output = undistorter.apply(frame)

    assert undistorter.distortion_coeffs.tolist() == [0.10, -0.02, 0.01, -0.01, 0.003]
    assert output.shape == frame.shape
    assert output.dtype == np.uint8
    assert not np.array_equal(output, frame)


def test_fisheye_undistorter_uses_four_coefficients() -> None:
    camera = {
        "stream_id": "robot0_camera0",
        "distortion_model": "equidistant",
        "intrinsics": {"fx": 90.0, "fy": 90.0, "cx": 20.0, "cy": 15.0},
        "D": [0.01, -0.002, 0.0, 0.0],
        "resolution": [40, 30],
    }
    undistorter = build_frame_undistorter(camera)

    output = undistorter.apply(np.full((30, 40, 3), 120, dtype=np.uint8))

    assert output.shape == (30, 40, 3)
    assert undistorter.prepared_geometry()["distortion_model"] == "none"


def test_mark_prepared_undistorted_records_derived_geometry() -> None:
    camera = _standard_camera()
    undistorter = build_frame_undistorter(camera)

    mark_prepared_undistorted(camera, undistorter)

    assert camera["prepared_image_geometry"] == {
        "operation": "undistort",
        "source_distortion_model": "brown_conrady",
        "source_distortion_coeffs": [0.1, -0.02, 0.01, -0.01, 0.003],
        "model": "pinhole",
        "distortion_model": "none",
        "intrinsics": {"fx": 70.0, "fy": 71.0, "cx": 32.0, "cy": 24.0},
        "resolution": {"width": 64, "height": 48},
    }


@pytest.mark.parametrize(
    ("camera", "message"),
    [
        ({"intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0}}, "分辨率"),
        (
            {
                "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "resolution": {"width": 4, "height": 4},
            },
            "畸变系数",
        ),
        (
            {
                "distortion_model": "unsupported",
                "intrinsics": {"fx": 1, "fy": 1, "cx": 0, "cy": 0},
                "D": [0, 0, 0, 0],
                "resolution": {"width": 4, "height": 4},
            },
            "不支持",
        ),
    ],
)
def test_invalid_calibration_is_rejected(camera: dict, message: str) -> None:
    with pytest.raises(UndistortionCalibrationError, match=message):
        build_frame_undistorter(camera)


def test_frame_resolution_mismatch_is_rejected() -> None:
    undistorter = build_frame_undistorter(_standard_camera())

    with pytest.raises(ValueError, match="分辨率与去畸变标定不一致"):
        undistorter.apply(np.zeros((47, 64, 3), dtype=np.uint8))


def test_plan_applies_a2d_calibration_with_opencv_coefficient_order() -> None:
    calibration = {"cameras": [_standard_camera()]}

    plan = plan_undistortion(calibration, "head_rgb", width=64, height=48)

    assert plan.status == "applied"
    assert plan.frame_transform is not None
    assert calibration["cameras"][0]["prepared_image_geometry"]["source_distortion_coeffs"] == [
        0.1,
        -0.02,
        0.01,
        -0.01,
        0.003,
    ]


def test_plan_marks_zero_dunjia_calibration_as_identity() -> None:
    camera = {
        "stream_id": "camera0",
        "distortion_model": "plumb_bob",
        "intrinsics": {"fx": 90.0, "fy": 90.0, "cx": 20.0, "cy": 15.0},
        "D": [0.0, 0.0, 0.0, 0.0],
        "resolution": [40, 30],
    }

    plan = plan_undistortion({"cameras": [camera]}, "camera0", width=40, height=30)

    assert plan.status == "identity"
    assert plan.frame_transform is None
    assert camera["prepared_image_geometry"]["operation"] == "identity"


def test_plan_applies_umi_equidistant_calibration() -> None:
    camera = {
        "stream_id": "robot0_camera0",
        "distortion_model": "equidistant",
        "intrinsics": {"fx": 90.0, "fy": 90.0, "cx": 20.0, "cy": 15.0},
        "D": [0.01, -0.002, 0.0, 0.001],
        "resolution": [40, 30],
    }

    plan = plan_undistortion({"cameras": [camera]}, "robot0_camera0", width=40, height=30)

    assert plan.status == "applied"
    assert plan.frame_transform is not None


@pytest.mark.parametrize(
    ("camera", "stream_id", "width", "height", "status"),
    [
        (
            {
                "stream_id": "ego_rgb",
                "intrinsics": {"fx": 90.0, "fy": 90.0, "cx": 20.0, "cy": 15.0},
                "resolution": [40, 30],
            },
            "ego_rgb",
            40,
            30,
            "missing_calibration",
        ),
        (
            {
                "stream_id": "ego_rgb",
                "distortion_model": "unknown_model",
                "intrinsics": {"fx": 90.0, "fy": 90.0, "cx": 20.0, "cy": 15.0},
                "D": [0.1, 0.0, 0.0, 0.0],
                "resolution": [40, 30],
            },
            "ego_rgb",
            40,
            30,
            "unsupported_calibration",
        ),
        (_standard_camera(), "head_rgb", 63, 48, "missing_calibration"),
    ],
)
def test_plan_records_non_applicable_calibration(
    camera: dict,
    stream_id: str,
    width: int,
    height: int,
    status: str,
) -> None:
    plan = plan_undistortion({"cameras": [camera]}, stream_id, width=width, height=height)

    assert plan.status == status
    assert plan.frame_transform is None
