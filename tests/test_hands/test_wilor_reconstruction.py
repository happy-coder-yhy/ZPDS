"""WiLoR 3D 重建阶段 5 测试。

验证：
- 重投影计算
- 重投影误差
- WiLoRReconstructionResult 工厂方法
- 2D/3D 接口分离
- 坐标系默认值和尺度标记
"""

from __future__ import annotations

import numpy as np
import pytest

from zpds.hands.wilor_reconstruction import (
    compute_reprojection_error,
    reconstruct,
    reproject,
)
from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRImageTransform,
    WiLoRModelInfo,
    WiLoRReconstructionResult,
)


# ════════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════════


def _make_info() -> WiLoRModelInfo:
    return WiLoRModelInfo(
        model_version="v1.0",
        checkpoint_sha256="a" * 64,
        device="cpu",
    )


def _make_detection() -> WiLoRDetection:
    return WiLoRDetection(
        handedness="Right",
        handedness_score=0.9,
        detection_score=0.85,
        bbox_xyxy_px=(100.0, 150.0, 300.0, 400.0),
    )


# ════════════════════════════════════════════════════════════════════
# WiLoRReconstructionResult — 工厂方法
# ════════════════════════════════════════════════════════════════════


def test_not_attempted_defaults() -> None:
    result = WiLoRReconstructionResult.not_attempted()
    assert not result.reconstruction_attempted
    assert not result.pose_valid
    assert result.coordinate_frame == "model_camera"
    assert result.scale_status == "uncalibrated"


def test_not_attempted_with_provenance() -> None:
    result = WiLoRReconstructionResult.not_attempted(
        model_version="v2.0",
        checkpoint_sha256="abc123",
    )
    assert result.model_version == "v2.0"
    assert result.checkpoint_sha256 == "abc123"


def test_failed_preserves_2d_detection_info() -> None:
    """3D 失败不应牵连 2D — pose_valid=False 但结构完整。"""
    result = WiLoRReconstructionResult.failed(
        reason="MANO optimization diverged",
        model_version="v1.0",
        checkpoint_sha256="def456",
    )
    assert result.reconstruction_attempted
    assert not result.pose_valid
    assert result.failure_reason == "MANO optimization diverged"
    assert result.model_version == "v1.0"


def test_default_reconstruction_is_model_camera_uncalibrated() -> None:
    """默认坐标系和尺度标记必须保守。"""
    result = WiLoRReconstructionResult()
    assert result.coordinate_frame == "model_camera"
    assert result.scale_status == "uncalibrated"
    assert not result.pose_valid


# ════════════════════════════════════════════════════════════════════
# 重投影
# ════════════════════════════════════════════════════════════════════


def test_reproject_identity() -> None:
    """相机平移 (0,0,1)，焦距 1，关键点在 z=0 平面 → 投影到图像中心。"""
    kp3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    cam_t = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    proj = reproject(kp3d, cam_t, focal_length=100.0, image_width=200, image_height=200)

    # (0,0,0) + (0,0,1) = (0,0,1) → proj = (fx*0/1 + cx, fy*0/1 + cy) = (100, 100)
    assert proj[0, 0] == pytest.approx(100.0)
    assert proj[0, 1] == pytest.approx(100.0)
    # (1,0,0) + (0,0,1) = (1,0,1) → proj = (100*1/1 + 100, 100*0/1 + 100) = (200, 100)
    assert proj[1, 0] == pytest.approx(200.0)


def test_reproject_nonzero_translation() -> None:
    kp3d = np.array([[0.0, 0.0, 0.0]], dtype=np.float32)
    cam_t = np.array([0.0, 0.0, 5.0], dtype=np.float32)

    proj = reproject(kp3d, cam_t, focal_length=500.0, image_width=640, image_height=480)
    assert proj[0, 0] == pytest.approx(320.0)
    assert proj[0, 1] == pytest.approx(240.0)


def test_reproject_output_shape() -> None:
    kp3d = np.random.default_rng(42).uniform(-0.5, 0.5, (21, 3)).astype(np.float32)
    cam_t = np.array([0.1, -0.1, 0.8], dtype=np.float32)

    proj = reproject(kp3d, cam_t, focal_length=5000.0, image_width=1920, image_height=1080)
    assert proj.shape == (21, 2)


# ════════════════════════════════════════════════════════════════════
# 重投影误差
# ════════════════════════════════════════════════════════════════════


def test_compute_reprojection_error_zero() -> None:
    """完全匹配 → 误差为 0。"""
    pts = np.array([[100.0, 200.0], [300.0, 400.0]], dtype=np.float32)
    assert compute_reprojection_error(pts, pts) == 0.0


def test_compute_reprojection_error_known_offset() -> None:
    projected = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)
    observed = np.array([[3.0, 0.0], [0.0, 4.0]], dtype=np.float32)

    error = compute_reprojection_error(projected, observed)
    # mean([3, 4]) = 3.5
    assert error == pytest.approx(3.5)


def test_compute_reprojection_error_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="形状不匹配"):
        compute_reprojection_error(
            np.zeros((10, 2)),
            np.zeros((21, 2)),
        )


# ════════════════════════════════════════════════════════════════════
# reconstruct() 占位
# ════════════════════════════════════════════════════════════════════


def test_reconstruct_stage5_stub() -> None:
    """阶段 5 占位：返回 .failed() 但 2D 检测信息独立保留。"""
    detection = _make_detection()
    info = _make_info()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = reconstruct(
        frame_rgb=frame,
        detection=detection,
        model_info=info,
    )

    assert result.reconstruction_attempted
    assert not result.pose_valid
    assert "尚未实现" in (result.failure_reason or "")
    assert result.model_version == "v1.0"
    assert result.checkpoint_sha256 == "a" * 64
    # 2D 检测不受影响
    assert detection.handedness == "Right"


def test_reconstruct_default_focal_length() -> None:
    """未提供焦距时使用 WiLoR 默认值 5000。"""
    detection = _make_detection()
    info = _make_info()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = reconstruct(frame_rgb=frame, detection=detection, model_info=info)
    assert isinstance(result, WiLoRReconstructionResult)


def test_reconstruct_with_custom_focal_length() -> None:
    detection = _make_detection()
    info = _make_info()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    result = reconstruct(
        frame_rgb=frame,
        detection=detection,
        model_info=info,
        focal_length=3000.0,
    )
    assert isinstance(result, WiLoRReconstructionResult)


def test_reconstruct_with_intrinsics() -> None:
    detection = _make_detection()
    info = _make_info()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    K = np.eye(3)
    result = reconstruct(
        frame_rgb=frame,
        detection=detection,
        model_info=info,
        camera_intrinsics=K,
    )
    assert isinstance(result, WiLoRReconstructionResult)


def test_reconstruct_rejects_bad_intrinsics_shape() -> None:
    detection = _make_detection()
    info = _make_info()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="形状"):
        reconstruct(
            frame_rgb=frame,
            detection=detection,
            model_info=info,
            camera_intrinsics=np.eye(4),
        )
