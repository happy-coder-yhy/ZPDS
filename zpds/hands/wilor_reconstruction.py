"""WiLoR 3D / MANO 重建（阶段 5：按需运行，非每帧）。

与每帧 2D BBox 检测严格分开：
    - :meth:`WiLoRAdapter.detect` → ``WiLoRDetection``（每帧）
    - :func:`reconstruct` → ``WiLoRReconstructionResult``（仅候选 span）

坐标系约定：未校准前，所有 3D 输出标为 ``model_camera`` + ``uncalibrated``。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRReconstructionResult,
    WiLoRModelInfo,
)


# ════════════════════════════════════════════════════════════════════
# 重投影
# ════════════════════════════════════════════════════════════════════


def reproject(
    keypoints_3d: np.ndarray,
    camera_translation: np.ndarray,
    focal_length: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """将 3D 关键点用针孔相机参数投影回 2D 像素坐标。

    使用 WiLoR 默认相机模型：光心位于图像中心，无畸变。

    Args:
        keypoints_3d: ``(N, 3)`` 相机空间关键点。
        camera_translation: ``(3,)`` 相机平移向量。
        focal_length: 焦距（像素）。
        image_width: 图像宽度（像素）。
        image_height: 图像高度（像素）。

    Returns:
        ``(N, 2)`` 2D 像素坐标。
    """
    cx = image_width / 2.0
    cy = image_height / 2.0

    points_cam = keypoints_3d + camera_translation
    z = points_cam[:, 2]

    # 防止除零
    z_safe = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6, z)

    x_2d = focal_length * points_cam[:, 0] / z_safe + cx
    y_2d = focal_length * points_cam[:, 1] / z_safe + cy

    return np.column_stack([x_2d, y_2d])


def compute_reprojection_error(
    projected_2d: np.ndarray,
    observed_2d: np.ndarray,
) -> float:
    """计算 2D 重投影均方误差（像素）。

    Args:
        projected_2d: ``(N, 2)`` 从 3D 投影回的 2D 坐标。
        observed_2d: ``(N, 2)`` WiLoR 原始 2D 关节（原图坐标）。

    Returns:
        平均 L2 误差（像素）。
    """
    if projected_2d.shape != observed_2d.shape:
        raise ValueError(
            f"形状不匹配: projected {projected_2d.shape}, "
            f"observed {observed_2d.shape}"
        )

    errors = np.linalg.norm(projected_2d - observed_2d, axis=1)
    return float(np.mean(errors))


# ════════════════════════════════════════════════════════════════════
# 重建入口
# ════════════════════════════════════════════════════════════════════


def reconstruct(
    *,
    frame_rgb: np.ndarray,
    detection: WiLoRDetection,
    model_info: WiLoRModelInfo,
    focal_length: float | None = None,
    camera_intrinsics: np.ndarray | None = None,
) -> WiLoRReconstructionResult:
    """对单个检测框执行 WiLoR 3D/MANO 重建。

    仅在候选片段上按需调用，不随 ``detect()`` 每帧运行。

    当前为阶段 5 占位——WiLoR 模型就绪（MANO 文件 + checkpoint）
    后替换为实际推理。

    Args:
        frame_rgb: RGB uint8 原图 ``(H, W, 3)``。
        detection: 对应的 2D 检测结果。
        model_info: WiLoR 运行时元信息。
        focal_length: 可选焦距（像素）。未提供时使用 WiLoR 默认值 5000。
        camera_intrinsics: 可选 ``(3, 3)`` 内参矩阵。未提供时使用针孔模型。

    Returns:
        WiLoRReconstructionResult。失败时 ``pose_valid=False`` + failure_reason。
    """
    if focal_length is None:
        focal_length = 5000.0  # WiLoR EXTRA.FOCAL_LENGTH 默认值

    if camera_intrinsics is not None:
        if camera_intrinsics.shape != (3, 3):
            raise ValueError(
                f"camera_intrinsics 形状必须为 (3, 3)，实际 {camera_intrinsics.shape}"
            )

    # 阶段 5 占位 — WiLoR 模型就绪后替换
    return WiLoRReconstructionResult.failed(
        reason="WiLoR 3D 重建尚未实现（阶段 5 占位，等待 MANO 模型就绪）",
        model_version=model_info.model_version,
        checkpoint_sha256=model_info.checkpoint_sha256,
    )


__all__ = [
    "compute_reprojection_error",
    "reconstruct",
    "reproject",
]
