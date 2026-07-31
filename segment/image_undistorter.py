"""基于相机内参与畸变系数的 RGB 去畸变。

本模块只处理已解码图像；调用方负责按 Prepared Segment 的写入流程
将其作为派生产物保存，不能覆盖 Raw 图像。
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable
from typing import Any

import cv2
import numpy as np

STANDARD_DISTORTION_MODELS = frozenset(
    {"", "none", "brown_conrady", "plumb_bob", "rational_polynomial"}
)
FISHEYE_DISTORTION_MODELS = frozenset(
    {"equidistant", "fisheye", "kannala_brandt4"}
)


class UndistortionCalibrationError(ValueError):
    """相机标定无法安全用于去畸变。"""


@dataclass(frozen=True)
class UndistortionPlan:
    """单路 Prepared RGB 的几何处置结果。"""

    status: str
    detail: str
    frame_transform: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(frozen=True)
class FrameUndistorter:
    """针对固定分辨率相机的可复用 OpenCV 重映射表。"""

    camera_matrix: np.ndarray
    distortion_coeffs: np.ndarray
    map_x: np.ndarray
    map_y: np.ndarray
    distortion_model: str
    width: int
    height: int

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """返回与输入同尺寸的去畸变图像。"""
        if frame.ndim < 2 or frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "帧分辨率与去畸变标定不一致: "
                f"expected {self.width}x{self.height}, "
                f"got {frame.shape[1] if frame.ndim >= 2 else 0}x"
                f"{frame.shape[0] if frame.ndim >= 2 else 0}"
            )
        return cv2.remap(
            frame,
            self.map_x,
            self.map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )

    def prepared_geometry(self) -> dict[str, Any]:
        """返回应写入 Prepared 标定的派生图像几何。"""
        return {
            "operation": "undistort",
            "source_distortion_model": self.distortion_model or "none",
            "source_distortion_coeffs": self.distortion_coeffs.tolist(),
            "model": "pinhole",
            "distortion_model": "none",
            "intrinsics": {
                "fx": float(self.camera_matrix[0, 0]),
                "fy": float(self.camera_matrix[1, 1]),
                "cx": float(self.camera_matrix[0, 2]),
                "cy": float(self.camera_matrix[1, 2]),
            },
            "resolution": {"width": self.width, "height": self.height},
        }


def build_frame_undistorter(
    camera: dict[str, Any],
    *,
    width: int | None = None,
    height: int | None = None,
) -> FrameUndistorter:
    """从标准化 camera 条目构建去畸变器。

    支持 A2D ``distortion`` 映射以及 MCAP ``D`` 数组。普通针孔模型使用
    :func:`cv2.initUndistortRectifyMap`，``equidistant``/``fisheye`` 使用
    OpenCV fisheye 实现。输出保持原始像素尺寸和内参矩阵，因此图像边缘可能
    出现无效像素；不裁剪 FOV，也不静默缩放标定。
    """
    image_width, image_height = _resolve_image_size(camera, width, height)
    camera_matrix = _camera_matrix(camera)
    model = str(camera.get("distortion_model", "")).strip().lower()
    distortion = _distortion_coefficients(camera, model)

    if model not in STANDARD_DISTORTION_MODELS | FISHEYE_DISTORTION_MODELS:
        raise UndistortionCalibrationError(f"不支持的畸变模型: {model}")

    if model in FISHEYE_DISTORTION_MODELS:
        if len(distortion) != 4:
            raise UndistortionCalibrationError(
                f"{model} 去畸变需要 4 个系数，实际为 {len(distortion)}"
            )
        map_x, map_y = cv2.fisheye.initUndistortRectifyMap(
            camera_matrix,
            distortion.reshape(4, 1),
            np.eye(3),
            camera_matrix,
            (image_width, image_height),
            cv2.CV_32FC1,
        )
    else:
        if len(distortion) not in {4, 5, 8, 12, 14}:
            raise UndistortionCalibrationError(
                "针孔去畸变需要 4、5、8、12 或 14 个系数，"
                f"实际为 {len(distortion)}"
            )
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera_matrix,
            distortion,
            None,
            camera_matrix,
            (image_width, image_height),
            cv2.CV_32FC1,
        )

    return FrameUndistorter(
        camera_matrix=camera_matrix,
        distortion_coeffs=distortion,
        map_x=map_x,
        map_y=map_y,
        distortion_model=model,
        width=image_width,
        height=image_height,
    )


def mark_prepared_undistorted(camera: dict[str, Any], undistorter: FrameUndistorter) -> None:
    """将 Prepared 派生几何写回对应标定条目。"""
    camera["prepared_image_geometry"] = undistorter.prepared_geometry()


def plan_undistortion(
    calibration: dict[str, Any],
    stream_id: str,
    *,
    width: int,
    height: int,
) -> UndistortionPlan:
    """为 Prepared RGB 流解析去畸变处置。

    该函数不会将无标定、模型不支持或分辨率不匹配伪装成无畸变。调用方应将
    返回的 ``status`` 持久化到 ``segment.json``，并只在 ``applied`` 时执行
    ``frame_transform``。
    """
    camera = _find_camera(calibration, stream_id)
    if camera is None:
        return UndistortionPlan("missing_calibration", "camera entry not found")

    try:
        model = str(camera.get("distortion_model", "")).strip().lower()
        coefficients = _distortion_coefficients(camera, model)
        _resolve_image_size(camera, width, height)
    except UndistortionCalibrationError as exc:
        return UndistortionPlan("missing_calibration", str(exc))

    if model not in STANDARD_DISTORTION_MODELS | FISHEYE_DISTORTION_MODELS:
        return UndistortionPlan("unsupported_calibration", f"不支持的畸变模型: {model}")
    if np.allclose(coefficients, 0.0):
        camera["prepared_image_geometry"] = _identity_prepared_geometry(
            camera,
            coefficients,
            width,
            height,
        )
        return UndistortionPlan("identity", "all distortion coefficients are zero")

    try:
        undistorter = build_frame_undistorter(camera, width=width, height=height)
    except UndistortionCalibrationError as exc:
        status = "unsupported_calibration" if "不支持" in str(exc) else "invalid_calibration"
        return UndistortionPlan(status, str(exc))

    mark_prepared_undistorted(camera, undistorter)
    return UndistortionPlan("applied", "undistort", undistorter.apply)


def _resolve_image_size(
    camera: dict[str, Any],
    width: int | None,
    height: int | None,
) -> tuple[int, int]:
    resolution = camera.get("resolution", {})
    if isinstance(resolution, dict):
        declared_width = resolution.get("width")
        declared_height = resolution.get("height")
    elif isinstance(resolution, (list, tuple)) and len(resolution) == 2:
        declared_width, declared_height = resolution
    else:
        declared_width = declared_height = None

    image_width = width if width is not None else declared_width
    image_height = height if height is not None else declared_height
    if not isinstance(image_width, int) or not isinstance(image_height, int):
        raise UndistortionCalibrationError("标定缺少整数分辨率")
    if image_width <= 0 or image_height <= 0:
        raise UndistortionCalibrationError("标定分辨率必须为正数")
    if declared_width and declared_height and (image_width, image_height) != (
        declared_width,
        declared_height,
    ):
        raise UndistortionCalibrationError(
            "图像分辨率与标定声明不一致: "
            f"image={image_width}x{image_height}, calibration={declared_width}x{declared_height}"
        )
    return image_width, image_height


def _find_camera(calibration: dict[str, Any], stream_id: str) -> dict[str, Any] | None:
    for camera in calibration.get("cameras", []):
        if camera.get("stream_id", camera.get("name")) == stream_id:
            return camera
    return None


def _identity_prepared_geometry(
    camera: dict[str, Any],
    coefficients: np.ndarray,
    width: int,
    height: int,
) -> dict[str, Any]:
    camera_matrix = _camera_matrix(camera)
    return {
        "operation": "identity",
        "source_distortion_model": str(camera.get("distortion_model", "none")),
        "source_distortion_coeffs": coefficients.tolist(),
        "model": "pinhole",
        "distortion_model": "none",
        "intrinsics": {
            "fx": float(camera_matrix[0, 0]),
            "fy": float(camera_matrix[1, 1]),
            "cx": float(camera_matrix[0, 2]),
            "cy": float(camera_matrix[1, 2]),
        },
        "resolution": {"width": width, "height": height},
    }


def _camera_matrix(camera: dict[str, Any]) -> np.ndarray:
    intrinsics = camera.get("intrinsics", {})
    try:
        fx = float(intrinsics["fx"])
        fy = float(intrinsics["fy"])
        cx = float(intrinsics["cx"])
        cy = float(intrinsics["cy"])
    except (KeyError, TypeError, ValueError) as exc:
        raise UndistortionCalibrationError("标定缺少 fx/fy/cx/cy") from exc

    if not np.isfinite([fx, fy, cx, cy]).all() or fx <= 0 or fy <= 0:
        raise UndistortionCalibrationError("fx/fy 必须是有限正数，cx/cy 必须有限")
    return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]], dtype=np.float64)


def _distortion_coefficients(camera: dict[str, Any], model: str) -> np.ndarray:
    raw_coefficients = camera.get("D")
    if raw_coefficients is not None:
        coefficients = list(raw_coefficients)
    else:
        distortion = camera.get("distortion")
        if not isinstance(distortion, dict):
            raise UndistortionCalibrationError("标定缺少 D 或 distortion 畸变系数")
        # A2D 原始定义为 k1, k2, k3, p1, p2；OpenCV 针孔模型的顺序为
        # k1, k2, p1, p2, k3，因此必须显式重排。
        if model in FISHEYE_DISTORTION_MODELS:
            coefficients = [distortion.get(f"k{i}") for i in range(1, 5)]
        else:
            coefficients = [
                distortion.get("k1"),
                distortion.get("k2"),
                distortion.get("p1"),
                distortion.get("p2"),
                distortion.get("k3"),
            ]

    try:
        result = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise UndistortionCalibrationError("畸变系数必须是数值数组") from exc
    if not len(result) or not np.isfinite(result).all():
        raise UndistortionCalibrationError("畸变系数不能为空且必须有限")
    return result


__all__ = [
    "FrameUndistorter",
    "UndistortionPlan",
    "UndistortionCalibrationError",
    "build_frame_undistorter",
    "mark_prepared_undistorted",
    "plan_undistortion",
]
