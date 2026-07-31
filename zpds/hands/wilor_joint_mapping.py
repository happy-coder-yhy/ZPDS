"""WiLoR → Hands V1 的 21 点关节映射。

阶段 4 核心：将 WiLoR 输出的原始关节映射到 MediaPipe 21 点拓扑，
并逆变换回原图像素坐标。

映射是版本化的——修改映射即改变模型输出版本。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from zpds.hands.schemas import HAND_KEYPOINT_COUNT, RawHandResult
from zpds.hands.wilor_schema import WiLoRDetection, WiLoRImageTransform


# ════════════════════════════════════════════════════════════════════
# 映射版本元数据
# ════════════════════════════════════════════════════════════════════

MAPPING_VERSION = "wilor-to-hands-v1-v1"
"""当前映射版本标识。

修改映射后应同步更新此版本号，
下游通过 config_sha256 + mapping_version 区分不同映射的输出。
"""

# ════════════════════════════════════════════════════════════════════
# WiLoR 原始关节名称（MANO 21 关节顺序）
# ════════════════════════════════════════════════════════════════════

# WiLoR 使用 MANO 手部模型，输出 21 个关节，顺序如下：
WILOR_JOINT_NAMES: tuple[str, ...] = (
    "wrist",           # 0
    "index_mcp",       # 1
    "index_pip",       # 2
    "index_dip",       # 3
    "index_tip",       # 4
    "middle_mcp",      # 5
    "middle_pip",      # 6
    "middle_dip",      # 7
    "middle_tip",      # 8
    "pinky_mcp",       # 9
    "pinky_pip",       # 10
    "pinky_dip",       # 11
    "pinky_tip",       # 12
    "ring_mcp",        # 13
    "ring_pip",        # 14
    "ring_dip",        # 15
    "ring_tip",        # 16
    "thumb_cmc",       # 17
    "thumb_mcp",       # 18
    "thumb_ip",        # 19
    "thumb_tip",       # 20
)

# ════════════════════════════════════════════════════════════════════
# 21 点映射：MANO 顺序 → MediaPipe Hands V1 顺序
# ════════════════════════════════════════════════════════════════════

# WILOR_TO_HANDS_V1_V1[i] = MANO 关节索引，对应 MediaPipe 位置 i。
# 基于 smplx MANO 21 关节拓扑和 MediaPipe Hand Landmarks 标准拓扑。
# 需人工 Preview 验证骨架连线正确。
WILOR_TO_HANDS_V1_V1: tuple[int, ...] = (
    0,   # MP[0]  = wrist        ← MANO[0]
    17,  # MP[1]  = thumb_cmc    ← MANO[17]
    18,  # MP[2]  = thumb_mcp    ← MANO[18]
    19,  # MP[3]  = thumb_ip     ← MANO[19]
    20,  # MP[4]  = thumb_tip    ← MANO[20]
    1,   # MP[5]  = index_mcp    ← MANO[1]
    2,   # MP[6]  = index_pip    ← MANO[2]
    3,   # MP[7]  = index_dip    ← MANO[3]
    4,   # MP[8]  = index_tip    ← MANO[4]
    5,   # MP[9]  = middle_mcp   ← MANO[5]
    6,   # MP[10] = middle_pip   ← MANO[6]
    7,   # MP[11] = middle_dip   ← MANO[7]
    8,   # MP[12] = middle_tip   ← MANO[8]
    13,  # MP[13] = ring_mcp     ← MANO[13]
    14,  # MP[14] = ring_pip     ← MANO[14]
    15,  # MP[15] = ring_dip     ← MANO[15]
    16,  # MP[16] = ring_tip     ← MANO[16]
    9,   # MP[17] = pinky_mcp    ← MANO[9]
    10,  # MP[18] = pinky_pip    ← MANO[10]
    11,  # MP[19] = pinky_dip    ← MANO[11]
    12,  # MP[20] = pinky_tip    ← MANO[12]
)

def _assert_valid_mapping(mapping: tuple[int, ...]) -> None:
    """校验映射合法性。"""
    assert len(mapping) == HAND_KEYPOINT_COUNT, (
        f"映射长度必须为 {HAND_KEYPOINT_COUNT}，实际 {len(mapping)}"
    )
    assert len(set(mapping)) == HAND_KEYPOINT_COUNT, (
        "映射包含重复索引"
    )
    assert all(0 <= idx for idx in mapping), (
        "映射索引不能为负数"
    )


# 模块加载时校验映射
_assert_valid_mapping(WILOR_TO_HANDS_V1_V1)


# ════════════════════════════════════════════════════════════════════
# 关键点坐标逆变换（crop → 原图）
# ════════════════════════════════════════════════════════════════════


def inverse_project_joints_to_original(
    joints_2d: np.ndarray,
    transform: WiLoRImageTransform,
) -> np.ndarray:
    """将 WiLoR 模型输出空间的关键点逆变换到原图像素坐标。

    WiLoR 模型在 crop 区域上运行，输出关节位于 crop 输入坐标系
    （归一化到 [0, 1] 或像素坐标）。本函数将其投影回原始图像空间。

    变换公式::

        if joints in [0, 1] normalized crop space:
            orig_x = crop_x1 + jx * crop_width
            orig_y = crop_y1 + jy * crop_height
        elif joints in pixel crop input space:
            orig_x = crop_x1 + jx / crop_input_width * crop_width
            orig_y = crop_y1 + jy / crop_input_height * crop_height

    Args:
        joints_2d: ``(N, 2)`` 关键点，位于 crop 输入坐标系。
        transform: 完整变换链信息。

    Returns:
        ``(N, 2)`` 原图像素坐标。

    Raises:
        ValueError: crop 参数未设置（全为 0）。
    """
    if joints_2d.ndim != 2 or joints_2d.shape[1] != 2:
        raise ValueError(
            f"joints_2d 形状必须为 (N, 2)，实际 {joints_2d.shape}"
        )

    # 判断坐标空间：全部在 [0, 1] → 归一化 crop 空间
    is_normalized = bool(np.all((joints_2d >= 0.0) & (joints_2d <= 1.0)))

    cx = transform.crop_x1
    cy = transform.crop_y1
    cw = transform.crop_width
    ch = transform.crop_height

    if cw <= 0 or ch <= 0:
        raise ValueError(
            "WiLoRImageTransform 的 crop_width / crop_height 未设置，"
            "无法执行关键点逆变换。"
            "请在 WiLoR 运行后填写 crop 参数。"
        )

    if is_normalized:
        orig_x = cx + joints_2d[:, 0] * cw
        orig_y = cy + joints_2d[:, 1] * ch
    else:
        iw = transform.crop_input_width
        ih = transform.crop_input_height
        if iw <= 0 or ih <= 0:
            raise ValueError(
                "crop_input_width / crop_input_height 未设置，"
                "无法从像素 crop 空间逆变换。"
            )
        orig_x = cx + joints_2d[:, 0] / iw * cw
        orig_y = cy + joints_2d[:, 1] / ih * ch

    return np.column_stack([orig_x, orig_y])


# ════════════════════════════════════════════════════════════════════
# WiLoRDetection → RawHandResult 转换
# ════════════════════════════════════════════════════════════════════


def convert_wilor_to_raw_hand_result(
    detection: WiLoRDetection,
    *,
    mapping: tuple[int, ...] | None = None,
    mapping_version: str = MAPPING_VERSION,
    image_width: int | None = None,
    image_height: int | None = None,
) -> RawHandResult | None:
    """将单个 WiLoRDetection 转换为公共 RawHandResult。

    使用 WiLoR 相机参数（cam_t + focal_length）将 3D 关键点投影到
    2D 像素坐标，然后应用 MANO→MediaPipe 映射。

    Args:
        detection: WiLoR 原始检测结果（须含 cam_t / focal 附加属性）。
        mapping: MANO→MediaPipe 关节映射。为 None 或空时跳过。
        image_width / image_height: 原图尺寸。

    Returns:
        RawHandResult，或 None。
    """
    if not mapping or len(mapping) != HAND_KEYPOINT_COUNT:
        return None

    # 推导图像尺寸
    if image_width is None and detection.transform is not None:
        image_width = detection.transform.original_width
    if image_height is None and detection.transform is not None:
        image_height = detection.transform.original_height
    if image_width is None or image_height is None:
        raise ValueError("无法确定图像尺寸")

    # 使用 WiLoR 相机投影得到 2D 像素关节
    joints_3d = detection.raw_keypoints_3d

    # ---- 投影路径选择 ----
    # Path 1 (preferred): pred_cam + batch context → cam_crop_to_full → project_full_img
    # Path 2 (fallback): pred_cam_t (crop space) → naive pinhole projection
    # Path 3 (last resort): raw_keypoints_2d → inverse_project_joints_to_original

    pred_cam = detection.pred_cam
    box_center = detection.box_center
    box_size = detection.box_size
    sfl = detection.scaled_focal_length

    if (
        joints_3d is not None
        and pred_cam is not None
        and box_center is not None
        and box_size is not None
        and sfl is not None
    ):
        # ---- Path 1: 正确的 crop→full 投影 ----
        if joints_3d.ndim != 2 or joints_3d.shape[0] < HAND_KEYPOINT_COUNT:
            return None

        # 左右手 mirroring（WiLoR demo: joints/verts x-mirror + pred_cam y-mirror）
        if detection.handedness == "Left":
            pred_cam = pred_cam.copy()
            pred_cam[1] *= -1.0                # y-mirror pred_cam
            joints_3d = joints_3d.copy()
            joints_3d[:, 0] *= -1.0             # x-mirror joints/verts

        full_cam_t = _cam_crop_to_full_np(
            pred_cam, box_center, box_size,
            float(image_width), float(image_height), sfl,
        )
        joints_pixel = _project_full_img_np(
            joints_3d, full_cam_t, sfl,
            float(image_width), float(image_height),
        )

    elif joints_3d is not None and detection.cam_t is not None:
        # ---- Path 2: 旧回退路径（pred_cam_t，crop 空间，有偏移） ----
        if joints_3d.ndim != 2 or joints_3d.shape[0] < HAND_KEYPOINT_COUNT:
            return None

        cam_t = detection.cam_t
        focal = detection.focal
        fl = float(focal[0]) if hasattr(focal, "__len__") else float(focal)
        joints_pixel = _project_3d_to_2d(
            joints_3d, cam_t, fl, image_width, image_height,
        )

    else:
        # ---- Path 3: 2D 关键点回退 ----
        joints_2d = detection.raw_keypoints_2d
        if joints_2d is None:
            return None
        if joints_2d.ndim != 2 or joints_2d.shape[0] < HAND_KEYPOINT_COUNT:
            return None
        # raw_keypoints_2d 在模型 crop 空间，需逆变换
        if detection.transform is not None and detection.transform.crop_width > 0:
            joints_pixel = inverse_project_joints_to_original(
                joints_2d, detection.transform
            )
        else:
            joints_pixel = joints_2d

    # 按映射重排
    reordered = np.zeros((HAND_KEYPOINT_COUNT, 2), dtype=np.float64)
    for mp_idx, wilor_idx in enumerate(mapping):
        if wilor_idx < joints_pixel.shape[0]:
            reordered[mp_idx, 0] = joints_pixel[wilor_idx, 0]
            reordered[mp_idx, 1] = joints_pixel[wilor_idx, 1]

    # 归一化坐标
    normalized = np.zeros((HAND_KEYPOINT_COUNT, 3), dtype=np.float64)
    normalized[:, 0] = reordered[:, 0] / image_width
    normalized[:, 1] = reordered[:, 1] / image_height

    # z 值从 3D 关键点获取（如果可用）
    if joints_3d is not None and joints_3d.shape[0] >= HAND_KEYPOINT_COUNT:
        for mp_idx, wilor_idx in enumerate(mapping):
            if wilor_idx < joints_3d.shape[0]:
                normalized[mp_idx, 2] = float(joints_3d[wilor_idx, 2])

    return RawHandResult.from_components(
        handedness=detection.handedness,
        handedness_score=detection.handedness_score,
        detection_score=detection.detection_score,
        normalized_landmarks=normalized,
        image_width=image_width,
        image_height=image_height,
        bbox_xyxy=detection.bbox_xyxy_px,
        label="hand_0",
    )


def _cam_crop_to_full_np(
    cam_bbox: np.ndarray,       # (3,)  pred_cam (s, tx, ty) in crop space
    box_center: np.ndarray,     # (2,)  BBox center in full image (pixels)
    box_size: float,            # scalar  BBox size (with rescale_factor)
    img_w: float,               # full image width
    img_h: float,               # full image height
    focal_length: float,        # scaled_focal_length
) -> np.ndarray:
    """Numpy reimplementation of WiLoR's ``cam_crop_to_full``.

    Converts weak-perspective camera parameters from the crop coordinate
    frame to the full-image coordinate frame.

    Reference: :file:`WiLoR/wilor/utils/renderer.py:12`
    """
    cx, cy = box_center[0], box_center[1]
    b = box_size
    w_2, h_2 = img_w / 2.0, img_h / 2.0
    bs = b * cam_bbox[0] + 1e-9
    tz = 2.0 * focal_length / bs
    tx = (2.0 * (cx - w_2) / bs) + cam_bbox[1]
    ty = (2.0 * (cy - h_2) / bs) + cam_bbox[2]
    return np.array([tx, ty, tz], dtype=np.float64)


def _project_full_img_np(
    points: np.ndarray,         # (N, 3)  3D keypoints in camera space
    cam_trans: np.ndarray,      # (3,)    full-image camera translation
    focal_length: float,        # scaled_focal_length
    img_w: float,               # full image width
    img_h: float,               # full image height
) -> np.ndarray:
    """Numpy reimplementation of WiLoR's ``project_full_img``.

    Projects 3D points from camera space to 2D pixel coordinates
    using a pinhole camera model.

    Reference: :file:`WiLoR/demo.py:134`
    """
    camera_center = np.array([img_w / 2.0, img_h / 2.0], dtype=np.float64)
    points_cam = points + cam_trans  # (N, 3)
    z = points_cam[:, 2]
    z_safe = np.where(np.abs(z) < 1e-9, np.sign(z) * 1e-9, z)
    x_2d = focal_length * points_cam[:, 0] / z_safe + camera_center[0]
    y_2d = focal_length * points_cam[:, 1] / z_safe + camera_center[1]
    return np.column_stack([x_2d, y_2d])


def _project_3d_to_2d(
    keypoints_3d: np.ndarray,
    cam_t: np.ndarray,
    focal_length: float,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    """Fallback: naive pinhole projection using raw ``pred_cam_t``.

    仅在 batch context 不可用时作为回退路径使用。
    注意：此路径未做 crop→full 坐标转换，关键点会有偏移。
    """
    cx = image_width / 2.0
    cy = image_height / 2.0

    points = keypoints_3d + cam_t
    z = points[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, np.sign(z) * 1e-6, z)

    x_2d = focal_length * points[:, 0] / z_safe + cx
    y_2d = focal_length * points[:, 1] / z_safe + cy

    return np.column_stack([x_2d, y_2d])


# ════════════════════════════════════════════════════════════════════
# 映射状态检查
# ════════════════════════════════════════════════════════════════════


def is_mapping_ready(mapping: tuple[int, ...] | None) -> bool:
    """映射是否已就绪（非空且长度为 21）。"""
    return mapping is not None and len(mapping) == HAND_KEYPOINT_COUNT


def get_mapping_summary(mapping: tuple[int, ...] | None) -> dict:
    """返回映射的可打印摘要。"""
    if not mapping:
        return {
            "mapping_version": MAPPING_VERSION,
            "status": "not_ready",
            "reason": "WILOR_TO_HANDS_V1_V1 尚未填写，待 WiLoR 输出确认",
        }
    return {
        "mapping_version": MAPPING_VERSION,
        "status": "ready",
        "joint_count": len(mapping),
        "unique_count": len(set(mapping)),
    }


__all__ = [
    "MAPPING_VERSION",
    "WILOR_JOINT_NAMES",
    "WILOR_TO_HANDS_V1_V1",
    "_cam_crop_to_full_np",
    "_project_full_img_np",
    "convert_wilor_to_raw_hand_result",
    "get_mapping_summary",
    "inverse_project_joints_to_original",
    "is_mapping_ready",
]
