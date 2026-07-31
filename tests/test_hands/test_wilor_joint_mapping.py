"""WiLoR 21 点映射阶段 4 测试。

验证：
- 映射合法性（长度 21、无重复、索引合法）
- 关键点 crop → 原图逆变换
- 转换守卫：映射未就绪/关节数不对 → None
- 完整转换链：WiLoRDetection → RawHandResult
"""

from __future__ import annotations

import numpy as np
import pytest

from zpds.hands.schemas import HAND_KEYPOINT_COUNT, RawHandResult
from zpds.hands.wilor_joint_mapping import (
    MAPPING_VERSION,
    _assert_valid_mapping,
    convert_wilor_to_raw_hand_result,
    get_mapping_summary,
    inverse_project_joints_to_original,
    is_mapping_ready,
)
from zpds.hands.wilor_schema import (
    WiLoRDetection,
    WiLoRImageTransform,
)


# ════════════════════════════════════════════════════════════════════
# 辅助
# ════════════════════════════════════════════════════════════════════


def _identity_mapping() -> tuple[int, ...]:
    """占位恒等映射（仅测试用，不代表真实语义）。"""
    return tuple(range(HAND_KEYPOINT_COUNT))


def _make_transform(with_crop: bool = False) -> WiLoRImageTransform:
    if with_crop:
        return WiLoRImageTransform(
            original_width=640,
            original_height=480,
            detector_width=256,
            detector_height=256,
            resize_scale_x=0.4,
            resize_scale_y=0.533,
            letterbox_left=0.0,
            letterbox_top=0.0,
            crop_x1=50.0,
            crop_y1=60.0,
            crop_width=200.0,
            crop_height=250.0,
            crop_input_width=256.0,
            crop_input_height=256.0,
        )
    return WiLoRImageTransform.from_resize(
        original_width=640, original_height=480,
        detector_width=256, detector_height=256,
    )


def _make_detection(
    *,
    with_joints: bool = False,
    with_transform: bool = False,
) -> WiLoRDetection:
    joints = None
    if with_joints:
        joints = np.random.default_rng(42).uniform(0, 1, (HAND_KEYPOINT_COUNT, 2)).astype(np.float32)
    return WiLoRDetection(
        handedness="Right",
        handedness_score=0.9,
        detection_score=0.85,
        bbox_xyxy_px=(100.0, 150.0, 300.0, 400.0),
        raw_keypoints_2d=joints,
        raw_keypoint_format="wilor_original" if with_joints else None,
        transform=_make_transform(with_crop=with_transform),
    )


# ════════════════════════════════════════════════════════════════════
# 映射校验
# ════════════════════════════════════════════════════════════════════


def test_identity_mapping_is_valid() -> None:
    _assert_valid_mapping(_identity_mapping())  # 不抛异常


def test_mapping_rejects_wrong_length() -> None:
    with pytest.raises(AssertionError, match="长度"):
        _assert_valid_mapping(tuple(range(20)))


def test_mapping_rejects_duplicates() -> None:
    # 21 个元素但只有 20 个唯一值
    dup = (0, 0) + tuple(range(2, HAND_KEYPOINT_COUNT))  # index 0 twice
    with pytest.raises(AssertionError, match="重复"):
        _assert_valid_mapping(dup)


# ════════════════════════════════════════════════════════════════════
# 关键点坐标逆变换
# ════════════════════════════════════════════════════════════════════


def test_inverse_project_joints_normalized() -> None:
    """归一化 crop 空间 → 原图像素。"""
    transform = _make_transform(with_crop=True)
    # crop 区域: (50, 60) → (250, 310)，输入 256×256
    # 关节在归一化 crop 空间
    joints = np.array([[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]], dtype=np.float32)

    result = inverse_project_joints_to_original(joints, transform)

    # (0,0) → crop 左上角 (50, 60)
    assert result[0, 0] == pytest.approx(50.0)
    assert result[0, 1] == pytest.approx(60.0)
    # (0.5, 0.5) → crop 中心
    assert result[1, 0] == pytest.approx(50.0 + 0.5 * 200.0)
    assert result[1, 1] == pytest.approx(60.0 + 0.5 * 250.0)
    # (1, 1) → crop 右下角
    assert result[2, 0] == pytest.approx(250.0)
    assert result[2, 1] == pytest.approx(310.0)


def test_inverse_project_joints_pixel() -> None:
    """像素 crop 输入空间 → 原图像素。"""
    transform = _make_transform(with_crop=True)
    # crop 输入 256×256，crop 区域在原图 200×250
    joints = np.array([[0.0, 0.0], [128.0, 128.0], [256.0, 256.0]], dtype=np.float32)

    result = inverse_project_joints_to_original(joints, transform)

    # (0, 0) in 256 → crop 左上角
    assert result[0, 0] == pytest.approx(50.0)
    assert result[0, 1] == pytest.approx(60.0)
    # (128, 128) → crop 中心
    assert result[1, 0] == pytest.approx(50.0 + 128.0 / 256.0 * 200.0)
    assert result[1, 1] == pytest.approx(60.0 + 128.0 / 256.0 * 250.0)
    # (256, 256) → crop 右下角
    assert result[2, 0] == pytest.approx(250.0)
    assert result[2, 1] == pytest.approx(310.0)


def test_inverse_project_joints_mixed_with_small_values() -> None:
    """值全在 [0,1] → 按归一化处理。"""
    transform = _make_transform(with_crop=True)
    # 所有值都在 [0, 1] → 判定为归一化空间
    joints = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)

    result = inverse_project_joints_to_original(joints, transform)

    assert result[0, 0] == pytest.approx(50.0 + 0.1 * 200.0)
    assert result[0, 1] == pytest.approx(60.0 + 0.2 * 250.0)


def test_inverse_project_joints_rejects_no_crop() -> None:
    """crop 参数未设置时报错。"""
    transform = _make_transform(with_crop=False)  # crop 全 0
    joints = np.zeros((5, 2), dtype=np.float32)

    with pytest.raises(ValueError, match="crop_width"):
        inverse_project_joints_to_original(joints, transform)


def test_inverse_project_joints_rejects_wrong_shape() -> None:
    transform = _make_transform(with_crop=True)
    with pytest.raises(ValueError, match="形状"):
        inverse_project_joints_to_original(np.zeros((21, 3)), transform)


# ════════════════════════════════════════════════════════════════════
# 转换守卫：不合适的输入 → None
# ════════════════════════════════════════════════════════════════════


def test_convert_returns_none_when_mapping_empty() -> None:
    """映射未填写时返回 None。"""
    detection = _make_detection(with_joints=True, with_transform=True)
    result = convert_wilor_to_raw_hand_result(detection, mapping=())
    assert result is None


def test_convert_returns_none_when_mapping_none() -> None:
    detection = _make_detection(with_joints=True, with_transform=True)
    result = convert_wilor_to_raw_hand_result(detection, mapping=None)
    assert result is None


def test_convert_returns_none_when_no_joints() -> None:
    """WiLoR 未输出关键点 → None。"""
    detection = _make_detection(with_joints=False)
    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=_identity_mapping(),
    )
    assert result is None


def test_convert_returns_none_when_joints_less_than_21() -> None:
    """关节数不足 21 → None，不伪造。"""
    detection = _make_detection(with_joints=False)
    # 手动设置一个只有 18 个关节的数组
    detection.raw_keypoints_2d = np.zeros((18, 2), dtype=np.float32)

    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=_identity_mapping(),
    )
    assert result is None


def test_convert_returns_none_when_mapping_wrong_length() -> None:
    """映射本身长度不是 21 → None。"""
    detection = _make_detection(with_joints=True, with_transform=True)
    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=tuple(range(20)),  # 只有 20
    )
    assert result is None


# ════════════════════════════════════════════════════════════════════
# 完整转换：WiLoRDetection → RawHandResult
# ════════════════════════════════════════════════════════════════════


def test_convert_with_identity_mapping() -> None:
    """恒等映射 + 归一化关节 → 正确 RawHandResult。"""
    detection = _make_detection(with_joints=True, with_transform=True)
    # 每个关节归一化到 [0, 1]
    detection.raw_keypoints_2d = np.array(
        [[i / 21.0, i / 42.0] for i in range(21)], dtype=np.float32
    )

    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=_identity_mapping(),
    )

    assert result is not None
    assert result.handedness == "Right"
    assert result.handedness_score == 0.9
    assert result.detection_score == 0.85
    # BBox = YOLO box ∪ keypoint bounds (kp min from crop x1=50, y1=60)
    assert result.bbox.x1 == 50.0
    assert result.bbox.y1 == 60.0
    assert result.bbox.x2 == 300.0
    assert result.bbox.y2 == 400.0
    assert len(result.keypoints.normalized) == HAND_KEYPOINT_COUNT
    assert len(result.keypoints.pixel) == HAND_KEYPOINT_COUNT
    # 关键点应该在原图范围内
    for px, py in result.keypoints.pixel:
        assert 0.0 <= px < 640.0
        assert 0.0 <= py < 480.0


def test_convert_with_permuted_mapping() -> None:
    """非恒等映射：WiLoR 关节顺序与 MediaPipe 不同。"""
    detection = _make_detection(with_joints=True, with_transform=True)
    # 每个关节有唯一位置标记
    detection.raw_keypoints_2d = np.array(
        [[float(i), 0.0] for i in range(21)], dtype=np.float32
    )
    # 反向映射：WiLoR[20] → MP[0], WiLoR[19] → MP[1], ...
    reverse_mapping = tuple(range(20, -1, -1))

    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=reverse_mapping,
    )

    assert result is not None
    # MP[0] 应来自 WiLoR[20] → x = 20 * (crop transform)
    # 但因为 crop 空间 → 原图，值会变化。验证顺序即可。
    pixels = result.keypoints.pixel
    # 反向映射下，MP 第 0 点的 x 应大于第 20 点的 x
    # (WiLoR[20].x=20 > WiLoR[0].x=0)
    assert pixels[0][0] > pixels[20][0]


def test_convert_preserves_clipped_keypoints() -> None:
    """部分关节超出边界 → any_clipped / clipped_count 记录。

    使用像素 crop 空间坐标，极端负值和超大正值使逆变换后越界。
    """
    detection = _make_detection(with_joints=True, with_transform=True)
    # crop 区域: (50,60) → (250,310)，输入 256×256
    # 要使逆变换后 x < 0: 50 + jx/256*200 < 0 → jx < -64
    # 要使逆变换后 x > 639: 50 + jx/256*200 > 639 → jx > 754
    joints = np.full((21, 2), 128.0, dtype=np.float32)  # 大部分在中间
    joints[0] = [-200.0, -200.0]  # 逆变换后 x ≈ -106, y ≈ -135 → clip 到 0
    joints[20] = [2000.0, 2000.0]  # 逆变换后 x ≈ 1612, y ≈ 2012 → clip 到 w-1

    detection.raw_keypoints_2d = joints

    result = convert_wilor_to_raw_hand_result(
        detection,
        mapping=_identity_mapping(),
    )

    assert result is not None
    assert result.keypoints.any_clipped
    assert result.keypoints.clipped_count >= 2
    # 被裁剪的关键点在图像边界
    assert result.keypoints.pixel[0][0] == 0.0  # clipped to left
    assert result.keypoints.pixel[0][1] == 0.0  # clipped to top


# ════════════════════════════════════════════════════════════════════
# is_mapping_ready / get_mapping_summary
# ════════════════════════════════════════════════════════════════════


def test_is_mapping_ready_none() -> None:
    assert not is_mapping_ready(None)


def test_is_mapping_ready_empty() -> None:
    assert not is_mapping_ready(())


def test_is_mapping_ready_valid() -> None:
    assert is_mapping_ready(_identity_mapping())


def test_is_mapping_ready_wrong_length() -> None:
    assert not is_mapping_ready(tuple(range(20)))


def test_get_mapping_summary_not_ready() -> None:
    summary = get_mapping_summary(())
    assert summary["status"] == "not_ready"
    assert summary["mapping_version"] == MAPPING_VERSION


def test_get_mapping_summary_ready() -> None:
    summary = get_mapping_summary(_identity_mapping())
    assert summary["status"] == "ready"
    assert summary["joint_count"] == 21
    assert summary["unique_count"] == 21
