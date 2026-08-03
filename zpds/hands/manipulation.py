"""操作有效性判断。"""

from __future__ import annotations


def is_valid_manipulation(
    hand_pose: dict,
    gripper_state: dict | None = None,
    *,
    hand_speed_threshold: float = 0.01,
    gripper_delta_threshold: float = 0.01,
) -> bool:
    """根据手部运动或夹爪状态变化判断是否存在操作。

    阈值单位分别为归一化图像对角线/秒和夹爪状态自身单位。本函数只判断
    单帧证据，持续时间聚合由视频清洗器完成。
    """
    if hand_speed_threshold < 0 or gripper_delta_threshold < 0:
        raise ValueError("操作阈值不能为负数")
    hand_speed = hand_pose.get(
        "center_speed_normalized_per_s",
        hand_pose.get("hand_center_speed_normalized_per_s", 0.0),
    )
    try:
        hand_moved = abs(float(hand_speed)) > hand_speed_threshold
    except (TypeError, ValueError):
        hand_moved = False
    if gripper_state is None:
        return hand_moved
    delta = gripper_state.get("delta", gripper_state.get("state_delta", 0.0))
    try:
        gripper_changed = abs(float(delta)) > gripper_delta_threshold
    except (TypeError, ValueError):
        gripper_changed = False
    return hand_moved or gripper_changed
