"""操作有效性判断。"""


def is_valid_manipulation(hand_pose: dict, gripper_state: dict | None = None) -> bool:
    """判断当前帧是否存在有效操作。"""
    raise NotImplementedError
