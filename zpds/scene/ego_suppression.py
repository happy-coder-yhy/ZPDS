"""Ego 运动误报抑制。"""


def suppress_ego_motion(cuts: list[int], imu, threshold: float = 0.5) -> list[int]:
    """过滤由 ego 运动引起的误检切点。"""
    raise NotImplementedError
