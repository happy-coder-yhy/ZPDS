"""A2D 真机 profile。"""

from typing import ClassVar

from .base import BaseProfile


class A2DRobotProfile(BaseProfile):
    """A2D 真机 episodic 数据 profile。"""

    # 必需流：所有必需流时间覆盖的交集构成公共有效范围
    REQUIRED_STREAMS: ClassVar[list[str]] = [
        "head_rgb",
        "hand_left_rgb",
        "hand_right_rgb",
        "robot_state",
        "robot_action",
    ]

    # 可选流：存在时纳入，不存在时不阻塞
    OPTIONAL_STREAMS: ClassVar[list[str]] = [
        "gripper_state",
        "gripper_action",
    ]

    def __init__(self):
        super().__init__(
            name="a2d_robot",
            description="A2D 真机：3 相机 JPEG/PNG + HDF5 + ROS2 MCAP joint/gripper",
            modalities={
                "human_hand": "not_applicable",
                "end_effector": "applicable",
            },
            primary_stream_id="head_rgb",
        )

    @property
    def required_streams(self) -> list[str]:
        return self.REQUIRED_STREAMS

    @property
    def optional_streams(self) -> list[str]:
        return self.OPTIONAL_STREAMS
