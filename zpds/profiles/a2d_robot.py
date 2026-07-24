"""A2D 真机 profile。"""

from .base import BaseProfile


class A2DRobotProfile(BaseProfile):
    """A2D 真机 episodic 数据 profile。"""

    def __init__(self):
        super().__init__(
            name="a2d_robot",
            description="A2D 真机：3 相机 JPEG/PNG + HDF5 + ROS2 MCAP joint/gripper",
            adapter_kind="a2d",
            required_globs=("meta_info.json", "camera/*/*"),
            optional_globs=("*.h5", "record/**/*.mcap", "logs/**/*"),
            metadata={
                "camera_roles": ("head", "hand_left", "hand_right"),
                "modalities": ("color", "depth"),
                "frame_mapping": "inferred_until_proven",
            },
        )
