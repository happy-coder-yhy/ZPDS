"""ROS2 CDR 编码 MCAP 专用读取。"""


class Ros2McapReader:
    """ROS2 CDR MCAP 读取器。"""

    def __init__(self, path: str):
        self.path = path
