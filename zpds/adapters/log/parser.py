"""设备日志解析器。"""


class LogParser:
    """录制/设备日志解析器。"""

    def parse(self, path: str) -> dict:
        """解析日志并返回结构化事件。"""
        raise NotImplementedError
