"""MCAP 消息解码、schema 解析。"""


class McapReader:
    """MCAP 消息读取器。"""

    def __init__(self, path: str):
        self.path = path

    def iter_messages(self, topic: str | None = None):
        """按 topic 迭代消息。"""
        raise NotImplementedError

    def topics(self) -> list[str]:
        """列出所有 topic。"""
        raise NotImplementedError
