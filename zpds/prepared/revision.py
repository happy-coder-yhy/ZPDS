"""revision.json 管理。"""


class RevisionManager:
    """修订版本管理器。"""

    def create(self, notes: str = "") -> str:
        """创建新 revision，返回 revision_id（如 r0002）。"""
        raise NotImplementedError

    def latest(self) -> str:
        """获取最新 revision_id。"""
        raise NotImplementedError
