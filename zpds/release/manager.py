"""Release 管理器。"""


class ReleaseManager:
    """Release JSON 读写。"""

    def create(self, prep_revision: str, exp_version: str, notes: str = "") -> str:
        """创建 release，返回 release_id。"""
        raise NotImplementedError

    def load(self, release_id: str) -> dict:
        raise NotImplementedError
