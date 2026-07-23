"""大型标注资产管理：masks / poses / tracks / pointclouds。"""


class AssetStore:
    """标注资产存储抽象。"""

    def put(self, key: str, data, format: str = "zarr") -> str:
        raise NotImplementedError

    def get(self, key: str):
        raise NotImplementedError
