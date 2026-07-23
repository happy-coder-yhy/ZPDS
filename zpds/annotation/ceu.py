"""CEU（Causal Experience Unit）读写。"""


class CeuReader:
    """CEU 读取器。"""

    def read(self, path: str) -> list[dict]:
        raise NotImplementedError


class CeuWriter:
    """CEU 写入器。"""

    def write(self, ceus: list[dict], path: str) -> None:
        raise NotImplementedError
