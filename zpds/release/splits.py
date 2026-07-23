"""train / val / test split 管理。"""


def generate_splits(segments: list[str], ratios: tuple = (0.7, 0.15, 0.15)) -> dict:
    """按比例生成 split 分配。"""
    raise NotImplementedError
