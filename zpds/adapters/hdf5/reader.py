"""HDF5 数据读取器。"""


class Hdf5Reader:
    """HDF5 数据集读取器。"""

    def __init__(self, path: str):
        self.path = path

    def read_dataset(self, key: str):
        """读取指定 dataset。"""
        raise NotImplementedError
