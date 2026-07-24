"""HDF5 数据读取器。"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from zpds.adapters.common import require_file, require_optional_module


class Hdf5Reader:
    """HDF5 数据集读取器。"""

    def __init__(self, path: str):
        self.path = require_file(path)
        self._file: Any = None

    def __enter__(self) -> "Hdf5Reader":  # noqa: PYI034 - Python 3.10 has no typing.Self
        h5py = require_optional_module("h5py", "hdf5")
        self._file = h5py.File(self.path, "r")
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None

    def read_dataset(self, key: str) -> Any:
        """显式读取小型 dataset；大型数据优先使用 iter_dataset。"""
        with self._opened_file() as file:
            return file[key][()]

    def iter_dataset(
        self,
        key: str,
        *,
        chunk_rows: int = 1024,
    ) -> Iterator[Any]:
        if chunk_rows <= 0:
            raise ValueError("chunk_rows must be positive")
        h5py = require_optional_module("h5py", "hdf5")
        with h5py.File(self.path, "r") as file:
            dataset = file[key]
            if dataset.ndim == 0:
                yield dataset[()]
                return
            for start in range(0, dataset.shape[0], chunk_rows):
                yield dataset[start : start + chunk_rows]

    def keys(self) -> tuple[str, ...]:
        keys: list[str] = []
        with self._opened_file() as file:
            file.visititems(
                lambda name, value: keys.append(name)
                if hasattr(value, "shape")
                else None
            )
        return tuple(keys)

    def _opened_file(self):
        h5py = require_optional_module("h5py", "hdf5")
        return h5py.File(Path(self.path), "r")
