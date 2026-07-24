"""HDF5/Zarr 适配器。"""

from .inspector import Hdf5Inspector
from .reader import Hdf5Reader

__all__ = ["Hdf5Inspector", "Hdf5Reader"]
