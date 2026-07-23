"""BaseProfile — profile 基类。"""

from dataclasses import dataclass


@dataclass
class BaseProfile:
    """采集源 profile 基类。"""

    name: str
    description: str = ""
