"""向后兼容的 Hands 模型数据结构导入。

共享契约集中在 :mod:`zpds.hands.schemas`。保留本模块作为稳定导入路径，
兼容远端模型后端、Writer 及已有调用方。
"""

from zpds.hands.schemas import (
    HandBBox,
    HandFrameResult,
    HandKeypoints,
    ModelAttemptResult,
    RawHandResult,
)

__all__ = [
    "HandBBox",
    "HandFrameResult",
    "HandKeypoints",
    "ModelAttemptResult",
    "RawHandResult",
]
