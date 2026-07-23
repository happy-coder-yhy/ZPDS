"""EPIC-KITCHENS-100 profile。"""

from .base import BaseProfile


class Epic100Profile(BaseProfile):
    """EPIC-KITCHENS-100 衍生标注 profile。"""

    def __init__(self):
        super().__init__(
            name="epic100",
            description="EPIC-KITCHENS-100：Mask R-CNN 实例 mask + hand-object pickle 标注",
        )
