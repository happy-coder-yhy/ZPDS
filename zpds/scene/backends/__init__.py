"""场景自动分割后端。"""

from zpds.scene.backends.brightness import BrightnessTransitionDetector
from zpds.scene.backends.dino import DinoV2SmallEmbedder
from zpds.scene.backends.histogram import HistogramTransitionDetector
from zpds.scene.backends.optical_flow import OpticalFlowTransitionDetector
from zpds.scene.backends.ssim import SSIMTransitionDetector

__all__ = [
    "BrightnessTransitionDetector",
    "DinoV2SmallEmbedder",
    "HistogramTransitionDetector",
    "OpticalFlowTransitionDetector",
    "SSIMTransitionDetector",
]
