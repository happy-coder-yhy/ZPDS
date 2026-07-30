"""手部与操作检测。"""

from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig
from zpds.hands.pipeline import (
    HandEstimator,
    HandsPipeline,
    HandsPipelineError,
    PipelineStats,
    PreparedFrameSource,
)
from zpds.hands.schemas import (
    HAND_KEYPOINT_COUNT,
    HandBBox,
    Handedness,
    HandFrameResult,
    HandKeypoints,
    HandObservation,
    ModelAttemptResult,
    PreparedFrame,
    RawHandResult,
)
from zpds.hands.model_router import (
    HandModelRouter,
    create_hand_model_router,
)
from zpds.hands.segment_reader import (
    PreparedSegmentError,
    PreparedSegmentReader,
    SampleMapValidationError,
    StreamNotFoundError,
    VideoDecodeError,
)

__all__ = [
    "HAND_KEYPOINT_COUNT",
    "HandBBox",
    "HandEstimator",
    "HandFrameResult",
    "HandKeypoints",
    "HandModelRouter",
    "HandObservation",
    "Handedness",
    "HandsOutputPaths",
    "HandsPipeline",
    "HandsPipelineConfig",
    "HandsPipelineError",
    "ModelAttemptResult",
    "PipelineStats",
    "PreparedFrame",
    "PreparedFrameSource",
    "PreparedSegmentError",
    "PreparedSegmentReader",
    "RawHandResult",
    "SampleMapValidationError",
    "StreamNotFoundError",
    "VideoDecodeError",
    "create_hand_model_router",
]
