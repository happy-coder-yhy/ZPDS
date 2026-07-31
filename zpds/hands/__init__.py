"""手部与操作检测。"""

from zpds.hands.backend_router import HandsBackendPolicy, HandsBackendRouter
from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig, WilorConfig
from zpds.hands.contracts import (
    BBoxWriter,
    FrameInferenceRecord,
    FrameStatusWriter,
    HandEstimator,
    InferenceStatus,
    RunFrameStatistics,
)
from zpds.hands.estimator_factory import (
    EstimatorRuntime,
    EstimatorUnavailableError,
    create_hand_estimator,
    validate_estimator_runtime,
)
from zpds.hands.pipeline import (
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
from zpds.hands.wilor_preflight import (
    WilorAssetCheck,
    WilorPreflightReport,
    check_wilor_assets,
)

__all__ = [
    "HAND_KEYPOINT_COUNT",
    "BBoxWriter",
    "EstimatorRuntime",
    "EstimatorUnavailableError",
    "FrameInferenceRecord",
    "FrameStatusWriter",
    "HandBBox",
    "HandEstimator",
    "HandFrameResult",
    "HandKeypoints",
    "HandModelRouter",
    "HandObservation",
    "Handedness",
    "HandsBackendPolicy",
    "HandsBackendRouter",
    "HandsOutputPaths",
    "HandsPipeline",
    "HandsPipelineConfig",
    "HandsPipelineError",
    "ModelAttemptResult",
    "InferenceStatus",
    "PipelineStats",
    "PreparedFrame",
    "PreparedFrameSource",
    "PreparedSegmentError",
    "PreparedSegmentReader",
    "RawHandResult",
    "RunFrameStatistics",
    "SampleMapValidationError",
    "StreamNotFoundError",
    "VideoDecodeError",
    "create_hand_model_router",
    "WilorAssetCheck",
    "WilorConfig",
    "WilorPreflightReport",
    "check_wilor_assets",
    "create_hand_estimator",
    "validate_estimator_runtime",
]
