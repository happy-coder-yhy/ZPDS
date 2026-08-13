"""手部与操作检测。"""

from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig, WilorConfig
from zpds.hands.contracts import (
    BBoxWriter,
    FrameInferenceRecord,
    FrameStatusHandEstimator,
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
from zpds.hands.frame_artifacts import (
    InferenceArtifactContext,
    ParquetBBoxWriter,
    ParquetFrameStatusWriter,
    validate_wilor_frame_artifacts,
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
    "FrameStatusHandEstimator",
    "FrameStatusWriter",
    "HandBBox",
    "HandEstimator",
    "HandFrameResult",
    "HandKeypoints",
    "HandObservation",
    "Handedness",
    "HandsOutputPaths",
    "HandsPipeline",
    "HandsPipelineConfig",
    "HandsPipelineError",
    "InferenceArtifactContext",
    "InferenceStatus",
    "ModelAttemptResult",
    "ParquetBBoxWriter",
    "ParquetFrameStatusWriter",
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
    "WilorAssetCheck",
    "WilorConfig",
    "WilorPreflightReport",
    "check_wilor_assets",
    "create_hand_estimator",
    "validate_estimator_runtime",
    "validate_wilor_frame_artifacts",
]
