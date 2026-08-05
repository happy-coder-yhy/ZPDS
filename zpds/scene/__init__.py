"""ZPDS 场景自动分割与语义复核契约。"""

from zpds.scene.config import SceneConfig
from zpds.scene.fusion import SceneBoundaryFusion, StageATransitionFusion
from zpds.scene.pipeline import ScenePipelineRun, run_scene_pipeline
from zpds.scene.preview import write_scene_previews
from zpds.scene.qc_integration import (
    build_scene_decisions,
    build_scene_metrics,
)
from zpds.scene.sampling import representative_frame_indices
from zpds.scene.schemas import (
    BoundaryScore,
    SceneProposal,
    TransitionProposal,
    VLMReviewResult,
)
from zpds.scene.validator import (
    SceneValidationReport,
    sha256_file,
    validate_scene_outputs,
)
from zpds.scene.vlm_review import (
    OpenAICompatibleVLMReviewer,
    SceneLabels,
    VLMUnavailableError,
    load_scene_labels,
    select_review_queue,
)
from zpds.scene.writer import SceneWriteResult, write_scene_run

__all__ = [
    "BoundaryScore",
    "OpenAICompatibleVLMReviewer",
    "SceneBoundaryFusion",
    "SceneConfig",
    "SceneLabels",
    "ScenePipelineRun",
    "SceneProposal",
    "SceneValidationReport",
    "SceneWriteResult",
    "StageATransitionFusion",
    "TransitionProposal",
    "VLMReviewResult",
    "VLMUnavailableError",
    "build_scene_decisions",
    "build_scene_metrics",
    "load_scene_labels",
    "representative_frame_indices",
    "run_scene_pipeline",
    "select_review_queue",
    "sha256_file",
    "validate_scene_outputs",
    "write_scene_previews",
    "write_scene_run",
]
