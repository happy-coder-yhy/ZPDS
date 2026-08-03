"""A2D 真机检测器。

覆盖 B6–B9：
  - B6: 完整性矩阵 (completeness matrix)
  - B7: 相机—机器人对齐
  - B8: state/action/夹爪/安全
  - B9: 质量视图聚合

Person B 独占域。
"""

from zpds_prepare.detectors.a2d.completeness import (
    A2DAssetStatus,
    A2DCompletenessReport,
    HDF5DatasetStatus,
    check_a2d_completeness,
)
from zpds_prepare.detectors.a2d.alignment import (
    A2DAlignmentReport,
    CameraRobotAlignmentRow,
    StreamAlignmentSummary,
    check_a2d_alignment,
    write_alignment_parquet,
)

from zpds_prepare.detectors.a2d.alignment import (
    A2DAlignmentReport,
    CameraRobotAlignmentRow,
    StreamAlignmentSummary,
    check_a2d_alignment,
    write_alignment_parquet,
)
from zpds_prepare.detectors.a2d.quality_views import (
    A2DQualityViewsReport,
    aggregate_a2d_quality_views,
)
from zpds_prepare.detectors.a2d.robot_quality import (
    A2DRobotQualityReport,
    GripperResponse,
    JointLimitViolation,
    StateActionLag,
    TimeSeriesQuality,
    check_a2d_robot_quality,
)

__all__ = [
    # B6
    "A2DAssetStatus",
    "A2DCompletenessReport",
    "HDF5DatasetStatus",
    "check_a2d_completeness",
    # B7
    "A2DAlignmentReport",
    "CameraRobotAlignmentRow",
    "StreamAlignmentSummary",
    "check_a2d_alignment",
    "write_alignment_parquet",
    # B8
    "A2DRobotQualityReport",
    "GripperResponse",
    "JointLimitViolation",
    "StateActionLag",
    "TimeSeriesQuality",
    "check_a2d_robot_quality",
    # B9
    "A2DQualityViewsReport",
    "aggregate_a2d_quality_views",
]
