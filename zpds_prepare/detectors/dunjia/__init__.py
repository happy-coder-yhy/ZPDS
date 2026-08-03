"""遁甲 (Dunjia) 末端观测源专用检测器。

覆盖 B1–B5：
  - B1: 来源完整性清单与相机角色
  - B2: RGB-D 质量
  - B3: IMU 质量
  - B4: 多相机覆盖与末端可见性
  - B5: 质量视图聚合

Person B 独占域；不修改统一 schema，不虚构 robot_bc_ready。
"""

from zpds_prepare.detectors.dunjia.completeness import (
    CameraRole,
    DunjiaCompletenessReport,
    StreamCompleteness,
    check_dunjia_completeness,
)
from zpds_prepare.detectors.dunjia.rgbd_quality import (
    DepthFrameSample,
    DunjiaRGBDReport,
    RGBDepthAlignment,
    check_dunjia_rgbd,
)
from zpds_prepare.detectors.dunjia.imu_quality import (
    DunjiaIMUReport,
    IMUGapSpan,
    IMUSpikeEvent,
    IMUStaticWindow,
    check_dunjia_imu,
)
from zpds_prepare.detectors.dunjia.coverage import (
    CameraCoverageRecord,
    DunjiaCoverageReport,
    EndEffectorVisibility,
    check_dunjia_coverage,
)
from zpds_prepare.detectors.dunjia.quality_views import (
    DunjiaQualityViewsReport,
    QualityView,
    aggregate_dunjia_quality_views,
)

__all__ = [
    # B1
    "CameraRole",
    "DunjiaCompletenessReport",
    "StreamCompleteness",
    "check_dunjia_completeness",
    # B2
    "DepthFrameSample",
    "DunjiaRGBDReport",
    "RGBDepthAlignment",
    "check_dunjia_rgbd",
    # B3
    "DunjiaIMUReport",
    "IMUGapSpan",
    "IMUSpikeEvent",
    "IMUStaticWindow",
    "check_dunjia_imu",
    # B4
    "CameraCoverageRecord",
    "DunjiaCoverageReport",
    "EndEffectorVisibility",
    "check_dunjia_coverage",
    # B5
    "DunjiaQualityViewsReport",
    "QualityView",
    "aggregate_dunjia_quality_views",
]
