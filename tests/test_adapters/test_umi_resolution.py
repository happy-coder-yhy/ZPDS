"""UMI 视频分辨率实测与标定不一致处理测试。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from segment.image_undistorter import plan_undistortion
from zpds_prepare.readers.umi_reader import _probe_video_size


def _write_synthetic_mp4(
    path: Path,
    *,
    width: int = 1600,
    height: int = 1300,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        10,
        (width, height),
    )
    try:
        frame: np.ndarray = np.zeros((height, width, 3), dtype=np.uint8)
        writer.write(frame)
    finally:
        writer.release()


def test_probe_video_size_measures_actual_resolution(tmp_path: Path) -> None:
    video = tmp_path / "cam.mp4"
    _write_synthetic_mp4(video, width=1600, height=1300)
    assert _probe_video_size(video) == (1600, 1300)


def test_probe_video_size_returns_none_for_missing(tmp_path: Path) -> None:
    assert _probe_video_size(tmp_path / "missing.mp4") is None


def test_plan_undistortion_reports_mismatch_without_crash() -> None:
    calibration = {
        "cameras": [
            {
                "stream_id": "robot0_camera0",
                "resolution": [640, 480],
                "distortion_model": "equidistant",
                "D": [0.0, 0.0, 0.0, 0.0],
            }
        ]
    }
    plan = plan_undistortion(
        calibration,
        "robot0_camera0",
        width=1600,
        height=1300,
    )
    assert plan.status == "missing_calibration"
    assert "640x480" in plan.detail
