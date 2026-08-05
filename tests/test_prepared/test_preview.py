"""前端预览压缩与 segment.json preview_uri 测试。"""

from __future__ import annotations

import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest

from segment.preview import create_preview
from segment.segment_writer import build_segment_json


def _write_synthetic_mp4(
    path: Path,
    *,
    width: int = 640,
    height: int = 360,
    frames: int = 10,
) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # type: ignore[attr-defined]
        10,
        (width, height),
    )
    try:
        for index in range(frames):
            frame: np.ndarray = np.full(
                (height, width, 3), index * 20, dtype=np.uint8
            )
            writer.write(frame)
    finally:
        writer.release()


@pytest.mark.skipif(
    shutil.which("ffmpeg") is None,
    reason="ffmpeg 未安装",
)
class TestCreatePreview:
    def test_preview_created_and_downscaled(self, tmp_path: Path) -> None:
        source = tmp_path / "src.mp4"
        _write_synthetic_mp4(source, width=640, height=360)
        preview = tmp_path / "preview.mp4"

        stats = create_preview(source, preview, max_width=320, crf=30)

        assert preview.is_file()
        assert preview.stat().st_size > 0
        assert stats["width"] <= 320
        assert stats["preview_size_bytes"] > 0
        assert stats["source_size_bytes"] > 0

    def test_no_upscale_when_smaller(self, tmp_path: Path) -> None:
        source = tmp_path / "src.mp4"
        _write_synthetic_mp4(source, width=320, height=180)
        preview = tmp_path / "preview.mp4"

        stats = create_preview(source, preview, max_width=1280, crf=28)

        assert stats["width"] == 320
        assert stats["height"] == 180


class TestCreatePreviewErrors:
    def test_missing_source_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="源视频不存在"):
            create_preview(tmp_path / "missing.mp4", tmp_path / "p.mp4")

    def test_invalid_crf_raises(self, tmp_path: Path) -> None:
        source = tmp_path / "src.mp4"
        _write_synthetic_mp4(source)
        with pytest.raises(ValueError, match="crf"):
            create_preview(source, tmp_path / "p.mp4", crf=99)


class TestSegmentJsonPreviewUri:
    def test_stream_includes_preview_uri(self, tmp_path: Path) -> None:
        span = {
            "source_start_ns": 0,
            "source_end_ns": 1_000_000_000,
            "duration_s": 1.0,
            "total_frames_in_span": 30,
            "reason": {"start": "test", "end": "test"},
            "trimmed_head_frames": 0,
            "trimmed_tail_frames": 0,
        }
        video_results = [
            {
                "stream_id": "camera_ego",
                "width": 1280,
                "height": 720,
                "output_fps": 30.0,
                "sample_map_uri": "maps/camera_ego_sample_map.parquet",
                "preview_uri": "data/camera_ego_preview.mp4",
            }
        ]
        segment = build_segment_json(
            dataset_path=str(tmp_path),
            span=span,
            video_results=video_results,
            source_assets=[],
        )
        stream = segment["streams"][0]
        assert stream["stream_id"] == "camera_ego"
        assert stream["uri"] == "data/camera_ego.mp4"
        assert stream["preview_uri"] == "data/camera_ego_preview.mp4"

    def test_stream_omits_preview_uri_when_absent(
        self, tmp_path: Path
    ) -> None:
        span = {
            "source_start_ns": 0,
            "source_end_ns": 1_000_000_000,
            "duration_s": 1.0,
            "total_frames_in_span": 30,
            "reason": {"start": "test", "end": "test"},
            "trimmed_head_frames": 0,
            "trimmed_tail_frames": 0,
        }
        segment = build_segment_json(
            dataset_path=str(tmp_path),
            span=span,
            video_results=[
                {
                    "stream_id": "camera_ego",
                    "width": 1280,
                    "height": 720,
                    "output_fps": 30.0,
                }
            ],
            source_assets=[],
        )
        assert "preview_uri" not in segment["streams"][0]
