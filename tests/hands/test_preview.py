"""test_preview — 验证 Hands 预览坐标转换和视频流过滤。"""

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from zpds.hands.preview import generate_hands_preview


def _write_video(path: Path, color: tuple[int, int, int], frames: int = 3) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (100, 120),
    )
    assert writer.isOpened()
    for _ in range(frames):
        frame = np.full((120, 100, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _write_segment(segment_dir: Path) -> None:
    data_dir = segment_dir / "data"
    data_dir.mkdir(parents=True)
    _write_video(data_dir / "front.mp4", (0, 0, 0))
    _write_video(data_dir / "wrist.mp4", (32, 32, 32))
    segment = {
        "streams": [
            {"stream_id": "front", "format": "mp4", "uri": "data/front.mp4"},
            {"stream_id": "wrist", "format": "mp4", "uri": "data/wrist.mp4"},
        ],
        "timeline": {"end_ns": 100_000_000},
    }
    (segment_dir / "segment.json").write_text(
        json.dumps(segment),
        encoding="utf-8",
    )


def _row(video_stream_id: str, bbox: tuple[float, float, float, float]) -> dict:
    keypoints = [[0.2 + i * 0.01, 0.25 + i * 0.01] for i in range(21)]
    return {
        "prep_revision": "r0001",
        "segment_id": "seg",
        "video_stream_id": video_stream_id,
        "output_frame_index": 0,
        "timestamp_ns": 0,
        "source_frame_index": 0,
        "source_timestamp_ns": 0,
        "detection_id": 0,
        "handedness": "Right",
        "handedness_score": 0.9,
        "bbox_x1": bbox[0],
        "bbox_y1": bbox[1],
        "bbox_x2": bbox[2],
        "bbox_y2": bbox[3],
        "keypoints_2d": keypoints,
        "keypoints_z_relative": [0.0] * 21,
        "model_name": "mediapipe",
        "model_version": "0.10",
        "checkpoint_sha256": "abc",
        "config_sha256": "def",
    }


def test_preview_filters_selected_stream_and_scales_normalized_coordinates():
    with tempfile.TemporaryDirectory() as td:
        segment_dir = Path(td) / "segment"
        segment_dir.mkdir()
        _write_segment(segment_dir)

        parquet_path = Path(td) / "hands.parquet"
        pd.DataFrame([
            _row("front", (0.18, 0.20, 0.45, 0.55)),
            _row("wrist", (0.70, 0.45, 0.95, 0.62)),
        ]).to_parquet(parquet_path, index=False)

        output = Path(td) / "preview.mp4"
        result = generate_hands_preview(
            str(segment_dir),
            str(parquet_path),
            output_path=str(output),
            video_stream_id="front",
        )

        assert result == str(output.resolve())
        cap = cv2.VideoCapture(str(output))
        ok, frame = cap.read()
        cap.release()
        assert ok

        # front 的归一化 bbox 应被缩放到大约 x=18..45, y=24..66；
        # wrist 的 x=70..95, y=54..74 框若错误叠加，会污染右侧中部区域。
        assert frame[24:67, 18:46].max() > 80
        assert frame[54:75, 70:96].max() < 80
