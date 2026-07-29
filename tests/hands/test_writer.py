"""test_writer — 验证 Writer 输出 Parquet 结构和数据往返。"""

import tempfile
from pathlib import Path

import pandas as pd

from zpds.hands.base import HandBBox, HandKeypoints, RawHandResult
from zpds.hands.writer import compute_config_sha256, write_hands_parquet


def _make_hand(handedness="Right", score=0.95):
    kp = HandKeypoints(
        normalized=[(0.5 + i * 0.01, 0.5 + i * 0.01, 0.0) for i in range(21)],
        pixel=[(100.0 + i * 10, 200.0 + i * 10) for i in range(21)],
    )
    bbox = HandBBox(x1=80, y1=180, x2=320, y2=380, confidence=score)
    return RawHandResult(
        handedness=handedness, handedness_score=score,
        keypoints=kp, bbox=bbox, detection_score=score,
        label="hand_0",
    )


class TestWriter:
    def test_write_empty(self):
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hands.parquet"
            path = write_hands_parquet([], str(out))
            df = pd.read_parquet(path)
            assert len(df) == 0

    def test_write_single_hand(self):
        hand = _make_hand("Left", 0.88)
        obs = [{
            "frame_meta": {
                "segment_id": "seg_000001", "video_stream_id": "ego_rgb_center",
                "output_frame_index": 0, "timestamp_ns": 0,
                "source_frame_index": 0, "source_timestamp_ns": 1234567890,
            },
            "hands": [hand],
        }]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hands.parquet"
            path = write_hands_parquet(obs, str(out), model_meta={
                "model_name": "mediapipe", "model_version": "0.10.14",
                "checkpoint_sha256": "abc123", "config_sha256": "def456",
            })
            df = pd.read_parquet(path)
            assert len(df) == 1
            row = df.iloc[0]
            assert row["handedness"] == "Left"
            assert row["handedness_score"] == 0.88
            assert row["bbox_x1"] == 80
            assert row["detection_id"] == 0
            assert len(row["keypoints_2d"]) == 21
            assert len(row["keypoints_2d"][0]) == 2
            assert len(row["keypoints_z_relative"]) == 21
            assert row["model_name"] == "mediapipe"
            assert row["checkpoint_sha256"] == "abc123"

    def test_write_two_hands_one_frame(self):
        left = _make_hand("Left", 0.9)
        right = _make_hand("Right", 0.85)
        obs = [{
            "frame_meta": {"segment_id": "seg_000001", "video_stream_id": "ego_rgb_center",
                           "output_frame_index": 5, "timestamp_ns": 166666666,
                           "source_frame_index": None, "source_timestamp_ns": None},
            "hands": [left, right],
        }]

        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hands.parquet"
            path = write_hands_parquet(obs, str(out))
            df = pd.read_parquet(path)
            assert len(df) == 2
            assert list(df["handedness"]) == ["Left", "Right"]
            assert list(df["detection_id"]) == [0, 1]
            assert all(df["output_frame_index"] == 5)

    def test_write_multi_frame(self):
        obs = []
        for fi in range(10):
            hands = [_make_hand("Right", 0.9)] if fi % 3 == 0 else []
            obs.append({
                "frame_meta": {"segment_id": "s1", "video_stream_id": "v1",
                               "output_frame_index": fi, "timestamp_ns": fi * 33333333,
                               "source_frame_index": fi, "source_timestamp_ns": fi * 33333333},
                "hands": hands,
            })
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "hands.parquet"
            path = write_hands_parquet(obs, str(out))
            df = pd.read_parquet(path)
            assert len(df) == 4  # frames 0,3,6,9
            assert df["output_frame_index"].tolist() == [0, 3, 6, 9]

    def test_config_sha256_deterministic(self):
        cfg = {"hands": {"model": "mediapipe", "num_hands": 2}}
        h1 = compute_config_sha256(cfg)
        h2 = compute_config_sha256(cfg)
        h3 = compute_config_sha256({"hands": {"num_hands": 2, "model": "mediapipe"}})
        assert h1 == h2
        assert h1 == h3  # sort_keys 保证顺序无关
