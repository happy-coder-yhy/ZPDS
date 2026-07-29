"""test_validator — 验证 Validator 对各种异常的检测能力。"""

import tempfile
from pathlib import Path

import pandas as pd

from zpds.hands.validator import validate_hands_parquet


def _make_parquet(path: str, rows: list[dict]) -> str:
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False)
    return path


def _valid_row(fi=0):
    return {
        "prep_revision": "r0001",
        "segment_id": "seg_000001",
        "video_stream_id": "ego_rgb",
        "output_frame_index": fi,
        "timestamp_ns": fi * 33333333,
        "source_frame_index": fi,
        "source_timestamp_ns": fi * 33333333,
        "detection_id": 0,
        "handedness": "Right",
        "handedness_score": 0.9,
        "bbox_x1": 100.0, "bbox_y1": 200.0, "bbox_x2": 300.0, "bbox_y2": 400.0,
        "keypoints_2d": [[float(100 + i * 5), float(200 + i * 5)] for i in range(21)],
        "keypoints_z_relative": [0.0] * 21,
        "model_name": "mediapipe", "model_version": "0.10",
        "checkpoint_sha256": "abc", "config_sha256": "def",
    }


class TestValidator:
    def test_valid_parquet_passes(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [_valid_row(0), _valid_row(1)])
            result = validate_hands_parquet(path)
            assert result["status"] == "pass"

    def test_missing_fields_fails(self):
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [{"handedness": "Left"}])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"
            assert "Missing required fields" in str(result["errors"])

    def test_bad_keypoints_count_fails(self):
        row = _valid_row()
        row["keypoints_2d"] = [[0.0, 0.0]] * 10  # only 10
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_kp_nan_fails(self):
        row = _valid_row()
        kp = [[float(i), float(200)] for i in range(21)]
        kp[5] = [float("nan"), float("nan")]
        row["keypoints_2d"] = kp
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_bad_handedness_fails(self):
        row = _valid_row()
        row["handedness"] = "Both"
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_score_out_of_range_fails(self):
        row = _valid_row()
        row["handedness_score"] = 1.5
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_negative_frame_index_fails(self):
        row = _valid_row()
        row["output_frame_index"] = -1
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_bbox_inverted_fails(self):
        row = _valid_row()
        row["bbox_x2"] = 50.0  # smaller than bbox_x1 (100)
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "fail"

    def test_missing_provenance_warns(self):
        row = _valid_row()
        row["model_name"] = ""
        row["model_version"] = ""
        with tempfile.TemporaryDirectory() as td:
            path = _make_parquet(Path(td) / "hands.parquet", [row])
            result = validate_hands_parquet(path)
            assert result["status"] == "warn"

    def test_nonexistent_file_fails(self):
        result = validate_hands_parquet("/nonexistent/path.parquet")
        assert result["status"] == "fail"
