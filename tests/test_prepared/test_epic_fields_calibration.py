"""EPIC-Fields 标定解析测试。"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from segment.epic_fields_calibration import (
    EPIC_FIELDS_CODE_COMMIT,
    EPIC_FIELDS_SAMPLE_URL,
    find_epic_fields_json,
    load_epic_fields_calibration,
    missing_epic_fields_calibration,
)


FIXTURE = Path(__file__).parent.parent / "fixtures" / "epic_fields" / "P28_101.json"


def test_load_epic_fields_opencv_calibration(tmp_path: Path) -> None:
    calibration_path = tmp_path / "P28_101.json"
    calibration_path.write_bytes(FIXTURE.read_bytes())

    calibration = load_epic_fields_calibration(tmp_path, "P28_101")

    camera = calibration["cameras"][0]
    assert calibration["coverage"] == {"status": "covered", "video_id": "P28_101"}
    assert camera["stream_id"] == "ego_rgb"
    assert camera["distortion_model"] == "plumb_bob"
    assert camera["D"] == pytest.approx(
        [-0.0013492140520415712, 0.00018135429459831083, 0.0008523602906170594, -0.0006090893593678664]
    )
    assert calibration["source"]["reference_url"] == EPIC_FIELDS_SAMPLE_URL
    assert calibration["source"]["git_commit"] == EPIC_FIELDS_CODE_COMMIT
    assert len(calibration["source"]["sha256"]) == 64


def test_epic_fields_rejects_non_opencv_model(tmp_path: Path) -> None:
    document = json.loads(FIXTURE.read_text(encoding="utf-8"))
    document["camera"]["model"] = "SIMPLE_PINHOLE"
    (tmp_path / "P28_101.json").write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="必须为 OPENCV"):
        load_epic_fields_calibration(tmp_path, "P28_101")


def test_epic_fields_missing_video_is_explicit(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="未覆盖视频"):
        find_epic_fields_json(tmp_path, "P99_999")

    calibration = missing_epic_fields_calibration("P99_999", tmp_path)
    assert calibration["coverage"]["status"] == "missing_calibration"
    assert calibration["cameras"] == []
