from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

from scripts.run_scene_regression import run
from zpds.scene.config import SceneConfig
from zpds.scene.regression import run_stage_a_regression
from zpds.scene.testing import hard_cut_fixture


def test_regression_report_is_provisional_and_preserves_thresholds() -> None:
    fixture = hard_cut_fixture()
    config = SceneConfig.load("configs/scene/default.yaml")

    report = run_stage_a_regression(
        fixture.frames,
        fps=fixture.fps,
        config=config,
    )

    assert report["calibration_status"] == "provisional"
    assert report["requires_adjudicated_gold"] is True
    assert report["thresholds_changed"] is False
    assert report["config_hash"] == config.config_hash
    assert report["fused_transition_count"] >= 1
    assert set(report["detectors"]) == {
        "histogram",
        "ssim",
        "optical_flow",
        "brightness",
    }
    assert report["detectors"]["histogram"]["scores"]["p99"] is not None


def test_regression_cli_writes_multi_case_report_without_changing_raw(
    tmp_path: Path,
) -> None:
    fixture = hard_cut_fixture()
    video_paths = [tmp_path / "first.mp4", tmp_path / "second.mp4"]
    height, width = fixture.frames[0].shape[:2]
    for path in video_paths:
        writer = cv2.VideoWriter(
            str(path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fixture.fps,
            (width, height),
        )
        assert writer.isOpened()
        for frame in fixture.frames:
            writer.write(frame)
        writer.release()

    output = tmp_path / "regression.json"
    args = argparse.Namespace(
        input=[str(path) for path in video_paths],
        profile=["configs/qc_thresholds/guida_ego.yaml"],
        name=["first", "second"],
        config="configs/scene/default.yaml",
        max_frames=None,
        output_json=str(output),
        quiet=True,
    )

    assert run(args) == 0
    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["calibration_status"] == "provisional"
    assert document["thresholds_changed"] is False
    assert [case["name"] for case in document["cases"]] == ["first", "second"]
    assert all(case["profile"] == "guida_ego" for case in document["cases"])
    assert all(case["raw_unchanged"] is True for case in document["cases"])
