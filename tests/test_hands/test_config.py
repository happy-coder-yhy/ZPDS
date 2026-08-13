import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from scripts.run_hands import _image_dimensions
from zpds.hands.config import HandsOutputPaths, HandsPipelineConfig
from zpds.hands.experience import write_hands_experience_manifest
from zpds.hands.schemas import HandObservation
from zpds.hands.writer import compute_config_sha256, write_hand_observations


def _config_document() -> dict:
    """单后端 WiLoR 配置（checkpoint 路径相对配置目录解析）。"""
    return {
        "hands": {
            "wilor": {
                "enabled": True,
                "ego_bbox_every_frame": True,
                "bbox_fps": 30.0,
                "write_frame_status": True,
                "checkpoint_path": "models/wilor.ckpt",
                "device": "cpu",
                "precision": "fp32",
                "model_version": "wilor_cvpr2025",
            }
        }
    }


def _write_config(root: Path, document: dict | None = None) -> Path:
    config_path = root / "config.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(document or _config_document()),
        encoding="utf-8",
    )
    return config_path


def _write_checkpoint(root: Path) -> Path:
    checkpoint = root / "models" / "wilor.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"wilor-checkpoint")
    return checkpoint


def _observation() -> HandObservation:
    return HandObservation(
        segment_id="seg_000001",
        video_stream_id="ego_rgb",
        output_frame_index=0,
        timestamp_ns=0,
        source_frame_index=0,
        source_timestamp_ns=0,
        detection_id=0,
        handedness="left",
        handedness_score=0.9,
        bbox_xyxy=(1.0, 1.0, 20.0, 20.0),
        keypoints_2d=[(2.0 + index / 2, 2.0 + index / 2) for index in range(21)],
        keypoints_z_relative=[0.0] * 21,
        model_name="wilor",
        model_version="test",
    )


def test_config_resolves_checkpoint_relative_to_yaml(tmp_path: Path) -> None:
    checkpoint = _write_checkpoint(tmp_path)
    config_path = _write_config(tmp_path)

    config = HandsPipelineConfig.load(config_path)

    assert config.wilor.checkpoint_path == str(checkpoint.resolve())
    assert config.checkpoint_sha256 == hashlib.sha256(b"wilor-checkpoint").hexdigest()
    assert config.config_sha256 == compute_config_sha256(_config_document())


def test_config_requires_wilor_enabled(tmp_path: Path) -> None:
    document = _config_document()
    document["hands"]["wilor"]["enabled"] = False
    config_path = _write_config(tmp_path, document)

    try:
        HandsPipelineConfig.load(config_path)
    except ValueError as error:
        assert "enabled" in str(error)
    else:
        raise AssertionError("单后端模式必须要求 wilor.enabled=true")


def test_config_rejects_missing_checkpoint(tmp_path: Path) -> None:
    config_path = _write_config(tmp_path)

    try:
        HandsPipelineConfig.load(config_path)
    except FileNotFoundError as error:
        assert "checkpoint" in str(error).lower() or "不存在" in str(error)
    else:
        raise AssertionError("缺失 checkpoint 必须报错")


def test_wilor_sampling_config_accepted(tmp_path: Path) -> None:
    _write_checkpoint(tmp_path)
    document = _config_document()
    document["hands"]["wilor"]["ego_bbox_every_frame"] = False
    document["hands"]["wilor"]["bbox_fps"] = 10.0
    config_path = _write_config(tmp_path, document)

    config = HandsPipelineConfig.load(config_path)

    assert config.wilor.ego_bbox_every_frame is False
    assert config.wilor.bbox_fps == 10.0


def test_output_paths_follow_experience_layout(tmp_path: Path) -> None:
    paths = HandsOutputPaths.experience(tmp_path / "exp_v1")

    assert paths.parquet == (
        tmp_path / "exp_v1" / "assets" / "poses" / "hands_2d.parquet"
    ).resolve()
    assert paths.validation_report.parent.name == "reports"
    assert paths.preview.parent.name == "previews"
    assert paths.experience_manifest.name == "experience_manifest.json"


def test_image_dimensions_use_actual_video_for_validation(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "rgb.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (20, 12),
    )
    assert writer.isOpened()
    writer.write(np.zeros((12, 20, 3), dtype=np.uint8))
    writer.release()
    segment = {
        "streams": [
            {
                "stream_id": "ego_rgb",
                "shape": [6, 8, 3],
                "uri": "rgb.mp4",
            }
        ]
    }

    assert _image_dimensions(segment, "ego_rgb") == (8, 6)
    assert _image_dimensions(segment, "ego_rgb", tmp_path) == (20, 12)


def test_experience_manifest_registers_hands_assets(tmp_path: Path) -> None:
    experience_dir = tmp_path / "exp_v1"
    paths = HandsOutputPaths.experience(experience_dir)
    write_hand_observations(
        [_observation()],
        paths.parquet,
        checkpoint_sha256="model-hash",
        config_sha256="config-hash",
    )
    paths.validation_report.parent.mkdir(parents=True, exist_ok=True)
    paths.validation_report.write_text('{"status":"pass"}', encoding="utf-8")
    paths.run_manifest.write_text('{"completed":true}', encoding="utf-8")
    paths.preview.parent.mkdir(parents=True, exist_ok=True)
    paths.preview.write_bytes(b"preview")
    assert paths.experience_manifest is not None
    paths.experience_manifest.write_text(
        '{"experience_version":"exp_v1",'
        '"annotations":{"objects_v1":{"rows":3}}}',
        encoding="utf-8",
    )

    manifest_path = write_hands_experience_manifest(
        experience_dir=experience_dir,
        experience_version="exp_v1",
        segment_id="seg_000001",
        video_stream_id="ego_rgb",
        outputs=paths,
        prep_revision="r0001",
        config_sha256="config-hash",
        checkpoint_sha256="model-hash",
        validation_status="pass",
    )
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    hands = manifest["annotations"]["hands_v1"]

    assert hands["rows"] == 1
    assert hands["model_name"] == "wilor"
    assert hands["annotated_frames"] == 1
    assert hands["validation_status"] == "pass"
    assert hands["files"]["hands_2d"]["uri"] == "assets/poses/hands_2d.parquet"
    assert hands["files"]["hands_preview"]["uri"] == "previews/hands_preview.mp4"
    assert manifest["annotations"]["objects_v1"] == {"rows": 3}
