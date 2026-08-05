from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
import yaml

from zpds.scene.backend_router import SceneBackendRouter
from zpds.scene.config import SceneConfig

DEFAULT_CONFIG = Path("configs/scene/default.yaml")


def _write_config(tmp_path: Path, document: dict) -> Path:
    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(document), encoding="utf-8")
    return path


def _default_document() -> dict:
    return yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))


def test_default_config_loads_all_required_sections() -> None:
    config = SceneConfig.load(DEFAULT_CONFIG)

    assert config.enabled is True
    assert config.stage_a.smoothing_window_frames == 5
    assert config.stage_a.merge_window_s == 0.5
    assert config.stage_b.model == "facebook/dinov2-small"
    assert config.stage_b.z_score_threshold == 2.0
    assert config.fusion.min_scene_duration_s == 3.0
    assert config.fusion.same_scene_similarity == 0.95
    assert config.vlm.api_key_env == "DASHSCOPE_API_KEY"
    assert config.vlm.base_url == (
        "https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    assert config.output_dir == (DEFAULT_CONFIG.resolve().parent / "../../output/scene").resolve()


def test_config_hash_uses_stable_full_document() -> None:
    document = _default_document()
    expected = hashlib.sha256(
        json.dumps(
            document,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert SceneConfig.load(DEFAULT_CONFIG).config_hash == expected


def test_config_resolves_relative_output_path(tmp_path: Path) -> None:
    document = _default_document()
    document["scene"]["output_dir"] = "artifacts/scene"
    config = SceneConfig.load(_write_config(tmp_path, document))

    assert config.output_dir == (tmp_path / "artifacts/scene").resolve()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda document: document["scene"]["stage_a"]["histogram"].update(
                {"threshold": 1.5}
            ),
            "histogram.threshold",
        ),
        (
            lambda document: document["scene"]["stage_a"]["ssim"].update(
                {"gaussian_window_size": 10}
            ),
            "gaussian_window_size",
        ),
        (
            lambda document: document["scene"]["stage_a"]["optical_flow"].update(
                {"pyr_scale": 1.0}
            ),
            "pyr_scale",
        ),
        (
            lambda document: document["scene"]["stage_a"].update(
                {"weights": {"unknown": 1.0}}
            ),
            "未知检测器",
        ),
        (
            lambda document: document["scene"]["stage_a"].update(
                {"smoothing_window_frames": 4}
            ),
            "smoothing_window_frames",
        ),
        (
            lambda document: document["scene"]["stage_b"].update(
                {"model": "openai/clip-vit-base-patch32"}
            ),
            "仅支持",
        ),
    ],
)
def test_config_rejects_invalid_values(tmp_path: Path, mutate, message: str) -> None:
    document = copy.deepcopy(_default_document())
    mutate(document)
    with pytest.raises(ValueError, match=message):
        SceneConfig.load(_write_config(tmp_path, document))


def test_router_does_not_load_model_runtime() -> None:
    router = SceneBackendRouter.from_config(SceneConfig.load(DEFAULT_CONFIG))

    assert router.policy.enabled is True
    assert router.policy.stage_a_backends == (
        "histogram",
        "ssim",
        "optical_flow",
        "brightness",
    )
    assert router.policy.semantic_backend == "dino"
