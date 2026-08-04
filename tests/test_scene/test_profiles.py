from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from zpds.scene.config import SceneConfig

DEFAULT_CONFIG = Path("configs/scene/default.yaml")
PROFILE_DIR = Path("configs/qc_thresholds")
EXPECTED_HISTOGRAM_THRESHOLDS = {
    "a2d_robot": 0.40,
    "dunjia_ego": 0.50,
    "epic100": 0.38,
    "guida_ego": 0.48,
    "jianzhi_umi": 0.52,
}


@pytest.mark.parametrize(
    ("profile_name", "histogram_threshold"),
    EXPECTED_HISTOGRAM_THRESHOLDS.items(),
)
def test_scene_profile_merges_over_default_and_preserves_governance(
    profile_name: str,
    histogram_threshold: float,
) -> None:
    config = SceneConfig.load_with_profile(
        DEFAULT_CONFIG,
        PROFILE_DIR / f"{profile_name}.yaml",
    )

    assert config.profile == profile_name
    assert config.stage_a.histogram.threshold == histogram_threshold
    assert config.stage_b.model == "facebook/dinov2-small"
    assert config.stage_b.sample_fps == 1.0
    assert config.fusion.min_scene_duration_s == 3.0
    assert config.governance.calibration_status == "provisional"
    assert config.governance.soft_boundary_action == "quarantine"
    assert config.governance.auto_finalize_soft_boundaries is False


def test_each_profile_produces_distinct_traceable_config_hash() -> None:
    hashes = {
        SceneConfig.load_with_profile(DEFAULT_CONFIG, path).config_hash
        for path in PROFILE_DIR.glob("*.yaml")
    }

    assert len(hashes) == len(EXPECTED_HISTOGRAM_THRESHOLDS)


def test_provisional_profile_cannot_accept_or_finalize_soft_boundaries(
    tmp_path: Path,
) -> None:
    profile = {
        "profile": "invalid",
        "scene": {
            "governance": {
                "calibration_status": "provisional",
                "soft_boundary_action": "accept",
                "auto_finalize_soft_boundaries": True,
            }
        },
    }
    profile_path = tmp_path / "invalid.yaml"
    profile_path.write_text(yaml.safe_dump(profile), encoding="utf-8")

    with pytest.raises(ValueError, match="quarantine"):
        SceneConfig.load_with_profile(DEFAULT_CONFIG, profile_path)


def test_profile_requires_scene_section(tmp_path: Path) -> None:
    profile_path = tmp_path / "missing-scene.yaml"
    profile_path.write_text("profile: missing_scene\n", encoding="utf-8")

    with pytest.raises(TypeError, match="Profile.scene"):
        SceneConfig.load_with_profile(DEFAULT_CONFIG, profile_path)
