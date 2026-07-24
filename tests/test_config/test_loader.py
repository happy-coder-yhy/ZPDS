from copy import deepcopy
from pathlib import Path

import pytest
import yaml

from zpds.config import (
    ConfigError,
    ConfigValidationError,
    PipelineConfigLoader,
    SchemaRegistration,
    SchemaRegistry,
    UnknownSchemaError,
    canonical_config_hash,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "pipeline" / "default.yaml"
PILOT_CONFIG = ROOT / "configs" / "pipeline" / "pilot.yaml"


@pytest.mark.parametrize("path", [DEFAULT_CONFIG, PILOT_CONFIG])
def test_pipeline_configs_load_without_prepared_resampling(path: Path) -> None:
    loaded = PipelineConfigLoader().load(path)

    assert loaded.version == "0.1.0"
    assert loaded.config_hash.startswith("sha256:")
    assert len(loaded.config_hash) == 71
    assert loaded.section("prepared")["preserve_source_rate"] is True
    assert "target_fps" not in loaded.section("prepared")
    assert loaded.section("alignment")["target_fps"] == 30


def test_loaded_config_is_immutable() -> None:
    loaded = PipelineConfigLoader().load(DEFAULT_CONFIG)

    with pytest.raises(TypeError):
        loaded.data["pipeline"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        loaded.section("prepared")["depth_format"] = "png"  # type: ignore[index]


def test_config_hash_ignores_mapping_order() -> None:
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    reversed_data = dict(reversed(list(data.items())))

    assert canonical_config_hash(data) == canonical_config_hash(reversed_data)


def test_loaded_snapshot_can_be_hashed_again() -> None:
    loaded = PipelineConfigLoader().load(DEFAULT_CONFIG)

    assert canonical_config_hash(loaded.data) == loaded.config_hash


def test_loader_rejects_target_fps_in_prepared() -> None:
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data["prepared"]["target_fps"] = 30

    with pytest.raises(ConfigValidationError, match="target_fps"):
        PipelineConfigLoader().load_mapping(data)


def test_loader_rejects_missing_required_section() -> None:
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    del data["alignment"]

    with pytest.raises(ConfigValidationError, match="alignment"):
        PipelineConfigLoader().load_mapping(data)


def test_loader_rejects_unknown_version() -> None:
    with DEFAULT_CONFIG.open(encoding="utf-8") as file:
        data = yaml.safe_load(file)
    data["pipeline"]["version"] = "9.9.9"

    with pytest.raises(UnknownSchemaError, match="9.9.9"):
        PipelineConfigLoader().load_mapping(data)


def test_loader_rejects_non_object_yaml(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("- item\n", encoding="utf-8")

    with pytest.raises(ConfigError, match="YAML object"):
        PipelineConfigLoader().load(path)


def test_schema_registry_resolves_and_rejects_duplicates() -> None:
    registration = SchemaRegistration("pipeline_config", "0.1.0", "pipeline_config")
    registry = SchemaRegistry((registration,))

    assert registry.resolve("pipeline_config", "0.1.0") == registration
    with pytest.raises(ValueError, match="already registered"):
        registry.register(deepcopy(registration))
    with pytest.raises(UnknownSchemaError, match="known versions: 0.1.0"):
        registry.resolve("pipeline_config", "0.2.0")
