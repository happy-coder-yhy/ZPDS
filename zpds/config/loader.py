"""严格加载 ZPDS Pipeline YAML，并生成可复现配置哈希。"""

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .registry import DEFAULT_SCHEMA_REGISTRY, SchemaRegistry


class ConfigError(ValueError):
    """配置读取或结构错误。"""


class ConfigValidationError(ConfigError):
    """配置不符合注册 Schema。"""

    def __init__(self, source: Path, errors: list[str]) -> None:
        self.source = source
        self.errors = tuple(errors)
        details = "\n".join(f"- {error}" for error in errors)
        super().__init__(f"Invalid pipeline config {source}:\n{details}")


@dataclass(frozen=True)
class LoadedConfig:
    """已校验且不可变的配置快照。"""

    source: Path
    version: str
    config_hash: str
    data: Mapping[str, Any]

    def section(self, name: str) -> Mapping[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, Mapping):
            raise KeyError(f"Config section not found: {name}")
        return value


class PipelineConfigLoader:
    """只接受已注册版本的完整 Pipeline 配置，不注入隐式默认值。"""

    def __init__(self, registry: SchemaRegistry = DEFAULT_SCHEMA_REGISTRY) -> None:
        self._registry = registry

    def load(self, path: str | Path) -> LoadedConfig:
        source = Path(path)
        if not source.is_file():
            raise ConfigError(f"Pipeline config not found: {source}")
        try:
            with source.open(encoding="utf-8") as file:
                value = yaml.safe_load(file)
        except yaml.YAMLError as error:
            raise ConfigError(f"Invalid YAML in pipeline config {source}: {error}") from error
        if not isinstance(value, dict):
            raise ConfigError(f"Pipeline config must be a YAML object: {source}")
        return self.load_mapping(value, source=source)

    def load_mapping(
        self,
        value: dict,
        *,
        source: str | Path = "<memory>",
    ) -> LoadedConfig:
        source_path = Path(source)
        data = deepcopy(value)
        version = _pipeline_version(data, source_path)
        errors = self._registry.validate(data, "pipeline_config", version)
        if errors:
            raise ConfigValidationError(source_path, errors)
        return LoadedConfig(
            source=source_path,
            version=version,
            config_hash=canonical_config_hash(data),
            data=_freeze(data),
        )


def canonical_config_hash(data: Mapping[str, Any]) -> str:
    """对配置语义而不是 YAML 排版计算确定性 SHA256。"""

    canonical = json.dumps(
        _plain(data),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _pipeline_version(data: dict, source: Path) -> str:
    pipeline = data.get("pipeline")
    if not isinstance(pipeline, dict):
        raise ConfigError(f"Missing pipeline object in config: {source}")
    version = pipeline.get("version")
    if not isinstance(version, str) or not version:
        raise ConfigError(f"Missing pipeline.version in config: {source}")
    return version


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value
