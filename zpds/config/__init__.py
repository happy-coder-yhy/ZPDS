"""版本化配置加载与 Schema 注册。"""

from .loader import (
    ConfigError,
    ConfigValidationError,
    LoadedConfig,
    PipelineConfigLoader,
    canonical_config_hash,
)
from .registry import (
    DEFAULT_SCHEMA_REGISTRY,
    SchemaRegistration,
    SchemaRegistry,
    UnknownSchemaError,
)

__all__ = [
    "DEFAULT_SCHEMA_REGISTRY",
    "ConfigError",
    "ConfigValidationError",
    "LoadedConfig",
    "PipelineConfigLoader",
    "SchemaRegistration",
    "SchemaRegistry",
    "UnknownSchemaError",
    "canonical_config_hash",
]
