"""对象类型与版本到 JSON Schema 的显式注册表。"""

from dataclasses import dataclass

from jsonschema import Draft202012Validator

from zpds.utils.schema_validator import load_schema, validate_json


class UnknownSchemaError(LookupError):
    """请求的对象类型或版本没有注册 Schema。"""


@dataclass(frozen=True)
class SchemaRegistration:
    object_type: str
    version: str
    schema_name: str

    def __post_init__(self) -> None:
        if not self.object_type:
            raise ValueError("object_type must not be empty")
        if not self.version:
            raise ValueError("version must not be empty")
        if not self.schema_name:
            raise ValueError("schema_name must not be empty")


class SchemaRegistry:
    """严格、显式的版本化 Schema 注册表。"""

    def __init__(self, registrations: tuple[SchemaRegistration, ...] = ()) -> None:
        self._registrations: dict[tuple[str, str], SchemaRegistration] = {}
        for registration in registrations:
            self.register(registration)

    def register(self, registration: SchemaRegistration) -> None:
        key = (registration.object_type, registration.version)
        if key in self._registrations:
            raise ValueError(
                f"Schema already registered for {registration.object_type!r} "
                f"version {registration.version!r}"
            )
        Draft202012Validator.check_schema(load_schema(registration.schema_name))
        self._registrations[key] = registration

    def resolve(self, object_type: str, version: str) -> SchemaRegistration:
        try:
            return self._registrations[(object_type, version)]
        except KeyError as error:
            known_versions = sorted(
                registered_version
                for registered_type, registered_version in self._registrations
                if registered_type == object_type
            )
            versions = ", ".join(known_versions) if known_versions else "none"
            raise UnknownSchemaError(
                f"No Schema registered for {object_type!r} version {version!r}; "
                f"known versions: {versions}"
            ) from error

    def validate(self, data: dict, object_type: str, version: str) -> list[str]:
        registration = self.resolve(object_type, version)
        return validate_json(data, load_schema(registration.schema_name))

    def registrations(self) -> tuple[SchemaRegistration, ...]:
        return tuple(
            self._registrations[key] for key in sorted(self._registrations)
        )


def _default_registrations() -> tuple[SchemaRegistration, ...]:
    names = (
        "dataset",
        "revision",
        "segment",
        "experience_manifest",
        "ceu",
        "release",
        "reason_code_registry",
        "quality_view_registry",
        "governance_config",
        "gold_manifest",
        "pipeline_config",
        "run_ledger",
        "run_metrics",
        "sample_map",
        "source_inventory",
        "validation_report",
    )
    return tuple(
        SchemaRegistration(object_type=name, version="0.1.0", schema_name=name)
        for name in names
    )


DEFAULT_SCHEMA_REGISTRY = SchemaRegistry(_default_registrations())
