"""JSON Schema 校验工具。"""

import json
from importlib.resources import files
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import ValidationError

SCHEMA_PACKAGE = "zpds.schemas"


def validate_json(data: dict, schema: dict) -> list[str]:
    """校验 data 是否符合 schema，返回错误列表。"""
    Draft202012Validator.check_schema(schema)
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    errors = sorted(validator.iter_errors(data), key=lambda error: list(error.absolute_path))
    return [_format_error(error) for error in errors]


def load_schema(name: str) -> dict:
    """加载内置 schema。"""
    filename = name if name.endswith(".schema.json") else f"{name}.schema.json"
    resource = files(SCHEMA_PACKAGE).joinpath(filename)
    if not resource.is_file():
        raise FileNotFoundError(f"Unknown ZPDS schema: {name}")
    return json.loads(resource.read_text(encoding="utf-8"))


def validate_with_schema(data: dict, schema_name: str) -> list[str]:
    """按内置 Schema 名称校验数据。"""
    normalized_name = schema_name.removesuffix(".schema.json")
    return validate_json(data, load_schema(normalized_name)) + _semantic_errors(
        data, normalized_name
    )


def load_json(path: str | Path) -> dict:
    """读取 UTF-8 JSON 对象。"""
    with Path(path).open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return data


def _format_error(error: ValidationError) -> str:
    path = ".".join(str(part) for part in error.absolute_path)
    return f"{path}: {error.message}" if path else error.message


def _semantic_errors(data: dict, schema_name: str) -> list[str]:
    """补充 JSON Schema 无法表达的跨字段约束。"""

    errors: list[str] = []
    if schema_name == "segment":
        _check_range(data.get("source_span"), "source_span", errors)
        assets = data.get("source_assets", [])
        asset_ids: list[str] = []
        for asset in assets:
            if not isinstance(asset, dict):
                continue
            asset_id = asset.get("source_asset_id")
            if isinstance(asset_id, str):
                asset_ids.append(asset_id)
        duplicate_asset_ids = sorted(
            asset_id for asset_id in set(asset_ids) if asset_ids.count(asset_id) > 1
        )
        if duplicate_asset_ids:
            errors.append(f"source_assets: duplicate IDs {duplicate_asset_ids}")
        known_asset_refs = {f"asset://{asset_id}" for asset_id in asset_ids}
        for index, stream in enumerate(data.get("streams", [])):
            if not isinstance(stream, dict):
                continue
            _check_range(stream.get("time"), f"streams.{index}.time", errors)
            origin = stream.get("origin", {})
            source_refs = origin.get("source_refs", []) if isinstance(origin, dict) else []
            missing_refs = sorted(
                ref
                for ref in source_refs
                if isinstance(ref, str) and ref not in known_asset_refs
            )
            if missing_refs:
                errors.append(
                    f"streams.{index}.origin.source_refs: unknown source assets "
                    f"{missing_refs}"
                )
        quality = data.get("quality", {})
        if isinstance(quality, dict):
            status = quality.get("status")
            decision = quality.get("decision")
            issues = quality.get("issues", [])
            expected_status = (
                {
                    "keep": "pass",
                    "keep_with_flag": "pass",
                    "quarantine": "quarantine",
                    "reject": "reject",
                }.get(decision)
                if isinstance(decision, str)
                else None
            )
            if expected_status is not None and status != expected_status:
                errors.append(
                    f"quality: decision {decision!r} requires status {expected_status!r}"
                )
            if decision == "keep" and issues:
                errors.append("quality: keep decision cannot contain issues")
            if decision in {"keep_with_flag", "quarantine", "reject"} and not issues:
                errors.append(f"quality: {decision} decision requires at least one issue")
    elif schema_name == "ceu":
        _check_range(data.get("time_range"), "time_range", errors)
    return errors


def _check_range(value: object, path: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        return
    start_ns = value.get("start_ns")
    end_ns = value.get("end_ns")
    if isinstance(start_ns, int) and isinstance(end_ns, int) and end_ns <= start_ns:
        errors.append(f"{path}: end_ns must be greater than start_ns")
