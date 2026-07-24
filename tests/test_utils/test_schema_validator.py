import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from zpds.utils.schema_validator import load_schema, validate_with_schema

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_SCHEMAS = {
    "dataset.json": "dataset",
    "revision.json": "revision",
    "segment.json": "segment",
    "segment_quarantine.json": "segment",
    "experience_manifest.json": "experience_manifest",
    "ceu.json": "ceu",
    "release.json": "release",
}


@pytest.mark.parametrize("schema_name", EXAMPLE_SCHEMAS.values())
def test_builtin_schemas_are_valid_draft_2020_12(schema_name: str) -> None:
    Draft202012Validator.check_schema(load_schema(schema_name))


@pytest.mark.parametrize(("filename", "schema_name"), EXAMPLE_SCHEMAS.items())
def test_minimal_examples_pass_schema(filename: str, schema_name: str) -> None:
    data = json.loads((ROOT / "examples" / "schemas" / filename).read_text("utf-8"))

    assert validate_with_schema(data, schema_name) == []


def test_strict_schema_rejects_legacy_version_name() -> None:
    data = json.loads(
        (ROOT / "examples" / "schemas" / "dataset.json").read_text("utf-8")
    )
    data["zrds_version"] = data.pop("zpds_version")

    errors = validate_with_schema(data, "dataset")

    assert any("zpds_version" in error for error in errors)
    assert any("zrds_version" in error for error in errors)


def test_semantic_validator_rejects_reversed_ceu_range() -> None:
    data = json.loads((ROOT / "examples" / "schemas" / "ceu.json").read_text("utf-8"))
    data["time_range"] = {"start_ns": 100, "end_ns": 10}

    assert "time_range: end_ns must be greater than start_ns" in validate_with_schema(
        data, "ceu"
    )


def test_segment_rejects_origin_reference_missing_from_source_assets() -> None:
    data = json.loads(
        (ROOT / "examples" / "schemas" / "segment.json").read_text("utf-8")
    )
    data["streams"][0]["origin"]["source_refs"].append("asset://not_registered")

    errors = validate_with_schema(data, "segment")

    assert any("asset://not_registered" in error for error in errors)


def test_segment_rejects_inconsistent_status_and_decision() -> None:
    data = json.loads(
        (ROOT / "examples" / "schemas" / "segment_quarantine.json").read_text("utf-8")
    )
    data["quality"]["status"] = "pass"

    errors = validate_with_schema(data, "segment")

    assert any("requires status 'quarantine'" in error for error in errors)
