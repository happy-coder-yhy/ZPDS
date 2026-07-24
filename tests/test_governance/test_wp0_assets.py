from copy import deepcopy
from pathlib import Path

import yaml

from scripts.validate_wp0 import validate_wp0
from zpds.utils.schema_validator import validate_with_schema

ROOT = Path(__file__).resolve().parents[2]


def _gold_manifest() -> dict:
    with (ROOT / "configs" / "gold" / "five_source_manifest.yaml").open(
        encoding="utf-8"
    ) as file:
        return yaml.safe_load(file)


def test_wp0_machine_contract() -> None:
    assert validate_wp0() == []


def test_frozen_gold_manifest_accepts_one_designated_reviewer() -> None:
    manifest = deepcopy(_gold_manifest())
    manifest["status"] = "frozen"
    for sample in manifest["samples"]:
        sample["review"]["status"] = "approved"
        sample["review"]["reviewers"] = [manifest["designated_reviewer"]]
        sample["review"]["reviewed_at"] = "2026-07-23T16:00:00+08:00"

    assert validate_with_schema(manifest, "gold_manifest") == []


def test_frozen_gold_manifest_rejects_missing_reviewer() -> None:
    manifest = deepcopy(_gold_manifest())
    manifest["status"] = "frozen"
    for sample in manifest["samples"]:
        sample["review"]["status"] = "approved"
        sample["review"]["reviewers"] = [manifest["designated_reviewer"]]
        sample["review"]["reviewed_at"] = "2026-07-23T16:00:00+08:00"
    manifest["samples"][0]["review"]["reviewers"] = []

    errors = validate_with_schema(manifest, "gold_manifest")

    assert any("reviewers" in error for error in errors)
