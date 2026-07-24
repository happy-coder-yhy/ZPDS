"""一键校验 WP0 Schema、配置注册表、五源 Gold Manifest 和最小样例。"""

from collections import Counter
from pathlib import Path

import yaml

from zpds.core.decisions import ReasonCode
from zpds.utils.schema_validator import load_json, validate_with_schema

ROOT = Path(__file__).resolve().parents[1]


def _load_yaml(relative_path: str) -> dict:
    with (ROOT / relative_path).open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise TypeError(f"Expected YAML object: {relative_path}")
    return value


def _check(errors: list[str], label: str, data: dict, schema: str) -> None:
    for error in validate_with_schema(data, schema):
        errors.append(f"{label}: {error}")


def validate_wp0() -> list[str]:
    errors: list[str] = []
    examples = {
        "dataset": "dataset",
        "revision": "revision",
        "segment": "segment",
        "segment_quarantine": "segment",
        "experience_manifest": "experience_manifest",
        "ceu": "ceu",
        "release": "release",
    }
    for filename, schema in examples.items():
        path = ROOT / "examples" / "schemas" / f"{filename}.json"
        _check(errors, str(path.relative_to(ROOT)), load_json(path), schema)

    governance = _load_yaml("configs/governance/versioning.yaml")
    reasons = _load_yaml("configs/reason_codes/v0.1.0.yaml")
    quality = _load_yaml("configs/quality_views/v0.1.0.yaml")
    gold = _load_yaml("configs/gold/five_source_manifest.yaml")
    _check(errors, "versioning", governance, "governance_config")
    _check(errors, "reason_codes", reasons, "reason_code_registry")
    _check(errors, "quality_views", quality, "quality_view_registry")
    _check(errors, "gold_manifest", gold, "gold_manifest")

    registry_codes = [item["code"] for item in reasons.get("codes", [])]
    enum_codes = {code.value for code in ReasonCode}
    if set(registry_codes) != enum_codes:
        errors.append("reason_codes: YAML registry and Python ReasonCode enum differ")
    duplicates = [code for code, count in Counter(registry_codes).items() if count > 1]
    if duplicates:
        errors.append(f"reason_codes: duplicate codes: {duplicates}")
    for item in reasons.get("codes", []):
        if item["auto_reject_allowed"] and "reject" not in item["allowed_decisions"]:
            errors.append(
                f"reason_codes.{item['code']}: auto reject requires reject decision"
            )

    metric_names = [item["name"] for item in quality.get("metrics", [])]
    known_metrics = set(metric_names)
    if len(metric_names) != len(known_metrics):
        errors.append("quality_views: duplicate metric names")
    view_ids = [view["view_id"] for view in quality.get("views", [])]
    if len(view_ids) != len(set(view_ids)):
        errors.append("quality_views: duplicate view IDs")
    for view in quality.get("views", []):
        required = set(view["required_metrics"])
        optional = set(view["optional_metrics"])
        referenced = required | optional
        unknown = sorted(referenced - known_metrics)
        if unknown:
            errors.append(f"quality_views.{view['view_id']}: unknown metrics {unknown}")
        overlap = sorted(required & optional)
        if overlap:
            errors.append(
                f"quality_views.{view['view_id']}: metrics cannot be both required "
                f"and optional: {overlap}"
            )

    samples = gold.get("samples", [])
    profiles = Counter(sample["source_profile"] for sample in samples)
    expected_profiles = {
        "guida_ego",
        "dunjia_ego",
        "jianzhi_umi",
        "a2d_robot",
        "epic100",
    }
    if set(profiles) != expected_profiles or any(count < 2 for count in profiles.values()):
        errors.append("gold_manifest: every source profile needs at least two samples")
    for profile in expected_profiles:
        cases = {
            sample["case_type"]
            for sample in samples
            if sample["source_profile"] == profile
        }
        if "positive" not in cases or cases == {"positive"}:
            errors.append(
                f"gold_manifest.{profile}: needs a positive and a risk/boundary sample"
            )
    sample_ids = [sample["sample_id"] for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        errors.append("gold_manifest: duplicate sample IDs")
    unknown_reasons = sorted(
        {
            code
            for sample in samples
            for code in sample["reason_codes"]
            if code not in enum_codes
        }
    )
    if unknown_reasons:
        errors.append(f"gold_manifest: unknown reason codes {unknown_reasons}")
    if gold.get("status") == "frozen":
        designated_reviewer = gold.get("designated_reviewer")
        incomplete_reviews = [
            sample["sample_id"]
            for sample in samples
            if sample["review"]["status"] != "approved"
            or sample["review"]["reviewers"] != [designated_reviewer]
        ]
        if incomplete_reviews:
            errors.append(
                "gold_manifest: frozen status requires designated reviewer approval for "
                f"every sample: {incomplete_reviews}"
            )
    return errors


def main() -> int:
    errors = validate_wp0()
    if errors:
        print("WP0 validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print("WP0 validation passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
