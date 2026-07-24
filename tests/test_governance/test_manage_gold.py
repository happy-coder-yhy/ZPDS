from copy import deepcopy
from pathlib import Path

import pytest

from scripts.manage_gold import (
    collect_assets,
    freeze_manifest,
    hash_file,
    load_manifest,
    review_sample,
    verify_assets,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[2]
GOLD_MANIFEST = ROOT / "configs" / "gold" / "five_source_manifest.yaml"


def test_hash_file_and_collect_assets(tmp_path: Path) -> None:
    source = tmp_path / "source.bin"
    source.write_bytes(b"zpds-gold")
    manifest = {
        "samples": [
            {
                "sample_id": "sample",
                "review": {
                    "status": "approved",
                    "reviewers": ["owner"],
                    "reviewed_at": "2026-07-23T16:00:00+08:00",
                    "notes": "previous review",
                },
                "source_assets": [
                    {
                        "asset_id": "source",
                        "relative_path": "source.bin",
                        "sha256": "outdated",
                        "size_bytes": 0,
                    }
                ],
            }
        ]
    }

    expected_hash, expected_size = hash_file(source)
    updates, errors = collect_assets(manifest, tmp_path)

    assert errors == []
    assert updates == ["sample:source"]
    assert manifest["samples"][0]["source_assets"][0]["sha256"] == expected_hash
    assert manifest["samples"][0]["source_assets"][0]["size_bytes"] == expected_size
    assert manifest["samples"][0]["review"]["status"] == "pending"
    assert manifest["samples"][0]["review"]["reviewers"] == []
    assert verify_assets(manifest, tmp_path) == []


def test_collect_assets_rejects_path_escape(tmp_path: Path) -> None:
    manifest = {
        "samples": [
            {
                "sample_id": "sample",
                "source_assets": [
                    {
                        "asset_id": "unsafe",
                        "relative_path": "../outside.bin",
                    }
                ],
            }
        ]
    }

    _, errors = collect_assets(manifest, tmp_path)

    assert any("unsafe relative path" in error for error in errors)


def test_review_and_freeze_with_one_reviewer() -> None:
    manifest = load_manifest(GOLD_MANIFEST)
    for sample in manifest["samples"]:
        review_sample(
            manifest,
            sample_id=sample["sample_id"],
            reviewer=manifest["designated_reviewer"],
            status="approved",
            notes="fixture approval",
            reviewed_at="2026-07-23T16:00:00+08:00",
        )

    assert freeze_manifest(manifest) == []
    assert manifest["status"] == "frozen"


def test_freeze_rejects_pending_review() -> None:
    manifest = load_manifest(GOLD_MANIFEST)

    errors = freeze_manifest(manifest)

    assert errors
    assert manifest["status"] == "draft"


def test_manifest_write_is_atomic_and_reloadable(tmp_path: Path) -> None:
    manifest = deepcopy(load_manifest(GOLD_MANIFEST))
    destination = tmp_path / "gold.yaml"

    write_manifest(destination, manifest)

    assert load_manifest(destination) == manifest


def test_review_rejects_unknown_sample() -> None:
    manifest = load_manifest(GOLD_MANIFEST)

    with pytest.raises(ValueError, match="Unknown Gold sample"):
        review_sample(
            manifest,
            sample_id="missing",
            reviewer="owner",
            status="approved",
            notes="",
        )


def test_review_rejects_non_designated_reviewer() -> None:
    manifest = load_manifest(GOLD_MANIFEST)

    with pytest.raises(ValueError, match="designated reviewer"):
        review_sample(
            manifest,
            sample_id=manifest["samples"][0]["sample_id"],
            reviewer="someone-else",
            status="approved",
            notes="",
        )
