import pytest

from zpds.core.migrations import migrate_legacy_fields


def test_legacy_names_migrate_to_frozen_names() -> None:
    migrated, changes = migrate_legacy_fields(
        {"zrds_version": "0.1.0", "record_revision": "r0001"}
    )

    assert migrated == {"zpds_version": "0.1.0", "prep_revision": "r0001"}
    assert changes == [
        "zrds_version->zpds_version",
        "record_revision->prep_revision",
    ]


def test_matching_legacy_and_canonical_names_are_deduplicated() -> None:
    migrated, changes = migrate_legacy_fields(
        {"zpds_version": "0.1.0", "zrds_version": "0.1.0"}
    )

    assert migrated == {"zpds_version": "0.1.0"}
    assert changes == ["zrds_version->zpds_version"]


def test_conflicting_legacy_and_canonical_names_fail() -> None:
    with pytest.raises(ValueError, match="Conflicting values"):
        migrate_legacy_fields(
            {"zpds_version": "0.1.0", "zrds_version": "9.9.9"}
        )
