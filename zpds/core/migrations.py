"""持久化元数据字段迁移。"""

from copy import deepcopy

LEGACY_FIELD_ALIASES = {
    "zrds_version": "zpds_version",
    "record_revision": "prep_revision",
}


def migrate_legacy_fields(data: dict) -> tuple[dict, list[str]]:
    """将 v0.x 旧字段迁移到 WP0 冻结的新字段。

    同时提供新旧字段且值不同属于不可解释冲突，必须失败。
    """

    migrated = deepcopy(data)
    changes: list[str] = []
    for legacy, canonical in LEGACY_FIELD_ALIASES.items():
        if legacy not in migrated:
            continue
        if canonical in migrated and migrated[canonical] != migrated[legacy]:
            raise ValueError(
                f"Conflicting values for legacy field {legacy!r} and canonical field "
                f"{canonical!r}"
            )
        migrated[canonical] = migrated.pop(legacy)
        changes.append(f"{legacy}->{canonical}")
    return migrated, changes
