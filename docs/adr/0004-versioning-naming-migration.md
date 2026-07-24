# ADR-0004：版本、命名和兼容迁移

- 状态：Accepted
- 冻结日期：2026-07-23
- 适用版本：ZPDS 0.1.x

## 命名冻结

持久化字段统一使用：

- `zpds_version`：元数据遵循的 ZPDS Schema/规范版本；
- `prep_revision`：Prepared 产物修订号，格式为 `rNNNN`；
- `dataset_version`：数据集内容版本，使用 SemVer；
- `experience_version` / `annotation_version`：标注版本，使用 `vSemVer`；
- `release_id`：不可变交付快照的标识。

`zrds_version` 和 `record_revision` 是早期拼写，不再写入新产物。

## 版本边界

- Schema 不兼容变化：提升 `zpds_version` 的主版本；
- 向后兼容的新字段：提升次版本；
- 文档澄清或兼容修复：提升补丁版本；
- Raw 或清洗规则变化：创建新的 `prep_revision`，旧 revision 不覆盖；
- 标注变化：创建新的 annotation/experience version；
- 选择范围或 split 变化：创建新的 release。

每个 Prepared Revision 必须固定代码提交、管线版本、配置版本与
`config_uri` 和 `sha256` 配置哈希。配置内容变化但版本未变化属于验收失败。

## 兼容迁移

读取旧数据时允许：

| 旧字段 | 新字段 |
| --- | --- |
| `zrds_version` | `zpds_version` |
| `record_revision` | `prep_revision` |

迁移只能发生在显式兼容入口。若新旧字段同时出现且值相同，移除旧字段；值不同则立即
失败，因为程序无法安全判断哪个是真值。当前严格 Schema 只接受新字段，防止旧命名继续
扩散。

## 配置迁移

配置注册表记录当前版本、兼容范围、迁移列表和配置文件哈希。迁移必须是确定性的，
生成迁移报告，并保留输入版本；生产流程禁止在没有记录的情况下自动“猜默认值”。
