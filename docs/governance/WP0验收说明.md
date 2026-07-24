# WP0 标准冻结与治理验收说明

## 本阶段交付物

| 交付 | 仓库位置 | 机器验收 |
| --- | --- | --- |
| 数据对象、边界、单位、Decision、Origin ADR | `docs/adr/` | ADR 文件存在且状态 Accepted |
| 核心对象 Schema | `zpds/schemas/*.schema.json` | Draft 2020-12 Schema 自检 |
| 最小合法对象 | `examples/schemas/*.json` | 每个样例通过对应 Schema |
| Reason Code Registry | `configs/reason_codes/v0.1.0.yaml` | Registry Schema、枚举一致性、唯一性 |
| Quality View Registry | `configs/quality_views/v0.1.0.yaml` | Registry Schema、指标引用完整性 |
| 版本和迁移规则 | `configs/governance/versioning.yaml` | Config Schema、迁移单测 |
| 五源 Gold Manifest | `configs/gold/five_source_manifest.yaml` | Manifest Schema、每源正/反例覆盖 |
| Gold 审核工具 | `scripts/manage_gold.py` | 资产 Hash、单人审核、冻结前复验 |

## 审阅者如何解释一个 Segment

先查看正常样例 `examples/schemas/segment.json`，再查看带证据的隔离样例
`examples/schemas/segment_quarantine.json`：

1. `timeline` 是本 Segment 的连续本地时间轴，从 0 开始，区间为左闭右开。
2. `source_span` 和 `source_assets` 说明它来自哪次 Raw Session 的哪个原始区间。
3. `streams[].origin` 说明每条流是原始记录、确定性变换、模型结果还是人工标注。
4. `quality.decision` 明确最终处置；`quality.issues[]` 说明问题、严重度、局部决策
   以及可复核证据。
5. `prep_revision`、`producer.code_commit`、`config_version`、`config_uri` 和
   `config_hash` 固定产生它的代码与配置版本。
6. Scene、Action 和 CEU 不改变物理边界，而是通过版本化 Annotation 引用时间段。

## Gate 0 判定

自动化检查全部通过只代表“机器契约完成”。Gold Manifest 初始状态为 `draft`；
五源样例必须由项目指定负责人复核并把各 sample 的 review 改为 `approved`，
随后将 manifest 状态改为 `frozen`，才算正式通过 Gate 0。不要求双人审核或
ADR 签字；审核身份、结论和对应版本通过 Manifest 与 Git 历史留痕。

任何字段冲突、单位不明、时钟映射无法证明或来源缺失，均不能通过口头约定放行：
必须进入 Reason Code、Evidence、Decision 和版本记录。

具体人工操作见 `docs/governance/Gold样例人工确认指南.md`。
