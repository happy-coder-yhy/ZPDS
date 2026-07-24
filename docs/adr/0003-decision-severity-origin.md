# ADR-0003：Decision、Severity、Evidence 与 Origin

- 状态：Accepted
- 冻结日期：2026-07-23
- 适用版本：ZPDS 0.1.x

## 问题与决策分离

一条 QC 结果由以下字段共同表达：

- `stage`：问题在哪个检查阶段发现；
- `reason_code`：机器稳定、可聚合的原因；
- `severity`：问题有多严重；
- `decision`：数据应该如何处置；
- `span`：影响哪个半开时间区间；
- `evidence`：审阅者到哪里复核；
- `producer/version/config_hash`：是谁、用什么版本和配置得出结果。

Severity 取 `info / warn / error / fatal`。Decision 取：

- `keep`：直接保留；
- `keep_with_flag`：可用但必须带风险标记；
- `quarantine`：进入隔离区，等待人工或更强检查；
- `trim`：删除首尾坏区间后保留；
- `split`：按坏区间或硬边界拆分；
- `reject`：整段拒绝。

Severity 和 Decision 互不替代。例如“失败后的恢复过程”可能是 `error`，但对研究
恢复行为有价值，因此可以是 `keep_with_flag`；未校准的模糊阈值通常先
`quarantine`，而不是自动 `reject`。

## 自动拒绝

只有高精度、可复现、证据充分的硬故障可以自动拒绝，例如哈希不匹配或容器确定损坏。
模糊、曝光、运动和语义模型在 Gold 集校准前不得自动拒绝长尾数据。

## Evidence

非 `keep` 决策至少需要一条 evidence。Evidence URI 可以引用报告、帧、时间窗、
日志、缩略图或可视化，但不能只写口头备注。涉及时间时使用 `[start_ns,end_ns)`。

Prepared Segment 的 `quality.decision` 表示最终处置，只允许 `keep`、
`keep_with_flag`、`quarantine`、`reject`。`trim` 和 `split` 是问题处理动作：
执行后产生的新 Segment 再记录其最终处置。所有 Stream 的 `origin.source_refs`
使用 `asset://<source_asset_id>`，并且必须能在顶层 `source_assets` 中找到。

## Origin

持久化值、流和标注必须声明以下来源之一：

- `source_recorded`
- `deterministic_transform`
- `model_estimated`
- `human_annotated`
- `simulation_ground_truth`
- `unknown`

`model_estimated` 必须声明模型名称和版本；确定性转换必须声明操作和源引用。来源未知
不会被伪装成人工真值，而是显式标记并进入对应质量门。

## Quality View

质量是多维向量，不合并成不透明总分。不同下游任务选择版本化 Quality View，
由 required/optional metrics 和 blocking decisions 决定是否适用。
