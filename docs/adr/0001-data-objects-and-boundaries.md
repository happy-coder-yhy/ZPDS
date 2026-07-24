# ADR-0001：数据对象、边界与语义单位

- 状态：Accepted
- 冻结日期：2026-07-23
- 适用版本：ZPDS 0.1.x

## 背景

五类来源同时包含文件、采集会话、传感器流、连续片段、场景、动作和训练样本。
如果把这些概念都叫“segment”，同一条数据会因不同算法被反复物理切割，造成
资产复制、时间错位和版本不可追踪。

## 决策

ZPDS 使用以下对象层级：

1. `Raw Session`：不可修改的采集事实，只登记 URI、哈希、许可、隐私和来源。
2. `Prepared Revision`：一次确定的清洗规则、代码和配置所产生的版本快照。
3. `Prepared Segment`：只表示解码、时钟和关键流在物理上连续的区间。
4. `Annotation`：附着于 Segment 时间轴的版本化解释，不改写 Prepared 数据。
5. `Scene`：环境或上下文相对稳定的语义区间。
6. `Action`：主体执行的原子动作区间。
7. `CEU`：面向训练/评测的 Canonical Experience Unit，可引用多个流和资产。
8. `Release`：明确选择 Prepared Revision、Annotation 版本和确定性 split 的交付物。

所有时间区间均为半开区间 `[start_ns, end_ns)`。Prepared Segment 的本地时间轴从
`0` 开始，且必须能映射回 Raw Session 的源时钟区间。

只有以下硬边界可以新建 Prepared Segment：

- 容器或关键流无法解码；
- 时间戳回退、时钟复位或不可跨越的长 gap；
- 关键流缺失，导致区间无法满足 Profile 的最低物理要求；
- 采集文件天然分段且无法证明可无损拼接。

Scene、Action、任务和 CEU 边界只进入 Annotation。它们不能为了语义方便而切开
Prepared Segment。

## 可解释性要求

审阅者应能沿以下链路回答一个 Segment：

`segment_id → source_session/source_assets → source_span → streams/origin →`
`quality.issues/evidence → decision → prep_revision/producer/config_hash`

边界如果来自 QC，必须带 `reason_code`、决策、时间范围和可访问的 evidence URI。

## 后果

- 同一 Prepared 数据可以承载多版 Scene/Action/CEU 标注。
- Exporter 只做训练视图转换，不在导出时偷偷修复或重新切分。
- Segment Schema 不接收 `scene_boundary`、`action_boundary` 或 `ceu_boundary` 字段。
