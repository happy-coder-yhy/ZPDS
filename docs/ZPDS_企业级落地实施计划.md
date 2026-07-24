# ZPDS 企业级落地实施计划

> 版本：v1.0-draft
> 编制日期：2026-07-23
> 适用范围：ZPDS 多源数据登记、读取、清洗、Prepared Segment、质量控制、CEU、Release 与训练格式导出
> 计划基线：`hwx` 分支 `f6cb0f0`，并参考 `origin/llccxx` 的 Guida Prepared Segment POC

## 1. 执行摘要

ZPDS 的产品定位是原始具身智能数据和训练数据集之间的规范化经验中间层。它不替代 MP4、MCAP、Parquet、LeRobot 或 RLDS，而是负责：

- 登记并保护不可变 Raw Session；
- 识别不同来源的文件、流、时钟、字段、单位和坐标；
- 恢复多模态真实时间轴并生成可审计的 sample map；
- 执行分级质量控制，形成可解释的保留、隔离、裁剪、切分和拒绝决策；
- 生成版本化 Prepared Segment；
- 保存独立版本化的 Scene、Action、CEU 和大型标注资产；
- 固定 Prepared Revision 与 Annotation Version，形成可复现 Release；
- 导出 LeRobot、RLDS 等训练视图。

当前仓库已经有较合理的分层骨架，但主体仍是 Pre-Alpha：

- `hwx/main` 保留了 `zpds/` 分层、配置和脚本骨架，但大量功能尚未实现，测试目录为空；
- `origin/llccxx` 已实现 Guida 的部分 Prepared Segment POC，但属于实验型平铺脚本，存在硬编码路径、魔法样本数、无自动化测试、删除主包结构等问题；
- 数据标准主要存在于 PDF 和 Markdown 中，尚未转成机器可验证的 Schema；
- 缺少 CI、质量门、任务恢复、审计、监控、安全隔离和正式发布流程。

因此，本计划不建议在 8–12 周内直接宣称“企业级完成”。建议目标拆为：

- **8–10 周：五源基础 Pilot**，证明数据契约、读取、时间恢复和 Prepared 路径可行；
- **18 周：Production Candidate**，达到可控批量生产、可审计、可恢复、可发布；
- **上线后 4 周：受控试运行**，至少完成三轮真实批次运行和问题闭环后再进入正式生产。

推荐团队为 6–8 个全职等效人力。若只有 3 人，计划应调整为 28–32 周。

## 2. 计划依据与优先级

本计划按以下顺序处理资料冲突：

1. 已确认的 ZPDS 数据标准及其最小校验规则；
2. `ZPDS_多源清洗流程.md` 中的当前 Prepared Segment 实施要求；
3. 已有综合 `plan.md` 中的阶段设计和五源风险；
4. 当前仓库中已经存在且通过测试的行为；
5. `23.pdf` 中的研究路线、模型选型和远期能力。

研究型能力只有在基础数据契约和低成本质量门稳定后才进入生产计划。3D 重建、全量 VLM、世界坐标手部重建、人形机器人动作重定向和硬件研发不进入首个 Production Candidate 的强制范围。

## 3. 已冻结的产品原则

### 3.1 Raw 与派生数据

- Raw Session 是不可变事实来源，只读保留，不覆盖、不移动、不物理删除。
- `reject` 表示不进入后续生产，不表示删除 Raw。
- 所有派生产物必须能回溯到 Raw asset、source sample、处理程序、配置和版本。
- Raw、Prepared、Annotation、Release 和 Export 必须使用不同的生命周期与访问权限。

### 3.2 时间层级

- Session 表示一次连续采集、一次机器人运行或一个原始 episode。
- Prepared Segment 只按解码、时钟和关键流的物理连续性切分。
- Scene、Action、Task 和 CEU 边界属于版本化 Annotation，不反向改写 Prepared。
- Segment 内时间轴从 `0 ns` 开始，`timeline.end_ns > 0`，并声明连续。
- 规范化时间统一使用 `int64 ns`，同时保留原始时间值和时钟来源。
- 禁止用“第 N 帧对应第 N 行”代替时间映射。
- 禁止跨 clock reset 或长 gap 插值。

### 3.3 单位、坐标与姿态

- 长度单位：米 `m`。
- 时间单位：纳秒 `ns`。
- 角度单位：弧度 `rad`。
- 四元数顺序：`xyzw`。
- Pose 表达：`T_parent_child`。
- 所有空间流必须声明 `frame_id`；所有变换必须声明父子 Frame、方向、来源和置信状态。
- 原始值、原始单位、转换规则和规范化结果同时可追溯。

### 3.4 重采样

- Prepared 默认保留原始频率，不为“看起来整齐”而强制统一 Hz。
- CFR 视频和统一训练时间网格属于显式派生视图。
- 任何非简单一一映射都必须生成 sample map，包含 source sample、target sample、映射方法和时间误差。
- 30 Hz perception 与 50 Hz control 等视图写入 `alignments/`，不能覆盖原始时序。

### 3.5 质量决策

问题严重度和数据处理决策必须分离：

- Severity：`INFO`、`WARN`、`ERROR`、`FATAL`；
- Decision：`keep`、`keep_with_flag`、`quarantine`、`trim`、`split`、`reject`。

自动 `reject` 只适用于高精度、可解释的硬故障。未经金标集校准的模糊、曝光、运动和语义规则默认进入 `quarantine`。

### 3.6 标注可信度

标注与派生值必须声明 Origin：

- `source_recorded`
- `deterministic_transform`
- `model_estimated`
- `human_annotated`
- `simulation_ground_truth`
- `unknown`

模型估计结果不得冒充人工标注或仿真真值；每个模型产物必须记录模型名、版本、配置哈希和运行批次。

## 4. 当前仓库差距评估

| 领域 | 标准要求 | `hwx/main` 现状 | `origin/llccxx` 可复用资产 | 处置 |
|---|---|---|---|---|
| 包结构 | 分层、可扩展、可安装 | 已有 `zpds/` 骨架 | 删除主包，改为根目录脚本 | 以 `zpds/` 为唯一架构基线 |
| 核心 Schema | Dataset/Revision/Segment/CEU/Release 可校验 | 只有少量 dataclass | 直接拼接字典 | 建立 Pydantic/JSON Schema |
| Adapter | 统一接口、流式读取、五源 Profile | 仅抽象骨架 | 只有 Guida 专用脚本 | 将 POC 代码迁入 Adapter/Profile |
| 时间与 Sample Map | 多 Clock、显式映射和误差 | 类型不足 | Guida 有初版 nearest map | 重构为通用 Clock/SampleMap 服务 |
| Prepared | 原子写入、回读验证、Revision | Writer/Validator 未实现 | 有 Writer/Validator POC | 迁移后补 Schema、事务和测试 |
| QC | Stage 0–12、Decision/Evidence | 13 个占位 Stage | 少量黑屏/时间检查 | 建立统一插件契约和级联调度 |
| Annotation/CEU | 独立版本、资产引用和 Origin | 占位 | 无 | Production Candidate 前完成最小版本 |
| Release/Export | 固定版本、split、LeRobot/RLDS | 占位 | 无 | 后期实现并端到端回读 |
| 测试 | 五源 fixture、回归和 E2E | 无有效测试 | 无 | 从第一阶段建立质量门 |
| 工程化 | CI、日志、重试、恢复、指标 | 无 | 无 | 建立 Pipeline Run Ledger 与 CI |
| 安全治理 | Pickle 隔离、许可、隐私、审计 | 无 | 无 | 作为 Stage 0 强制能力 |

### 4.1 `origin/llccxx` 迁移原则

不得整分支合并。可迁移的算法资产包括：

- Guida `index.jsonl` 解析思路；
- Span 初版判定逻辑；
- 视频转码与 sample map POC；
- IMU 规范化；
- Calibration 提取；
- Segment JSON 构造；
- 写出后验证检查项。

迁移时必须去除：

- `E:/datasets/...` 等硬编码路径；
- 固定 `983` 样本等数据集特例；
- 固定 `seg_000001`、`guida_session_001` 等 ID；
- 根目录平铺模块；
- 宽松的 `±2` 帧无依据容差；
- 直接使用字典且缺少 Schema 的接口；
- 在库代码中使用 `print`；
- 未声明的 OpenCV/编码回退行为。

迁移完成的标准是：代码位于 `zpds/` 正确分层、配置驱动、类型明确、具有 fixture 测试，并通过同一份 Guida 金标集。

## 5. 目标架构

```text
                    ┌──────────────── Control Plane ────────────────┐
                    │ Config Registry / Schema Registry             │
                    │ Run Ledger / Audit / Review / Release Manager │
                    └──────────────────────┬────────────────────────┘
                                           │
Immutable Raw Store
    │
    ▼
Source Registry ──> Profile Resolver ──> Container Adapters
                                           │
                                           ▼
                              Stream Catalog + Clock Catalog
                                           │
                                           ▼
                                QC Stage 0–8 + Evidence
                                           │
                                           ▼
                       Physical Span / Trim / Split / Quarantine
                                           │
                                           ▼
               Prepared Writer ──> Validator ──> Prepared Revision
                                           │
                      ┌────────────────────┴────────────────────┐
                      ▼                                         ▼
          QC Stage 9–11 / Review                     Alignment Views
                      │
                      ▼
       Versioned Annotation / CEU / Assets
                      │
                      ▼
         Stage 12 Delivery Validation
                      │
                      ▼
           Release ──> LeRobot / RLDS
```

### 5.1 数据面模块

- `zpds/core/`：持久化 Schema、ID、时间、单位、Decision、Evidence、Provenance；
- `zpds/adapters/`：视频、MCAP、ROSBag、HDF5、日志、隔离 Pickle；
- `zpds/profiles/`：来源文件模式、Topic/Key、字段映射、单位、Frame、必需流；
- `zpds/pipeline/`：阶段调度、幂等执行、断点恢复、Run Ledger；
- `zpds/qc/`：Stage 0–12 插件与结果聚合；
- `zpds/segmentation/`：Physical Span、Trim、Split；
- `zpds/prepared/`：写入、Revision、Validator、Sample Map、Alignment；
- `zpds/annotation/`：CEU、Manifest、Asset Store、Review；
- `zpds/release/`：版本冻结、选择条件、split；
- `zpds/export/`：LeRobot、RLDS；
- `zpds/observability/`：结构化日志、指标、成本与质量统计。

### 5.2 控制面原则

首个 Pilot 使用 CLI 与本地 Run Ledger，不立即引入重型工作流平台。接口稳定后再接入企业调度系统。

建议 CLI：

```text
zpds inventory
zpds inspect
zpds prepare
zpds qc
zpds review
zpds validate
zpds release
zpds export
zpds run
```

每次运行生成唯一 `run_id`，记录：

- 输入 Dataset/Session；
- Git commit、包版本和配置哈希；
- Stage 状态与起止时间；
- 输入/输出 Artifact；
- Warning/Error 与重试；
- CPU/GPU 时间和处理成本；
- 最终 Manifest 哈希。

## 6. 机器可执行的数据契约

### 6.1 必须建立的 Schema

在仓库增加 `schemas/`，至少包含：

- `dataset.schema.json`
- `revision.schema.json`
- `segment.schema.json`
- `calibration.schema.json`
- `quality_report.schema.json`
- `experience_manifest.schema.json`
- `ceu.schema.json`
- `release.schema.json`
- `run_manifest.schema.json`
- `gold_sample.schema.json`

Python 侧使用 Pydantic v2 作为边界校验，JSON Schema 作为跨语言和持久化合同。内部高吞吐结构可使用 dataclass，但进入磁盘前必须通过正式 Schema。

### 6.2 Segment 最小合同

每个 `segment.json` 至少包含：

- 标准版本、Record/Prepared Revision、Segment ID；
- Source Session 与 Source Assets；
- Segment Timeline 与 Source Span；
- Stream 列表；
- Calibration 引用或明确 unavailable；
- Quality 状态、issues、decisions 和 evidence；
- Provenance 与 config hash。

每个 Stream 至少声明：

- `stream_id`
- `role`
- `modality`
- `uri`
- `format`
- `time`
- `origin`
- 表格流的 `fields`
- 空间流的 `frame_id`
- 数组流的 `shape/dtype`
- 非简单映射的 `sample_map_uri`

### 6.3 Prepared 目录

```text
<dataset_id>/
├── dataset.json
├── prepared_segments/<prep_revision>/
│   ├── revision.json
│   └── <segment_id>/
│       ├── segment.json
│       ├── data/
│       ├── calibration/
│       ├── maps/
│       ├── alignments/
│       └── reports/
├── experiences/<experience_version>/
│   ├── experience_manifest.json
│   ├── ceus.parquet
│   └── assets/
├── releases/<release_id>.json
└── exports/{lerobot,rlds}/<release_id>/
```

Prepared 写入采用临时目录、完整校验、原子重命名。失败运行不得留下可被误判为成功的最终目录。

## 7. QC 级联设计

### 7.1 原子 Stage

| Stage | 领域 | Production Candidate 范围 |
|---:|---|---|
| 0 | 文件、Hash、来源、许可、隐私 | 必须 |
| 1 | 容器、索引、Schema、可解码性 | 必须 |
| 2 | 时间单调、Gap、Clock Model、同步 | 必须 |
| 3 | 黑屏、曝光、模糊、冻结、坏帧 | 必须 |
| 4 | 丢帧、重复帧、VFR、运动异常 | 必须 |
| 5 | 深度 dtype、单位、invalid、配准 | 有该模态时必须 |
| 6 | IMU 频率、Gap、偏置、饱和 | 有该模态时必须 |
| 7 | Robot State/Command/Gripper | 机器人来源必须 |
| 8 | 标定完整性、方向、重投影 | 有标定时必须 |
| 9 | 手部出现、跟踪、Pose Ready | 分级启用 |
| 10 | 任务、动作、语义、敏感内容 | Pilot + 人审 |
| 11 | 完全重复、近重复、分布偏差 | Release 前必须 |
| 12 | 文件、Schema、引用、回读、签名 | Release 前必须 |

### 7.2 成本级联

- Gate A：Stage 0–2，低成本硬故障；
- Gate B：Stage 3–8，模态与物理质量；
- Gate C：Stage 9–10，手部、动作和语义候选；
- Gate D：Stage 11，去重与分布；
- Gate E：Stage 12，交付验证。

高成本模型只能处理通过 Gate A/B 的短片段和争议样本。

### 7.3 QC 统一输出

每项检查必须表达：

- metric name；
- value 和 unit；
- applicability；
- severity；
- decision；
- reason code；
- source/segment span；
- evidence URI；
- producer/version/config hash；
- reviewer 与 override（如果有）。

质量不能只保存一个总分。至少提供技术可读性、时间连续性、视觉质量、同步质量、标定质量、动作可用性和标注可信度等维度。

## 8. 五源实施与验收

| 来源 | 首要风险 | 实施重点 | 来源级验收 |
|---|---|---|---|
| Guida ego | 容器 FPS 与 `index.jsonl` 时间混用；IMU 路径不一致 | 以 `timestamp_ns` 为权威；RGB/Depth 配对；CFR sample map | 任意输出帧可回溯到 seq 和源时间；预览 MP4 不作权威源 |
| 遁甲 MCAP | Topic/Schema 差异；多路 H264 与双时间 | 保留 message time 与 log time；按 GOP 重建；多相机覆盖统计 | H264 帧时间与消息时间一致；Topic/Schema/Gap 可查询 |
| UMI MCAP | robot0/1 混淆；VIO reset；编码器语义未知 | 双 stream group；鱼眼/外参；原始编码器 scalar | 双端不混淆；VIO reset 不插值；编码器映射可追溯 |
| A2D | 多套时间轴；稀疏相机目录；行号硬配对 | 六文件 completeness matrix；日志恢复映射；Robot Command/State | 不使用行号同步；冲突和推断误差明确；失败/恢复分开 |
| EPIC 衍生包 | 不可信 Pickle；原视频版本与许可 | 隔离解析；Schema 转换；orphan 检查；模型 Origin | 不执行未知代码；每条标注绑定确定视频或明确 orphaned |

来源接入顺序：

1. Guida 作为 Reference Implementation；
2. 遁甲与 UMI 复用 MCAP 平台；
3. A2D 验证复杂多容器与机器人时间对齐；
4. EPIC 验证外部标注、安全隔离和许可治理。

## 9. 工作包

### WP0：标准冻结与治理

交付：

- 数据对象、边界、单位、Decision 和 Origin ADR；
- 机器可执行 Schema；
- Reason Code Registry；
- Quality View；
- 配置版本和迁移规则；
- 五源 Gold Manifest；
- `zrds_version`/`zpds_version`、`record_revision`/`prep_revision` 命名决议。

验收：

- 任意审阅者能解释一个 Segment 的边界、问题、证据、决策和版本；
- 所有最小示例均通过 Schema；
- 冲突字段不存在口头约定。

### WP1：平台基础与运行框架

状态（2026-07-24）：工程实现与自动验收已完成。Gate 1 尚未完成，仍需 WP2 的 Guida
Adapter、真实 Stage 0～2 和 Prepared 最小闭环。

交付：

- 可安装包和 CLI；
- Pipeline Stage 接口；
- Run Ledger；
- 幂等、重试、断点恢复；
- 结构化日志和指标；
- Config Loader、Schema Registry；
- 本地文件系统 Storage Adapter，预留对象存储接口。

验收：

- 中断后重跑只执行未完成 Stage；
- 相同输入、代码和配置得到相同 Manifest；
- 基础包不因缺少 Torch/MCAP 等可选依赖而导入失败。

### WP2：Adapter 与 Profile

状态（2026-07-24）：工程实现、自动测试和五源真实只读验收已完成。已交付统一
Source Inventory、Stream/Clock/Calibration Catalog、完整读取/解码入口，以及 Guida
Stage 0～2 纵向闭环。补充验收已覆盖 HDF5 全量分块遍历、MCAP 内嵌媒体负载解码和
EPIC primitive-only Pickle 隔离内容摘要。

交付：

- Video、MCAP Protobuf、ROS2 CDR、HDF5、Log、隔离 Pickle Adapter；
- 五源 Profile；
- Stream/Clock/Calibration Catalog；
- 全量读取/解码报告；
- 小型可提交 Fixture。

验收：

- 五源均可生成 Inventory、结构报告和 Hash；
- 所有关键流完成全量读取或解码；
- Reader 使用流式/分块接口，内存有界。

### WP3：Clock、Sample Map 与 Prepared

状态（2026-07-24）：Guida Reference Slice 已完成并通过真实数据验收。当前实现包括
权威时间恢复、video/IMU Sample Map、基础硬质量检查、物理区间 trim/split、规范化 IMU、
Calibration 记录、Prepared 原子写入和 Raw hash 回读。其余四源尚未进入 Prepared；
RGB/Depth 当前是明确标注 `materialized: false` 的 source selection，不等同于媒体物化。
因此 WP3 只能表述为“Guida 基础闭环完成”，不能表述为“五源 WP3 完成”。

交付：

- Clock Model；
- Source Span 与 Segment Timeline；
- Sample Map；
- 单位/坐标规范化；
- Prepared Writer、Revision Manager、Validator；
- Alignment View；
- Guida Reference Pipeline。

验收：

- 任意输出样本可定位到 Raw；
- 所有时间序列单调且位于 Segment 范围内；
- 写出后全量回读成功；
- 同 Revision 重跑 Manifest Hash 一致。

### WP4：QC 与 Physical Segmentation

交付：

- Stage 0–8；
- QCCascade；
- Decision/Evidence；
- Span Proposer、Trimmer、Splitter；
- QC 报告；
- 金标评估工具。

验收：

- 硬故障金标召回率 100%；
- 自动 reject 精确率达到 100%，否则降级为 quarantine；
- Trim/Split 边界误差满足各 Profile 金标标准；
- 阈值附近样本进入人审。

### WP5：Annotation、Review 与高成本模型

交付：

- Experience Manifest、CEU、Asset Store；
- Review API/UI；
- 稀疏手检测、跟踪、自适应加密；
- 模型/人工 Origin；
- Scene/Action Proposal；
- VLM 争议复核。

验收：

- Review 修改有审计记录；
- CEU 完全位于 Prepared Segment；
- 3/10/30 FPS 手部方案完成成本、召回和连续性对比；
- 模型输出均可定位到模型版本和证据帧。

### WP6：Release、Export 与生产交付

交付：

- Release Manager；
- 确定性、无泄漏 split；
- LeRobot v3 Exporter；
- RLDS Exporter；
- Stage 11/12；
- Dataset Card、质量报告和签名 Manifest。

验收：

- Release 固定 Dataset、Prepared Revision、Annotation Version 和 Split；
- LeRobot/RLDS 可回读；
- 同 Session/Group 不跨 train/validation/test；
- Release 中不存在悬空 Stream/Asset/Annotation 引用。

### WP7：工程质量、安全与运维

交付：

- CI、代码检查、类型检查和测试报告；
- 依赖、许可证和漏洞扫描；
- Pickle 隔离运行时；
- 备份、保留和删除策略；
- 可观测性 Dashboard；
- Runbook、故障演练和发布回滚。

验收：

- 主分支所有质量门通过；
- 隔离解析无法访问网络和生产凭据；
- 一次 Worker 中断和一次存储失败演练通过；
- 可根据 `run_id` 定位全部日志、输入和输出。

## 10. 18 周排期

| 阶段 | 周期 | 主要工作 | 退出门 |
|---|---:|---|---|
| M0 标准冻结 | W1–W2 | WP0、金标协议、Schema、ADR | Gate 0：合同冻结 |
| M1 Reference Slice | W3–W5 | WP1、Guida Adapter、Stage 0–2、Prepared 最小闭环 | Gate 1：Guida E2E |
| M2 五源接入 | W6–W9 | Video/MCAP/HDF5/Log/Pickle、五源 Inventory、Stage 0–6 | Gate 2：五源只读接入 |
| M3 物理质量生产 | W10–W12 | Stage 3–8、Clock/Sample Map、Trim/Split、Prepared Revision | Gate 3：五源 Prepared Pilot |
| M4 标注与复核 | W13–W15 | Stage 9–11、Review、CEU、手部/Scene/VLM 分级流程 | Gate 4：可审阅 Annotation |
| M5 发布与硬化 | W16–W18 | Stage 12、Release、LeRobot/RLDS、性能、安全、运维 | Gate 5：Production Candidate |
| 受控试运行 | 上线后 4 周 | 三轮真实批次、问题闭环、SLO 验证 | Production Go-Live |

### 10.1 里程碑定义

**Gate 0：合同冻结**

- Schema v0.1；
- Decision、Reason Code、Origin 和 Quality View 完成；
- 五源 Gold Manifest 可加载；
- 命名和单位冲突已记录确认。

**Gate 1：Guida E2E**

- Raw 登记到 Prepared 回读全链路；
- `index.jsonl` 为权威时间；
- RGB CFR sample map 与 IMU 原频率保留；
- 无硬编码路径和固定样本数；
- E2E Fixture 在 CI 运行。

**Gate 2：五源只读接入**

- 五源 Inventory、Stream/Clock Catalog；
- 全量读取/解码报告；
- 许可与隐私状态；
- 不产生规范化数据也能安全完成检查。

**Gate 3：五源 Prepared Pilot**

- 每源至少 20 个金标/反例样本；
- Clock、单位、Frame、Sample Map 和 Calibration 可查询；
- Prepared 写出和回读验证；
- Pass/Quarantine/Reject 解释完整。

**Gate 4：可审阅 Annotation**

- Review UI 能展示证据并记录人工 Override；
- Scene/Action/CEU 不修改 Prepared；
- 模型估计带完整 Provenance；
- 高成本模型预算可统计。

**Gate 5：Production Candidate**

- 可固定 Release；
- LeRobot/RLDS 端到端回读；
- 性能、恢复、安全和数据治理验收；
- 三类用户完成 UAT：数据工程、算法训练、质量审阅。

## 11. 前 10 个工作日任务清单

### 第 1 周

1. 建立 ADR：
   - Prepared/Scene/Action/CEU 边界；
   - `zpds_version` 与 PDF 中 `zrds_version` 的兼容方案；
   - Prepared Revision 字段命名；
   - 长度 `m`、时间 `ns`、四元数 `xyzw`、Pose `T_parent_child`。
2. 建立正式 Decision、Evidence、Origin 和 Reason Code 模型。
3. 建立 Dataset/Revision/Segment 最小 Pydantic 模型和 JSON Schema。
4. 建立五源 Profile Schema，不实现具体解析。
5. 定义 Gold Manifest 格式和评审流程。

### 第 2 周

1. 建立 pytest、Ruff、mypy 和 CI。
2. 为核心 Schema、单位、时间和 Decision 编写测试。
3. 建立 `PipelineStage` 与 `RunManifest`。
4. 从 `origin/llccxx` 迁移 Guida index 解析和 sample map 算法，去除硬编码。
5. 制作 Guida 最小 Fixture。
6. 完成 Gate 0 评审，不通过则不开始五源并行开发。

## 12. 测试与质量策略

### 12.1 测试金字塔

- 单元测试：Schema、时间、单位、Hash、Decision、Reason Code、算法边界；
- Contract Test：所有 Adapter/Profile 必须通过同一套契约测试；
- Fixture Integration：每种来源至少一个正常、损坏、临界和不确定样本；
- Golden Regression：决策、边界和 Manifest Hash 与金标比较；
- E2E：Raw 到 Prepared、Annotation、Release、Export 回读；
- Performance：真实大小样本上的吞吐、峰值内存、临时空间和模型成本；
- Security：不可信 Pickle、路径穿越、恶意压缩包、超大字段和资源限制。

### 12.2 CI 强制门

- `ruff check` 通过；
- `mypy` 对核心和公共接口通过；
- 单元与 Contract Test 通过；
- 核心数据合同覆盖率不低于 90%；
- 总体行覆盖率在 Production Candidate 前不低于 80%；
- Schema 示例和迁移测试通过；
- 不包含密钥、绝对数据路径和大型二进制；
- 依赖许可证与漏洞扫描无阻断项。

## 13. 非功能指标

### 13.1 正确性

- 100% Raw Asset 进入 Inventory 并具备 Hash 或明确 Hash Pending 状态；
- 100% Prepared Stream 声明 Origin；
- 100% 非简单重采样具备 Sample Map；
- 100% Release 引用通过完整性验证；
- 相同输入、代码和配置重跑得到相同 Manifest Hash。

### 13.2 可靠性

- Stage 幂等；
- Worker 异常后可以从最近成功 Stage 恢复；
- Prepared 使用原子发布；
- 失败任务不产生成功标记；
- 任何隔离、人工修改和 Release 操作均有审计记录。

### 13.3 性能

硬件基线在 M1 固定，所有数字基于同一参考节点报告。Production Candidate 最低要求：

- 除模型权重外，单 Worker 峰值内存不高于 4 GB；
- 读取和转换使用流式或分块方式；
- Stage 0–2 不重复解码同一媒体；
- 已完成 Stage 在重跑时不重复执行；
- 高成本模型按处理分钟数、GPU 秒数和单 Segment 成本统计；
- 三个标准规模批次在容量测试中无内存泄漏和临时磁盘失控。

不设置跨设备统一的“每小时视频处理时间”承诺；M1 基准测试后，按 CPU/GPU、编码和存储类型制定 Profile 级 SLO。

### 13.4 可观测性

至少采集：

- Run/Stage 成功率、耗时和重试；
- 输入字节、输出字节和处理时长；
- 各 Decision/Reason Code 数量；
- 各源保留率与清洗前后分布；
- 时间同步误差、Sample Map 误差；
- 模型成本和人工审阅耗时；
- Schema/回读失败率；
- Quarantine 积压量。

## 14. 安全、许可与数据治理

- Raw Store 与 Prepared Store 使用不同写权限；
- Pipeline 对 Raw 只有读权限；
- EPIC Pickle 在无网络、最小权限、资源受限容器中解析；
- 所有外部路径做目录穿越和符号链接检查；
- Dataset 必须包含许可证、用途、隐私和再分发状态；
- 未知或禁止用途的数据在 Stage 0 阻断 Release；
- PII/敏感内容检查结果进入审计，不直接混入普通日志；
- 日志不得记录访问令牌、完整隐私内容和内部凭据；
- 模型权重、数据和代码分别做版本与供应链记录；
- Release 生成 Dataset Card、质量统计和已知限制。

## 15. 团队配置

推荐 6–8 FTE：

| 角色 | 人数 | 主要责任 |
|---|---:|---|
| 产品负责人/数据标准 Owner | 1 | 范围、标准、验收和跨团队决策 |
| 技术负责人/架构师 | 1 | Schema、架构、质量门和技术评审 |
| 数据平台工程师 | 2 | Pipeline、存储、运行框架、Prepared、Release |
| 多模态/机器人数据工程师 | 2 | Adapter、Clock、Calibration、Robot QA |
| CV/ML 工程师 | 1 | Hand、Scene、VLM、模型评估与成本 |
| QA/数据质量工程师 | 1 | Gold、回归、Review、UAT、质量报告 |
| DevOps/Security | 0.5–1 | CI、运行环境、监控、安全隔离和发布 |
| 数据审阅专家 | 2 人兼职 | 金标、临界样本和业务解释 |

若人力不足，优先减少高成本 Annotation 范围，不能削减 Schema、Clock、Provenance、Validator 和测试。

## 16. 分支与发布策略

- `main`：受保护，只接收通过质量门的变更；
- `hwx`：当前集成开发分支；
- `origin/llccxx`：视为 POC 来源，不作为合并基线；
- 功能分支：从 `hwx` 创建，保持小批次；
- Schema 变更必须附迁移说明和兼容测试；
- 版本采用 SemVer；
- Prepared Revision、Annotation Version、Release ID 独立；
- 禁止在 Release 中只引用“latest”；
- 每个 Release 固定 Git commit、配置哈希、Schema 版本和 split。

建议保护规则：

- 至少一名代码审阅者和一名数据标准审阅者；
- CI 全绿；
- 禁止直接 force push 到 `main`；
- Adapter 合并前必须提供 Fixture 和 Contract Test；
- QC 规则合并前必须报告金标表现和影响分布。

## 17. 风险登记

| 风险 | 概率/影响 | 缓解措施 | 触发升级条件 |
|---|---|---|---|
| A2D 无真实逐帧时间映射 | 高/高 | 日志证据、误差模型、Quarantine | 无法给出可接受误差上限 |
| UMI Encoder/VIO 语义不明 | 高/中 | 原始值保留、禁止提前命名、供应商确认 | 影响 action 定义或对齐 |
| EPIC Pickle 安全与许可 | 高/高 | 隔离解析、版本/hash 绑定、许可 Gate | 无法确认视频版本或再分发权 |
| 标准字段命名冲突 | 中/高 | M0 ADR、Schema 兼容层 | Gate 0 未记录确认 |
| 多源 Profile 过度硬编码 | 高/高 | Contract Test、Profile 配置驱动 | 同类修复在多个 Adapter 重复 |
| 模型成本失控 | 中/高 | 低成本 Gate、自适应采样、预算指标 | GPU 成本超预算 20% |
| 误删长尾数据 | 中/高 | Raw 不变、Quarantine、自动 Reject 高精度要求 | Reject 金标精确率低于 100% |
| 清洗导致分布偏差 | 中/高 | 清洗前后分布报告、分层抽检 | 任一关键域下降超过约定阈值 |
| POC 整体合并破坏架构 | 高/高 | 只迁移算法、逐模块测试 | PR 删除 `zpds/` 或测试体系 |
| 大文件导致内存/磁盘失控 | 中/高 | 流式读取、配额、临时目录监控 | 单 Worker 超出资源上限 |

## 18. 范围控制

### Production Candidate 必须完成

- 五源 Raw Inventory 与全量读取/解码；
- Stream/Clock/Calibration Catalog；
- 时间、单位、坐标与 Sample Map；
- Stage 0–8、11、12；
- Physical Trim/Split；
- Prepared Revision 与回读验证；
- 最小 CEU/Experience Manifest；
- Release、LeRobot、RLDS；
- Gold、Review、审计、CI、恢复和安全隔离。

### 条件完成

- Stage 9 手部稀疏检测与跟踪；
- Stage 10 Scene/Action/VLM Proposal；
- WiLoR/HaWoR POC；
- 标定重投影高级评估；
- 去重 Embedding。

这些功能必须通过成本和质量评估后才进入默认生产配置。

### 暂不纳入

- 全量 3D/4D 场景重建；
- 生产级世界坐标人体重建；
- 人形机器人动作重定向完整平台；
- 自研 UMI 采集硬件；
- 大规模通用标注平台；
- 自研基础模型训练。

这些工作可作为独立产品线消费 ZPDS Release，不应阻塞数据底座上线。

## 19. Production Go-Live 清单

- 五源至少各完成一批真实数据；
- 完成三轮可重复批处理；
- 所有 Release 可回读；
- Gold Regression 全部通过；
- 自动 Reject 精确率 100%；
- Quarantine 有明确负责人和 SLA；
- Raw 未被任何流水线修改；
- Sample Map 和 Provenance 完整；
- 时间、单位、Frame、Pose 约定通过审计；
- train/validation/test 无 Session/Group 泄漏；
- 安全隔离和许可 Gate 通过；
- 监控、告警、Runbook、备份和回滚演练通过；
- 数据工程、训练团队和质量审阅三方签字。

## 20. 需要尽快确认的管理决策

以下问题应在 Gate 0 前由项目负责人确认：

1. 对外标准名称到底使用 ZPDS 还是 ZRDS，持久化字段如何兼容；
2. Prepared Revision 的正式字段名；
3. 五源真实数据的访问方式、责任人和许可状态；
4. A2D 时间映射可接受的最大误差；
5. UMI 磁编码器和 VIO 的物理语义；
6. Production Candidate 是否强制包含 Stage 9/10；
7. Review UI 是仅内部使用还是需要多租户/RBAC；
8. 目标部署环境、参考硬件和对象存储；
9. Quarantine 审阅 SLA；
10. LeRobot/RLDS 首个真实训练消费者及其验收数据。

在这些决策未完成前，可以开发通用 Schema、运行框架和 Guida Reference Slice，但不能冻结正式 Release v1。
