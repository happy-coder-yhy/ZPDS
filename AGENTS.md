# AGENTS.md

这个文件为在 ZPDS 仓库中工作的自动化代码助手提供统一约定。进入本仓库后，先阅读本文件，再结合目标模块、配置和测试上下文行动。

本文件适用于整个仓库；如果未来某个子目录包含更具体的 `AGENTS.md`，则该子目录内优先遵守更具体的约定。

## 项目定位

- 项目名：`ZPDS`（Ziki Physical AI Data Specification）。
- 目标：作为原始采集数据与训练格式之间的“规范化经验中间层”，清洗、质检、切分并标准化多来源具身智能数据，最终形成可追溯、可发布、可导出的训练数据。
- ZPDS 不替代 MP4、MCAP、Parquet、LeRobot 或 RLDS；它定义经验的来源、处理过程、标注、置信度、资产引用和训练视图。+
- 语言与版本：Python 3.10+。
- 构建系统：setuptools，配置入口为 `pyproject.toml`。
- 主包：`zpds/`。
- 命令行脚本：`scripts/`。
- 流水线配置：`configs/pipeline/`。
- Profile QC 阈值：`configs/qc_thresholds/`。
- 基础依赖：OpenCV、NumPy、Pandas、PyYAML。
- 可选能力：MCAP、HDF5、Torch/手部模型，分别由对应 optional dependency 提供。
- 代码质量工具：pytest、Ruff、mypy。
- 当前阶段：`0.1.0`、Pre-Alpha。仓库已经建立领域边界和接口骨架，但大量模块仍未实现，不能把目录或类名视为已经可用的能力。

## 目录速览

- `zpds/core/`：跨模块共享的领域模型，包括数据清单、质量决策、质量报告和 provenance。
- `zpds/adapters/`：原始数据容器的探测、校验和读取，覆盖 MCAP、HDF5、ROSBag、视频和日志。
- `zpds/profiles/`：不同采集源的 Profile 定义及注册表。
- `zpds/scene/`：场景切点、ego 运动抑制、语义边界和 VLM 复核。
- `zpds/hands/`：手部检测、姿态估计、跟踪、指标和操作有效性。
- `zpds/segmentation/`：有效区间提议、首尾裁剪和坏区间切分。
- `zpds/prepared/`：标准化 Prepared Segment 的约定、写入、校验和 revision 管理。
- `zpds/qc/`：Stage 0～12 质量检查及级联调度。
- `zpds/annotation/`：CEU、Experience Manifest 和大型标注资产。
- `zpds/release/`：Release 元数据及 train/val/test split。
- `zpds/export/`：LeRobotDataset v3、RLDS/Open X-Embodiment 等训练格式导出。
- `zpds/utils/`：哈希、时间、单位、Schema 校验和可视化等无业务状态工具。
- `configs/`：流水线行为与数据源阈值配置。
- `scripts/`：端到端入口和 Pilot 分阶段脚本；目前多数仍是占位实现。
- `tests/`：测试目录，应按照 `zpds/` 的模块边界组织。
- `docs/`、`notebooks/`：设计文档和探索性分析，不应承载生产流水线逻辑。

## 目标数据流

仓库的目标依赖方向如下：

```text
不可变 Raw Session + Source Profile
    -> inventory / hash / license / privacy
    -> Adapter inspect / validate / read
    -> typed stream catalog + clock catalog
    -> 低成本硬检查 -> 模态 QC -> 高成本模型/人工复核
    -> physical segmentation
    -> Prepared Revision（pass / quarantine / reject manifest）
    -> Versioned Annotation / CEU / Assets
    -> Release + Dataset Split
    -> LeRobot / RLDS / ARIO / World Model 等训练视图
```

- `core` 是底层领域协议，不得反向依赖 adapters、qc、prepared、release 或 export。
- 上层模块通过 `core` 中的明确类型交互，不要依赖其他模块的内部字典结构。
- Annotation、Release 和 Export 应面向标准化产物，不应直接耦合某个原始容器 Reader。
- Raw 数据是不可变事实来源。清洗只能生成 manifest、派生产物和决策记录，不得覆盖、移动或删除 Raw。
- 采用分级成本清洗：先做来源、许可、文件、结构、解码和时间硬检查，再对候选片段运行光流、embedding、手部模型或 VLM。
- 质量是多维向量，不压缩成一个不透明总分；不同下游任务可以选择不同 quality profile。
- 当前编排顺序尚未完整实现。若任务只涉及局部模块，不要顺手虚构或重排整条流水线；需要改变阶段顺序时，应同时更新入口、配置、测试和文档。

## 工作执行流程

- 开始工作前读取 `pyproject.toml`、目标模块、相邻抽象基类、对应配置和现有测试。
- 先确认任务属于 Adapter、Profile、QC、Prepared、Annotation、Release 或 Export 中的哪一层。
- 检查工作树并保留用户已有改动，不覆盖或回滚无关文件。
- 优先完成一个可验证的纵向闭环，不要为了“补齐架构”一次性实现所有 `NotImplementedError`。
- 修改公共数据结构或方法签名时，检索全部调用方并同步更新测试、配置和脚本。
- 实现完成后执行与改动风险相匹配的测试和静态检查。
- 最终回复说明：改了什么、验证了什么、哪些能力仍未实现或未验证。

## 核心数据契约

- 跨模块公共类型优先定义在 `zpds/core/`，避免在多个模块重复定义相似字典。
- `SessionInventory` 描述一次采集会话及其数据流；新增字段时必须提供清晰默认值并保持旧数据兼容。
- `SourceStream.kind` 使用 `StreamKind`，不要散落自定义字符串。
- 时间戳统一使用整数纳秒；跨时钟域的数据必须明确 `ClockDomain` 和对齐方式。
- 角度使用弧度，坐标系使用右手系（x-forward、y-left、z-up）。
- 长度规范当前存在待解决冲突：`prepared/conventions.py` 使用毫米，调研计划中的通用规范写为米。修改相关代码前必须核对目标 Schema；在冲突解决前保留原值、显式记录单位和转换来源，不得静默假设。
- 统一约定以 `zpds/prepared/conventions.py` 为准；配置与常量冲突时不得静默选择其中一个。
- QC 结果使用 `Decision`、`Severity`、`ReasonCode`、`QualityMetric` 和 `QualityReport`。
- 比较严重等级和原因码时使用枚举成员，不要依赖裸字符串。
- 清洗决策需要区分 `keep`、`keep_with_flag`、`quarantine`、`trim`、`split` 和 `reject`。现有核心模型尚未完整表达这套 taxonomy；实现相关功能时应先建立显式类型，不要把决策含义塞进 message 字符串。
- 新增持久化格式时要包含 provenance、生产版本和配置哈希，确保结果可复现。
- ID、哈希、数据切分和输出顺序必须确定性生成；涉及随机行为时显式接收并记录 seed。

## Adapter 与 Profile 约定

- Adapter 的职责是理解外部容器，不承载 QC 策略、数据集发布策略或训练格式逻辑。
- `inspect()` 应尽量只读取元数据，返回 `SessionInventory`，避免解码整段视频或把大型数据全部载入内存。
- `validate()` 用于快速检查 header、magic、索引和必要 schema，返回值和异常边界必须清楚。
- Reader/Decoder 优先提供迭代式或分块接口，处理 MCAP、视频、HDF5 等大文件时禁止无理由整文件读入内存。
- 文件、视频句柄和临时资源必须可靠关闭；优先使用上下文管理器。
- 可选依赖只在对应模块实际使用时导入。缺少可选依赖时给出包含安装 extra 的可操作错误，不得导致 `import zpds` 整体失败。
- 外部工具（例如 ffmpeg/ffprobe）必须检查退出码，并在错误中包含命令目标和 stderr 摘要。
- Profile 用于描述采集源差异。与特定数据源相关的 topic、字段映射、流要求和 QC 阈值不要硬编码进通用 Adapter。
- 新增 Profile 时同步更新注册表、阈值配置和至少一个注册/配置测试。
- 计划中的目标 Adapter 契约包含：
  - `inspect(session_uri) -> SessionInventory`
  - `read_stream_catalog() -> list[SourceStream]`
  - `read_clock_catalog() -> list[ClockDomain]`
  - `propose_physical_valid_spans() -> list[SpanProposal]`
  - `prepare_span(span, profile) -> PreparedArtifacts`
  - `validate_segment(segment) -> ValidationReport`
- 当前 `BaseAdapter` 只声明了 `inspect()` 和 `validate()`。任务涉及 Adapter 公共接口时，应兼顾当前调用方并逐步收敛到目标契约，不要在各 Adapter 中发明互不兼容的方法。

## QC 约定

- Stage 编号为 `0` 到 `12`，实际共 13 个阶段。新增说明时使用“Stage 0～12”或“13 个阶段”，避免继续使用含义不清的“12 级”。
- 阶段职责：
  - Stage 0：文件登记、缺失文件和哈希。
  - Stage 1：容器、索引和结构 Schema。
  - Stage 2：时间戳、间隔、回退和时钟对齐。
  - Stage 3：黑帧、过曝、模糊和冻结画面。
  - Stage 4：视频帧时序、丢帧、重复帧和 VFR。
  - Stage 5：深度有效性和单位。
  - Stage 6：IMU 间隔、漂移和饱和。
  - Stage 7：关节、指令超时和夹爪。
  - Stage 8：内外参和重投影误差。
  - Stage 9：手部存在与跟踪。
  - Stage 10：语义一致性。
  - Stage 11：近重复数据。
  - Stage 12：最终交付完整性。
- 每个 Stage 的检查结果应返回 `list[Decision]`，不要返回含义不明的普通字典或布尔值。
- 持久化的检查结果至少应能表达：metric name、value/unit、applicability、severity、decision、reason code、span、evidence URI、producer/version/config hash。
- `FATAL` 表示数据不可继续使用，`ERROR` 表示必须修复，`WARN` 表示建议处理，`INFO` 仅记录。
- Severity 表示问题严重度，不等同于最终数据决策；例如技术完整的失败/恢复轨迹可以被保留并带 flag。
- 阈值从 `configs/qc_thresholds/` 读取；通用默认值放流水线配置，Profile 特有值放对应 Profile 文件。
- 配置缺失、字段拼写错误或单位不一致时应明确失败，不要静默使用魔法默认值。
- `QCCascade` 负责调度、停止条件和聚合报告，不应重复实现各 Stage 的检测算法。
- 质量报告的 `overall_pass` 必须由 decisions/metrics 一致计算，不能只依赖初始化默认值。
- 自动 `reject` 只用于高精度、可解释的硬故障。未经金标集校准的模糊、曝光、运动或语义阈值，优先产生 `quarantine`，避免误删长尾数据。
- 报告除保留率和原因分布外，还应比较清洗前后的来源、设备、场景、任务和动作分布，避免质量门引入隐性偏差。

## Prepared、Annotation、Release 与 Export 约定

- Prepared Segment 是原始采集格式和下游训练格式之间的稳定边界。
- Prepared Segment 只表示解码、时钟和关键流的物理连续性。Scene、atomic action、task 和 CEU 边界属于版本化 Annotation，不能为了语义方便强切 Prepared。
- Prepared 层保留各流原始频率。30 Hz perception、50 Hz control 等统一时间网格放入 `alignments/`，并记录最近邻、线性、SLERP 或 ZOH 等方法、映射和误差；禁止跨长 gap 或 clock reset 插值。
- Prepared 写入应采用“临时目录写入、完整校验、原子落盘”的方式，避免留下看似完成的半成品。
- `segment.json`、revision、release 和 manifest 等元数据需要正式 Schema；修改格式时考虑版本迁移和向后兼容。
- 大型 mask、pose、track、pointcloud 等资产放入 `AssetStore`，manifest 中保存引用，不要嵌入巨型 JSON。
- train/val/test split 必须确定、互斥且可复现；同一来源会话存在数据泄漏风险时应按 session 或 group 切分。
- Exporter 只负责格式转换，不应在导出阶段悄悄修复上游 QC 问题。
- 导出前验证 Release 和所有引用资产；失败时不要生成“成功”标记或不完整索引。
- 标准产出结构遵循：

```text
<dataset_id>/
├── dataset.json
├── prepared_segments/<prep_revision>/
│   ├── revision.json
│   └── <segment_id>/
│       ├── segment.json
│       ├── data/
│       ├── calibration/
│       └── alignments/
├── experiences/<experience_version>/
│   ├── experience_manifest.json
│   ├── ceus.parquet
│   └── assets/
├── releases/<release_id>.json
└── exports/{lerobot,rlds}/<release_id>/
```

## 五种来源的硬约束

- 墨现 Guida ego：
  - `index.jsonl` 的 `timestamp_ns` 是权威时间轴，预览 MP4 仅用于 QA。
  - RGB/Depth 必须验证逐帧配对；转 CFR 时保存 source frame 到 output frame 的 sample map。
  - IMU 文件可能采用 `imu/imu_*.csv`，不能只相信 meta 中的固定路径。
- 遁甲 ego MCAP：
  - 同时保留 MCAP log time 和消息内 timestamp，不得静默合并。
  - H264 重建必须保持消息时间、关键帧/GOP 和 topic 到输出流的映射。
  - 深度 PNG 的 dtype、invalid 值和物理单位必须实测，不能按扩展名推断。
- 简智新创 UMI：
  - `/robot0` 与 `/robot1` 分组必须始终明确，避免左右端混淆。
  - 保留 `arnold.common.Header` 时间与 MCAP log time。
  - 磁编码器含义确认前只保存 raw scalar，不直接命名为 gripper action。
  - 禁止跨 VIO reset 或长 gap 插值。
- A2D 真机：
  - 建立 head/left/right × color/depth 的 completeness matrix。
  - 相机 frame index、HDF5 行和 ROS 消息频率不同，严禁“第 N 行对应第 N 帧”。
  - 无逐帧真值时只能生成带误差说明的推断映射，不能伪造精确同步。
  - 技术损坏、操作失败、失败恢复和任务未完成必须分别标注。
- EPIC-KITCHENS-100 衍生包：
  - Pickle 是不可信输入，必须在无网络、最小权限、有限内存/时间的隔离进程解析，禁止在主流水线直接反序列化未知 pickle。
  - 自动 mask/hand-object 标注标记为 `model_estimated`，不得冒充人工真值。
  - 原视频缺失或 hash/version 不匹配时标记为 orphaned，不生成伪时间映射。
  - 代码当前注册名为 `epic100`，计划文档使用过 `epic100_auto_annotation`；完成显式迁移前保持现有注册名兼容。

## 配置与版本管理

- YAML 只保存配置数据，不在配置中嵌入可执行代码。
- 配置读取后应校验必填字段、类型、范围和单位。
- 同一阈值只保留一个权威来源，不要在 Python、默认配置和 Profile 配置中重复硬编码。
- 流水线版本、包版本和产物 provenance 需要明确关联；不要在多个文件中无说明地独立升级版本号。
- 对结果有影响的配置必须进入 `config_hash` 或等价的 provenance 记录。
- 不提交真实密钥、访问令牌、内部数据地址、个人绝对路径或敏感设备信息。
- 许可、隐私和再分发条件属于正式质量门，而不是文档备注；技术可读的数据不一定允许进入 Release。

## 代码风格

- 遵循现有包结构和命名方式，行宽上限为 100。
- 新增公共函数、类方法和复杂内部函数应提供类型标注。
- 简单领域数据优先使用 dataclass；集合默认值使用 `default_factory`，禁止共享可变类属性或默认参数。
- 路径处理优先使用 `pathlib.Path`；如果既有公共接口使用 `str`，保持兼容并在内部规范化。
- 函数保持单一职责。解析、校验、转换、写入和报告生成不要混在一个超长函数中。
- 不捕获宽泛异常后静默继续；补充输入路径、topic、stage 等上下文后重新抛出或转换为明确领域错误。
- 不使用裸 `print` 作为库代码日志。命令行入口可以输出用户信息，库模块应使用 logging 或返回结构化结果。
- 未实现能力可以保留明确的 `NotImplementedError`，但已宣称完成的路径不得留下占位返回值、假数据或吞错逻辑。
- 未说明原因时不要新增依赖；能使用标准库和现有依赖完成时优先复用。
- `utils` 只放无业务状态、跨模块复用的工具；仅被单个模块使用的逻辑留在该模块内部。

## 实施优先级

- Phase 0：冻结 Prepared/Scene/Action/CEU 边界、决策 taxonomy、质量视图和各源金标/反例集。
- Phase 1：实现只读 inventory、hash、license/privacy、结构探测和 stream/clock catalog，不做重编码。
- Phase 2：实现 clock model、sample map、单位/坐标规范化、calibration 和基础 Prepared。
- Phase 3：实现低成本 QC、physical trim/split、pass/quarantine/reject 及证据。
- Phase 4：实现稀疏手检测、跟踪、自适应加密、scene/action proposal 和 VLM 复核。
- Phase 5：实现机器人 QA、端到端 validator、review、统计、Release 与试验导出。
- 除非用户明确要求改变优先级，优先完成当前 Phase 的可验收纵向切片，不要提前把高成本模型接入未稳定的数据契约。

## 测试策略

- 测试目录尽量镜像源码结构，例如 Adapter 测试放在 `tests/test_adapters/`。
- 每次实现新行为至少覆盖：
  - 正常输入。
  - 缺失、损坏或空输入。
  - 边界时间戳、单位和阈值。
  - 预期异常及错误信息。
- 使用 `tmp_path` 创建输出，测试结束后不在仓库中留下缓存、视频、模型或数据集产物。
- Fixture 应小、确定、可提交；不要把真实大型 MCAP、ROSBag、视频或模型权重加入 Git。
- 可选依赖测试使用 `pytest.importorskip` 或清晰的条件跳过；不要让基础测试强制安装 Torch 等重型依赖。
- 外部程序和模型推理在单元测试中使用可验证的替身；核心解析逻辑尽量使用小型真实 fixture。
- 修复缺陷时先添加能复现问题的回归测试。
- 只修改文档时可以不运行完整测试，但最终回复必须说明未运行的原因。
- 如果完整测试受缺失依赖、模型、外部工具或样例数据阻塞，运行能够执行的子集，并明确报告阻塞项。

## 常用命令

```bash
# 安装基础包
python -m pip install -e .

# 安装开发依赖
python -m pip install -e ".[dev]"

# 按需安装可选能力
python -m pip install -e ".[mcap]"
python -m pip install -e ".[hdf5]"
python -m pip install -e ".[hands]"

# 运行测试
python -m pytest -q

# 运行指定测试目录
python -m pytest tests/test_adapters -q
python -m pytest tests/test_qc -q

# 静态检查
python -m ruff check .
python -m mypy zpds

# 基础导入与 Profile 注册冒烟检查
python -c "import zpds; from zpds.profiles.registry import list_all; print(zpds.__version__, list_all())"
```

`scripts/run_pipeline.py`、`scripts/run_qc_report.py`、`scripts/run_export.py` 目前仍可能因未实现功能而失败。只有当任务确实实现了对应纵向链路后，才把这些脚本的成功运行作为验收结论。

## 完成标准

- 变更位于正确的架构层，没有制造反向依赖或跨层耦合。
- 新增或修改的行为有对应测试，且相关测试通过。
- 配置、类型、单位、Schema 和错误信息保持一致。
- 大文件路径采用流式或分块处理，没有无理由的全量内存加载。
- 可选依赖缺失不会破坏基础包导入。
- Raw 未被修改；任意 Prepared sample 能追溯到 source asset/sample、处理版本和配置。
- physical boundary 与 scene/action/CEU boundary 没有混用，重采样只存在于显式 alignment 或 export 视图。
- 没有提交生成物、缓存、模型权重、大型数据或敏感信息。
- 没有夹带与任务无关的重构、格式化或版本升级。
- 最终回复使用中文，列出改动、验证命令及结果；未完成或未验证的部分必须明确说明。
