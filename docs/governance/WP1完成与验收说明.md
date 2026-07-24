# WP1 完成与验收说明

WP1 的目标是建立可安装、可配置、可恢复、可观测的单机 Pipeline 运行框架，不实现
具体数据源的 Adapter 或 QC 算法。

## 已完成范围

### 1. 可安装包与 CLI

`pyproject.toml` 注册了：

```text
zpds = "zpds.cli:main"
```

安装开发环境后可执行：

```powershell
zpds --version
zpds config validate configs/pipeline/default.yaml
zpds run execute --help
zpds run status --help
```

也可以不依赖脚本入口：

```powershell
python -m zpds --version
```

### 2. Config 与 Schema

- Pipeline Config 使用版本化 Schema 严格校验；
- 配置加载后不可变；
- 未知字段、缺失字段和未知版本明确失败；
- 对配置语义计算确定性 SHA256；
- Prepared 保留源频率，Alignment 才定义派生时间网格；
- Runner 的重试次数和退避时间进入配置哈希。

### 3. Stage、Storage、Ledger 与 Runner

- Stage 0～12 使用统一 `StageDescriptor`、`StageContext`、`StageResult`；
- `raw://` 只读，`artifact://` 支持安全、原子写入；
- Ledger 记录 pending、running 和终态，并保留 attempt、错误、Decision、Evidence；
- execution key 绑定 Stage、session、输入、配置哈希和代码版本；
- 自动重试；
- 中断后继续；
- 已成功 Stage 幂等复用；
- 相同 run_id 的身份冲突拒绝覆盖。

### 4. 结构化日志与指标

运行时事件以 JSON Lines 记录：

```text
<artifact_root>/runs/<run_id>/events.jsonl
```

事件覆盖：

- `run_started`
- `stage_reused`
- `stage_attempt_started`
- `stage_attempt_finished`
- `stage_retry_scheduled`
- `stage_interrupted`
- `run_finished`
- `run_interrupted`

运行指标原子写入：

```text
<artifact_root>/runs/<run_id>/metrics.json
```

指标包含：

- Run/Stage 状态；
- attempts 与 retries；
- Stage 总耗时；
- Decision Type 分布；
- Reason Code 分布；
- Severity 分布；
- config hash 与 code version。

`metrics.json` 使用正式的 `run_metrics.schema.json` 校验。

### 5. 可选依赖隔离

基础包导入测试确认：

```python
import zpds
```

不会加载 Torch、HDF5 或 MCAP。具体 Adapter 只有在实际使用时才应导入对应可选依赖。

## CLI 运行方式

Stage 代码必须由操作者明确指定为可信插件，不从 YAML 执行：

```powershell
zpds run execute `
  --config configs/pipeline/pilot.yaml `
  --raw-root D:/datasets `
  --artifact-root D:/zpds-output `
  --run-id guida-demo-001 `
  --session-id session-001 `
  --input-ref raw://egos/session-001/index.jsonl `
  --code-version <git-commit-or-build-id> `
  --stage your_package.stages:create_stage0 `
  --stage your_package.stages:create_stage1
```

查询状态：

```powershell
zpds run status `
  --artifact-root D:/zpds-output `
  --run-id guida-demo-001
```

## 验收结果

当前自动验收：

```text
完整单元/Contract Test：76 passed
mypy：102 source files，0 issues
WP1 核心模块覆盖率：91%
WP0 validation：passed
全仓 Ruff：passed
可安装 CLI：zpds 0.1.0
基础包可选依赖隔离：passed
```

WP1 验收项对应关系：

| 验收项 | 证据 |
|---|---|
| 中断后只执行未完成 Stage | Runner interruption/resume test |
| 相同身份复用成功结果 | execution key + repeated-run test |
| 配置、Ledger、指标机器可校验 | Config、Run Ledger、Run Metrics Schema |
| 基础包不要求重型可选依赖 | isolated import subprocess test |
| 可安装包和 CLI | `zpds.exe --version` 与 CLI Contract Test |
| 日志和指标 | JSONL event test + metrics Schema test |

## WP1 之外的边界

以下内容不属于 WP1 完成声明：

- Guida、遁甲、UMI、A2D、EPIC 的真实 Adapter；
- Stage 0～2 的真实 Inventory、容器和时钟检查；
- Prepared Writer 的真实数据转换；
- S3/OSS 对象存储；
- 跨进程锁和分布式调度；
- Release 和 Export 的生产链路。

下一阶段进入 WP2，优先实现 Guida 的只读 Adapter、Profile、Stream/Clock Catalog 和小型
Fixture，再把真实 Stage 0～2 接入当前 Runner。
