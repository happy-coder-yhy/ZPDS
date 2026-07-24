# ZPDS

ZPDS（Ziki Physical AI Data Specification）是面向具身智能多源数据的清洗、
标准化、质量治理和训练视图中间层。

## WP0：标准冻结与治理

当前机器契约包括：

- `docs/adr/`：数据对象、边界、单位、决策、来源和版本命名；
- `zpds/schemas/`：Draft 2020-12 JSON Schema；
- `configs/reason_codes/`：版本化 Reason Code Registry；
- `configs/quality_views/`：面向不同任务的 Quality View；
- `configs/gold/`：五源 Gold Manifest；
- `examples/schemas/`：通过 Schema 的最小对象。

在仓库根目录执行：

```bash
python -m scripts.validate_wp0
python -m scripts.manage_gold status
python -m pytest
```

自动校验通过代表机器契约完整；Gold Manifest 经指定负责人复核并冻结后才代表 Gate 0
正式通过。详见 `docs/governance/WP0验收说明.md`。

## WP1：平台基础与运行框架

WP1 工程实现与自动验收已经完成：建立严格 Pipeline Config Schema、不可变 Config Loader、版本化 Schema
Registry、统一 Stage 契约，以及本地 Storage、原子 Run Ledger 和支持幂等、重试、
断点恢复的顺序 Runner。CLI、结构化 JSONL 日志、运行指标和可选依赖隔离也已完成。

- 前三步：`docs/governance/WP1前三步实现说明.md`；
- 第 4～6 步：`docs/governance/WP1第4至6步实现说明.md`。
- 完成与验收：`docs/governance/WP1完成与验收说明.md`。

对象存储仍属于后续部署能力；真实 Guida Stage 0～2 已在 WP2 实现。

## WP2：Adapter 与 Profile

WP2 已完成五源只读 Adapter、Profile、Source Inventory、Stream/Clock/Calibration
Catalog、完整读取/解码入口，以及 Guida Stage 0～2 纵向闭环。常用命令：

```bash
zpds source inspect --profile guida_ego --raw-root <raw-root> --input-ref raw://<session>
zpds source validate --profile guida_ego --raw-root <raw-root> --input-ref raw://<session>
zpds source scan --profile guida_ego --raw-root <raw-root> --input-ref raw://<session>
```

详见 `docs/governance/WP2完成与验收说明.md`。WP2 不生成 Prepared Segment；补充验收已经
覆盖 HDF5 全量分块读取、MCAP 内嵌 PNG/H264 负载解码和 EPIC primitive-only Pickle
隔离内容解析。

## WP3：Guida 基础清洗闭环

WP3 当前完成了 Guida 的第一个可执行纵向切片：以 `index.jsonl` 为权威视频时间轴，
保留重复但不回退的 IMU 时间戳，执行黑/纯色、冻结帧、深度有效率、IMU gap 等硬检查，
形成物理有效区间和显式决策，生成 video/IMU Source Sample Map、规范化 IMU、Calibration
记录、质量证据、`segment.json` 与 `revision.json`。写入采用临时目录校验后原子落盘，
并可回读检查 Schema、内部引用、时间单调性与 Raw SHA-256。

```bash
python -m zpds clean guida \
  --config configs/pipeline/default.yaml \
  --raw-root <raw-root> \
  --input-ref raw://<session> \
  --output-root <dataset-root> \
  --code-version <至少7位的代码版本>

python -m zpds prepared validate \
  --segment-dir <dataset-root>/prepared_segments/r0001/<segment-id> \
  --raw-root <raw-root>
```

当前 RGB/Depth 采用 manifest-first source selection：不会修改 Raw，也不冒充已经重编码了
媒体文件。WP3 还没有扩展到其余四源，也未包含模糊/曝光模型、语义/VLM、人工 Review、
Release 或训练格式导出。详见
`docs/governance/WP2补充验证与WP3基础清洗闭环.md`。
