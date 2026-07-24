# WP2 Adapter 与 Profile 完成及验收说明

## 1. 本阶段解决了什么

WP2 把五种真实数据源接入同一套只读入口。它不会修改 Raw，也不会在这个阶段重编码、
重采样或生成 Prepared Segment。它完成的是“先看懂原始数据并留下可追溯清单”：

```text
Raw Session
  -> Profile 识别来源约束
  -> Adapter inspect / validate / scan
  -> Source Inventory + Hash
  -> Stream / Clock / Calibration Catalog
  -> Stage 0 文件登记
  -> Stage 1 结构校验
  -> Stage 2 时间校验
```

三个动作的区别：

- `inspect`：只读元数据，列出文件、流、时钟和标定，适合日常快速登记；
- `validate`：快速检查必需文件、magic、索引、Topic、Schema 和完整性；
- `scan`：逐条读取或解码，适合导入前验收，耗时与数据规模成正比。

## 2. 已交付能力

### 2.1 统一契约

- `SourceAsset`：文件身份、相对路径、大小、媒体类型和 SHA-256；
- `SourceStream`：流类型、角色、Topic、编码、频率、frame 和所属时钟；
- `ClockDescriptor`：时钟域、来源、是否权威及说明；
- `CalibrationDescriptor`：来源记录的内参、外参或 calibration message 引用；
- `ValidationReport`：检查量、解码量、问题级别、问题码和证据上下文；
- `source_inventory.schema.json` 与 `validation_report.schema.json`：产物落盘前机器校验。

### 2.2 容器 Reader

- Video：OpenCV 元数据探测、随机读帧和逐帧流式解码；
- MCAP Protobuf：Summary、Topic Catalog、原始消息读取和 Protobuf 解码；
- ROS2 CDR：MCAP 内 CDR 解码；DB3 提供只读消息读取，并明确不冒充已解码；
- HDF5：dataset Catalog、分块读取和全部 dataset 数据遍历；
- Log：逐行解析，汇总最多保留 100 条样例，避免大日志全部驻留内存；
- Pickle：主进程只做 opcode 静态检查；primitive-only 内容使用隔离子进程读取摘要，
  禁用网络、限制可加载类型并设置超时；包含对象构造 opcode 时禁止加载。

### 2.3 五源 Profile

- Guida：`index.jsonl` 是权威时间轴；发现 `imu/imu_*.csv`；保留视频容器时间；
- 遁甲：校验 IMU、Depth 和 Camera Topic；保留 MCAP log/publish time；
- UMI：校验 robot0/robot1 的 Camera、IMU、磁编码器和 VIO Topic；磁编码器仅记 raw scalar；
- A2D：建立 head/left/right × color/depth 完整性矩阵；分别登记 camera frame、
  HDF5 row 和 ROS 时钟；不假设第 N 行等于第 N 帧；
- EPIC：兼容 `epic100` 和旧名 `epic100_auto_annotation`；Pickle 作为不可信输入；
  原视频未绑定前不伪造时间映射。

## 3. Stage 0～2 纵向闭环

命令：

```powershell
zpds run source `
  --profile guida_ego `
  --config configs/pipeline/default.yaml `
  --raw-root D:\datasets\egos `
  --artifact-root D:\zpds-artifacts `
  --run-id guida_demo_001 `
  --session-id guida_demo `
  --input-ref raw://墨现 `
  --code-version local-wp2
```

产物：

```text
<artifact-root>/runs/<run-id>/
├── ledger.json
├── events.jsonl
├── metrics.json
├── stage-0/inventory.json
├── stage-1/structure.json
└── stage-2/time.json
```

Stage 0 对每个登记资产做流式 SHA-256；Stage 1 把警告、错误和致命问题映射为显式
Decision；Stage 2 检查权威时间轴或分别检查多个时钟。重复运行同一 run 时会复用已完成
Stage。

## 4. 真实数据只读验收

2026-07-24 在本机五源样例上完成如下检查：

| Profile | Assets | Streams | Clocks | Calibrations | 全量结果 |
|---|---:|---:|---:|---:|---|
| Guida | 11 | 3 | 3 | 4 | 3,932 条视频/IMU记录完成解码；声明 IMU 路径与实存路径不一致，保留 WARN |
| 遁甲 | 1 | 9 | 2 | 4 | 4,280 / 4,280 条 Protobuf 消息解码成功 |
| UMI | 1 | 14 | 2 | 2 | 34,860 / 34,860 条 Protobuf 消息解码成功 |
| A2D | 1,024 | 55 | 4 | 3 | 50,766 / 50,766 条图像、HDF5、ROS2 CDR 和日志记录读取成功；不完整相机 tuple 保留 WARN |
| EPIC | 1,407 | 1 | 1 | 0 | 快速检查确定性抽样 20 个；全量 `scan` 静态检查全部 Pickle，不在主进程反序列化 |

这里的 WARN 不等于程序失败：

- Guida WARN 说明 `meta.json` 写的是 `imu/imu.csv`，实际发现
  `imu/imu_000000.csv`，程序使用发现结果但保留证据；
- A2D WARN 说明部分帧目录缺少六元相机 tuple 中的成员，数据仍可登记，但后续不能当作
  完整多相机样本。

## 5. 验证命令

```powershell
# 快速登记
zpds source inspect --profile guida_ego --raw-root D:\datasets\egos --input-ref raw://墨现

# 快速结构校验
zpds source validate --profile guida_ego --raw-root D:\datasets\egos --input-ref raw://墨现

# 全量读取/解码（大数据可能需要较长时间）
zpds source scan --profile epic100 --raw-root D:\datasets --input-ref raw://epic-kitchens-100

# 自动测试和静态检查
python -m pytest -q
python -m ruff check .
python -m mypy zpds
```

`validate` 通过只表示结构检查没有 ERROR/FATAL；真正准备导入生产前还应运行 `scan`。

## 6. WP2 边界与下一阶段

WP2 已能回答“有哪些文件、哪些流、用什么时钟、标定来自哪里、是否读得出来”。它还没有
解决“不同流的某个样本怎样精确对齐”和“怎样生成可回读的 Prepared Segment”。

下一阶段 WP3 应依次实现：

1. Clock Model 和跨时钟关系；
2. Source Sample Map；
3. 单位与坐标规范化；
4. Calibration 规范化及误差表达；
5. Prepared Segment 原子写入、回读校验和 Revision。

Guida Prepared 最小闭环现已实现，补充的真实负载验证、WP3 产物和当前边界见
`WP2补充验证与WP3基础清洗闭环.md`。Gate 1 是否正式通过仍取决于项目对 Gate 1 的人工
验收定义，不能只由测试数量自动推断。
