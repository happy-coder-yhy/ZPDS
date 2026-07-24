# WP2 补充验证与 WP3 基础清洗闭环

## 1. 交付结论

本轮完成两件事：

1. 把 WP2 从“容器能打开”补强到“关键数据负载确实读过”；
2. 以 Guida 为首个参考源，打通从 Raw 到可回读 Prepared Revision 的基础清洗闭环。

Raw 始终只读。程序只在输出目录生成 manifest、映射、规范化 IMU、质量记录和版本产物。

```text
Guida Raw
  -> 结构与索引校验
  -> 全量 RGB / Depth 解码
  -> index.jsonl 权威时间恢复
  -> IMU 单调性与 gap 检查
  -> 基础硬质量检查
  -> 物理有效区间 trim / split
  -> Source Sample Map
  -> Prepared 临时写入
  -> Schema / 引用 / 时间 / Raw hash 回读
  -> 原子发布 Prepared Revision
```

## 2. WP2 补充验证

### 2.1 HDF5

`Hdf5Inspector.scan()` 不再只读取每个 dataset 的首尾样本，而是沿第一维分块遍历全部数据。
报告会记录：

- `dataset_count`；
- `rows_or_scalars_read`；
- `elements_read`；
- `full_dataset_scan: true`。

这样可以发现中间 chunk 的读取错误，同时避免无理由把整个 dataset 一次性装入内存。

### 2.2 MCAP 媒体负载

MCAP 的 Protobuf 外壳成功不代表内部 PNG/H264 一定可解码。补充扫描会：

- 用 `cv2.imdecode` 解码 PNG/JPEG；
- 按 topic 流式重建 H264/H265 临时码流；
- 用 OpenCV 解码重建码流；
- 分别记录 packet 数、图片数、视频帧数和解码问题；
- 临时码流在扫描结束后删除。

MCAP log time 与 publish time 仍分别保留，没有静默合并。

### 2.3 EPIC Pickle

Pickle 仍是不可信输入：

- 主进程只做 opcode 静态检查；
- 包含对象构造/global opcode 的文件禁止内容加载，记为 `untrusted_pickle`；
- primitive-only 文件才会交给 `-I -S` 隔离子进程；
- 子进程禁用网络、限制可加载类型并设置超时；
- 主进程只接收有界摘要，不接收原对象。

当前真实数据只完成了代表样例的内容级解析；EPIC 全库隔离解析是独立耗时作业，不能把
代表样例结果表述为“全库内容已解析”。

### 2.4 真实数据补充结果

| 来源 | 补充校验结果 |
|---|---|
| 遁甲 | 1,377 个媒体负载全部解码：312 张 PNG、1,065 帧 H264 |
| UMI | 3,082 个 H264 媒体消息全部解码为 3,082 帧 |
| A2D `aligned_joints.h5` | 21 个 dataset，分块读完 34,230 行/标量、327,630 个元素 |
| A2D `raw_joints.h5` | 24 个 dataset，分块读完 97,750 行/标量、1,423,622 个元素 |
| EPIC 代表样例 | 23,358 字节、1,354 个 opcode；隔离解析为 674 项列表 |

## 3. WP3 Guida 基础清洗实现

### 3.1 时间和 Source Map

- `index.jsonl` 的 `timestamp_ns` 是视频权威时间；
- 每个输出视频样本记录 source seq、源文件、源 segment、源帧索引和原始时间；
- IMU 允许同一时间戳出现多条记录，但禁止时间回退；
- 每个视频样本建立 nearest-IMU 映射并记录 `error_ns`；
- 映射保存在正式 `sample_map.schema.json` 约束下。

真实 Guida 中 IMU 以同一时间戳成对出现。程序保留这项事实，没有擅自删除其中一行。

### 3.2 当前基础硬检查

- RGB/Depth 是否完整解码；
- 分辨率是否在同一源片段内变化；
- 持续黑帧/纯色帧；
- 持续完全重复帧；
- Depth 是否为 `uint16`；
- Depth invalid value 比例；
- Depth 单位是否明确声明；
- IMU 时间回退、gap、NaN/Inf；
- 视频和 IMU 是否有共同时间覆盖；
- meta 声明的 IMU 文件与实际分片是否一致。

硬坏区间从物理连续区间中排除；足够长的剩余区间形成 Prepared Segment。没有可靠单位或
来源声明冲突时进入 `quarantine`/`keep_with_flag`，程序不会猜测或自动放行。

### 3.3 Prepared 产物

每个 Segment 至少包含：

```text
<segment-id>/
├── segment.json
├── data/
│   ├── rgb.source.json
│   ├── depth.source.json
│   └── imu.csv
├── alignments/
│   ├── video_source_map.json
│   ├── imu_source_map.json
│   └── video_imu_alignment.json
├── calibration/
│   └── calibrations.json
└── quality/
    └── report.json
```

Revision 还包含 `revision.json` 和 `cleaning_report.json`。所有 source asset 保存 URI 和
SHA-256，producer 保存代码版本、配置版本和配置哈希。

RGB/Depth 当前使用 `source_selection_v1`：产物明确声明 `materialized: false`，表示它引用
Raw 中的确定帧范围，并不冒充已生成新的媒体文件。此设计先验证时间、质量、边界和追溯
闭环，后续需要时再增加可复现、无损或明确编码参数的物化步骤。

### 3.4 原子写入与回读

写入流程是：

1. 在目标 Revision 旁创建临时目录；
2. 写入所有文件；
3. 校验 Segment Schema、Sample Map Schema、引用和时间；
4. 同步文件；
5. 原子重命名 Segment；
6. 所有 Segment 完成后原子重命名 Revision。

目标已存在时明确失败，不覆盖旧 Revision。异常时清理本次临时目录。

## 4. 真实 Guida 验收结果

真实样例结果：

- 视频权威帧：983，其中 982 帧位于视频/IMU 共同时间覆盖内；
- IMU 记录：1,966；
- Prepared Segment：1；
- 回读 Schema：通过；
- 内部文件引用：通过；
- Sample Map 连续性和时间单调性：通过；
- Raw source asset SHA-256 重算：通过；
- Raw 文件修改：无。

该样例保留三个业务问题：

- `depth_unit_unknown -> quarantine`：meta 未明确给出深度物理单位；
- `required_stream_missing -> keep_with_flag`：meta 声明 `imu/imu.csv`，实际使用
  `imu/imu_000000.csv`；
- `clock_misalign -> quarantine`：视频帧到最近 IMU 的最大误差为 21.44 ms，超过当前
  配置阈值 20 ms。

因此“程序闭环通过”不等于“样例质量已放行”。在深度单位和 21.44 ms 对齐误差没有得到
确认或修正前，该 Segment 正确状态仍是 quarantine。

## 5. 验收命令

```powershell
python -m zpds clean guida `
  --config configs/pipeline/default.yaml `
  --raw-root D:\datasets\egos `
  --input-ref "raw://墨现" `
  --output-root D:\zpds-output `
  --prep-revision r0001 `
  --code-version local-wp3

python -m zpds prepared validate `
  --segment-dir D:\zpds-output\prepared_segments\r0001\<segment-id> `
  --raw-root D:\datasets\egos

python -m pytest -q
python -m ruff check .
python -m mypy zpds
```

成功的 `prepared validate` 输出应包含：

```json
{"errors":[],"raw_hashes_checked":true,"status":"valid"}
```

## 6. 当前边界和下一步

本轮是“Guida 基础清洗闭环”，不是全部 WP3/全来源生产完成：

- 其余四源还没有生成 Prepared；
- Depth 物理单位仍需业务确认；
- RGB/Depth 尚未物化为新的标准媒体；
- 尚未做标定重投影误差；
- 未实现模糊、曝光、VFR、漂移、饱和等完整 Stage 3～8 算法；
- 未实现人工 Review UI、Scene/Action/VLM、Release 和 Export；
- EPIC 全库内容级隔离扫描尚未实际跑完。

下一步应先确认 Guida 深度单位并把真实样例从 quarantine 重新评估，然后将同一闭环扩展到
遁甲或 UMI：建立各自的 clock/sample map，禁止跨 reset/gap 插值，再复用 Prepared Writer
和 Validator。
