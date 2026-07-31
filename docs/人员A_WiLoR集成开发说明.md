# 人员 A：WiLoR 集成开发说明

## 1. 职责与完成状态

人员 A 负责将 Prepared Segment、Sample Map、模型工厂、逐帧 Pipeline、产物 Writer、
Validator 和批处理完成条件连接成可执行的端到端流程。

当前已经完成：

- ego 数据路由到 WiLoR，non-ego 数据保留 MediaPipe。
- 每个 Prepared 输出帧都执行一次模型调用并产生明确状态。
- 保留 `detected`、`no_hand`、`failed`、`skipped_invalid_input` 四种状态。
- 单帧失败不会被记成 `no_hand`，也不会导致后续帧静默丢失。
- 接入人员 B 的真实 WiLoR backend、checkpoint、MANO 和 21 点映射。
- 接入人员 C 的 Hands V1 Writer、WiLoR Validator、Preview 和 Experience Manifest。
- 实现正式的 WiLoR frame-status/BBox Parquet Writer。
- 将主 WiLoR 状态和显式 MediaPipe 回退结果分开记录。
- 增加 run manifest、全帧覆盖率、失败率和产物完整性门禁。
- 完成真实 20 帧 smoke 和 984 帧完整 Prepared Segment 验收。

人员 A 没有修改人员 C 的 `writer.py`、`validator.py`、`preview.py` 或
`experience.py`。

## 2. 配置和路由

配置采用 WiLoR 与 MediaPipe 并行结构：

```yaml
hands:
  ego_bbox_backend: wilor
  non_ego_bbox_backend: mediapipe
  fallback_2d_backend: mediapipe

  mediapipe:
    backend: tasks_hand_landmarker

  wilor:
    enabled: true
    ego_bbox_every_frame: true
    write_frame_status: true
```

路由规则：

- `--source-kind ego`：使用 WiLoR。
- `--source-kind non_ego`：使用 MediaPipe。
- WiLoR 初始化、checkpoint、commit 或许可证检查失败时明确终止。
- 禁止将 WiLoR 初始化失败静默替换成 MediaPipe 主模型。

WiLoR 运行时会校验：

- 上游 Git commit。
- checkpoint SHA-256。
- 模型版本。
- 设备和精度配置。
- MANO、模型配置、detector 和关节映射资源。

## 3. Pipeline 契约

`HandsPipeline` 同时保留两类信息：

1. WiLoR 主尝试的逐帧状态，用于全帧覆盖和失败统计。
2. 最终有效手部结果，用于 Hands V1 写出；显式回退时可以来自 MediaPipe。

逐帧记录为 `FrameInferenceRecord`，主要字段包括：

```text
frame
inference_status
raw_hands
effective_hands
frame_result
failure_reason
active_backend
inference_ms
```

语义：

- `raw_hands`：主 WiLoR 尝试的结果。
- `effective_hands`：最终可写入 Hands V1 的结果。
- WiLoR 失败但 MediaPipe 回退成功时：
  - frame status 仍为 `failed`；
  - WiLoR BBox 表不伪造检测行；
  - Hands V1 可以写入 MediaPipe 行；
  - 行内必须标记 requested/active backend 和 fallback reason。

## 4. 正式产物

一次 ego WiLoR 运行生成：

```text
hands_2d.parquet
wilor_frame_status.parquet
wilor_hands_bbox.parquet
hands_run.json
hands_validation.json
hands_preview.mp4        # 使用 --preview 时
experience_manifest.json # 使用 --experience-dir 时
```

### 4.1 `wilor_frame_status.parquet`

一帧一行，字段包括：

```text
prep_revision, segment_id, video_stream_id,
output_frame_index, timestamp_ns,
source_frame_index, source_timestamp_ns,
inference_status, hand_count, failure_reason,
model_name, model_version,
checkpoint_sha256, config_sha256,
active_backend, device, inference_ms
```

验收要求：

- 行数等于本次处理的 Sample Map 行数。
- `output_frame_index` 和 `timestamp_ns` 与 Sample Map 逐行一致。
- `detected` 必须至少有一只主模型手部结果。
- `failed` 和 `skipped_invalid_input` 必须有失败原因。
- 正式 ego run 不允许 `skipped_invalid_input`。
- 失败率必须低于 2%。

### 4.2 `wilor_hands_bbox.parquet`

WiLoR 主尝试检测到的一只手一行，字段包括：

```text
prep_revision, segment_id, video_stream_id,
output_frame_index, timestamp_ns,
source_frame_index, source_timestamp_ns,
detection_id, handedness, handedness_score, detection_score,
bbox_x1, bbox_y1, bbox_x2, bbox_y2,
model_name, model_version,
checkpoint_sha256, config_sha256
```

该表只写 WiLoR 主尝试的 BBox。MediaPipe 回退结果不会冒充 WiLoR BBox。

### 4.3 `hands_2d.parquet`

复用人员 C 的 `write_hand_observations()` 写出。WiLoR 21 点映射已经接通，因此正式运行会同时写入
Hands V1。每行保留模型和 backend attribution，能够区分 WiLoR 与显式 MediaPipe 回退。

## 5. Validator 和完成条件

人员 A 新增的全帧资产校验包括：

- 两张 Parquet 可读性和字段检查。
- frame status 与 Sample Map 的逐行对齐。
- 四种状态及失败原因语义。
- WiLoR 失败率和 skipped 数量。
- BBox 坐标合法性。
- 每帧 BBox 行数与 `hand_count` 一致。
- 模型版本、checkpoint 和配置哈希完整性。

人员 C 的 WiLoR Validator 继续检查：

- Hands V1 Schema。
- 关键点是否在原图范围内。
- BBox 是否包含关键点。
- wrist 与 BBox 的重投影一致性。
- WiLoR provenance。
- fallback attribution。
- run 级失败率。

生产 run 只有同时满足以下条件才会写为 `completed=true`：

- 没有使用 `--max-frames`。
- 已处理帧数等于 Sample Map 和 decoded frame 数。
- 四种 frame status 计数完整。
- frame-status 和 BBox 资产存在且校验通过。
- 请求 Validator 时，Validator 结果不是 `fail`。

Smoke run 会保留全部产物，但固定标记为 `completed=false`。

## 6. 主要代码

| 文件 | 作用 |
|---|---|
| `zpds/hands/contracts.py` | Estimator、Frame Record 和 Writer 最小契约 |
| `zpds/hands/backend_router.py` | ego/non-ego 后端路由 |
| `zpds/hands/config.py` | 并行配置解析、路径和哈希 |
| `zpds/hands/estimator_factory.py` | MediaPipe/WiLoR 创建及运行时契约 |
| `zpds/hands/pipeline.py` | 逐帧推理、状态保留和 Hands V1 转换 |
| `zpds/hands/frame_artifacts.py` | 正式 frame-status/BBox Writer 和校验 |
| `zpds/hands/orchestration.py` | Writer 工厂与生命周期 |
| `scripts/run_hands.py` | 单 Segment 端到端入口 |
| `scripts/batch_run_hands.py` | 批处理、断点续跑和完成门禁 |
| `scripts/check_wilor_readiness.py` | WiLoR 资产就绪检查 |
| `scripts/smoke_wilor_segment.py` | 真实 WiLoR Pipeline smoke |

## 7. 验证命令

### 7.1 资产检查

```powershell
.\wilor_env\Scripts\python.exe -m scripts.check_wilor_readiness
```

### 7.2 单元测试

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_hands tests\hands -q
```

当前结果：

```text
266 passed
```

### 7.3 静态检查

```powershell
.\.venv\Scripts\python.exe -m ruff check `
  scripts\run_hands.py `
  scripts\smoke_wilor_segment.py `
  zpds\hands\config.py `
  zpds\hands\contracts.py `
  zpds\hands\estimator_factory.py `
  zpds\hands\frame_artifacts.py `
  zpds\hands\orchestration.py `
  zpds\hands\pipeline.py
```

### 7.4 20 帧真实 smoke

```powershell
.\wilor_env\Scripts\python.exe -m scripts.run_hands `
  --segment output\guida_final\prepared_segments\seg_000001 `
  --stream-id ego_rgb `
  --source-kind ego `
  --config config.yaml `
  --output output\hands_wilor_smoke\hands_2d.parquet `
  --max-frames 20 `
  --validate
```

验收结果：

```text
processed=20
detected=20
observations=40
validation=pass
```

### 7.5 完整 Segment

```powershell
.\wilor_env\Scripts\python.exe -m scripts.run_hands `
  --segment output\guida_final\prepared_segments\seg_000001 `
  --stream-id ego_rgb `
  --source-kind ego `
  --config config.yaml `
  --output output\hands_wilor_full\hands_2d.parquet `
  --validate `
  --preview
```

本次 984 帧结果：

```text
frame_status_rows=984
detected=982
no_hand=2
failed=0
skipped_invalid_input=0
bbox_rows=1889
hands_v1_rows=1889
sample_map_alignment=pass
frame_artifact_validation=pass
```

## 8. 当前剩余问题

完整 Segment 的最终 Hands Validator 仍为 `fail`，原因只有一条：

```text
Hands V1 row=561
output_frame_index=288
detection_id=2
wrist_y=504.725
bbox_y2=502.0
outside=2.725 px
validator_tolerance=2 px
```

这条记录同时导致 `bbox_contains_keypoints=warn` 和
`reprojection_consistency=fail`。A 没有放宽 Validator 或修改数据掩盖问题。

该问题需要人员 B 检查：

- detector BBox padding 是否应用。
- crop/resize 逆变换是否一致。
- wrist 关节映射和回投影是否正确。
- 低置信度第三个检测是否应该被保留。

在该问题解决并重新运行完整 Validator 前，run manifest 正确保持
`completed=false`，不能宣称达到最终 `hand_pose_ready` 发布门槛。

