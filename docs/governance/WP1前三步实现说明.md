# WP1 前三步实现说明

本阶段只建立可验证的配置与 Stage 契约，不包含 Storage、Run Ledger、Runner 或真实
Guida Adapter。

## 1. Prepared 与 Alignment 配置边界

`configs/pipeline/default.yaml` 和 `pilot.yaml` 统一规定：

```yaml
prepared:
  preserve_source_rate: true

alignment:
  enabled: true
  target_fps: 30
  method: nearest
```

含义是 Prepared 保存原始采样频率；30 FPS 只生成派生 Alignment View。后续不同
模态可以按 Profile 选择 nearest、linear、SLERP 或 ZOH，并记录 sample map。

`pipeline_config.schema.json` 拒绝把 `target_fps` 再写回 `prepared`。

## 2. Config Loader 与 Schema Registry

加载配置：

```python
from zpds.config import PipelineConfigLoader

config = PipelineConfigLoader().load("configs/pipeline/pilot.yaml")
print(config.version)
print(config.config_hash)
print(config.section("prepared")["preserve_source_rate"])
```

Loader 不注入隐式默认值，并执行：

- YAML 顶层对象检查；
- 按 `pipeline.version` 选择 Schema；
- 必填字段、类型、范围和未知字段检查；
- 对规范化配置语义计算 `sha256`；
- 返回不可变配置快照。

Schema Registry 使用 `(object_type, version)` 作为键。未注册版本会明确失败，不会回退
到“最接近”的版本。

## 3. Pipeline Stage 契约

一个 Stage 提供稳定的 `StageDescriptor` 和 `execute(context)`：

```python
from zpds.pipeline import StageDescriptor

descriptor = StageDescriptor(
    stage_id=0,
    name="file_registry",
    version="0.1.0",
)
```

`StageContext` 固定一次运行的 run、session、输入引用、配置快照和代码版本。
`StageResult` 只表示终态：

- `succeeded`
- `failed`
- `skipped`

`pending/running` 将由后续 Run Ledger 管理。StageResult 会检查：

- Stage 编号为 0～12；
- 名称使用 lower_snake_case；
- 版本使用 SemVer；
- 配置哈希格式正确；
- 开始/结束时间带时区且顺序正确；
- failed 必须携带错误；
- Decision 的 stage 必须与当前 Stage 一致。

## 验证

```powershell
python -m pytest tests/test_config tests/test_pipeline -q -p no:cacheprovider
python -m pytest -q -p no:cacheprovider
python -m ruff check .
```

## 下一步边界

下一步依次实现：

1. 本地 Storage Adapter 与原子写入；
2. Run Ledger；
3. 幂等、重试和断点恢复 Runner；
4. CLI 与结构化日志；
5. Guida Stage 0～2 最小闭环。
