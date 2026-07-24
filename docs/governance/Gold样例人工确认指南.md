# Gold 样例人工确认指南

## 当前准备状态

五源 10 个样例已经绑定真实相对路径、文件大小和 SHA256。哈希过程只读取文件字节，
不会解析或修改 Raw，尤其不会反序列化 EPIC Pickle。

| 样例 | 已确认的客观事实 | 仍需人工/后续工具确认 |
| --- | --- | --- |
| `guida_authoritative_index` | index 共 985 行；RGB/Depth/IMU 已绑定 | 逐帧对应和 keep 决策 |
| `guida_meta_imu_mismatch` | meta 指向的 `imu/imu.csv` 不存在，分片存在 | `keep_with_flag` 是否符合业务 |
| `dunjia_h264_timing` | MCAP 文件与哈希已绑定 | topic、GOP、消息时间和 sample map |
| `dunjia_dual_clock` | MCAP 文件与哈希已绑定 | 双时钟偏差、阈值和真实异常窗口 |
| `umi_dual_robot_groups` | MCAP 文件与哈希已绑定 | robot0/robot1 的业务含义和 topic 清单 |
| `umi_vio_reset` | MCAP 文件与哈希已绑定 | 真实 VIO reset 时间窗与 pose 证据 |
| `a2d_complete_camera_tuple` | `camera/0` 六个相机文件齐全 | 解码、内容和 keep 决策 |
| `a2d_incomplete_camera_tuple` | `camera/1147` 只有六项中的三项 | HDF5/ROS 映射与 quarantine 决策 |
| `epic_annotation_linked` | 两个 Pickle 已做字节级哈希 | 隔离解析、原视频和模型版本关联 |
| `epic_annotation_orphaned` | Pickle 已绑定，当前根下无原视频 | orphaned/许可结论 |

标记为“仍需确认”的内容不得因为文件存在或哈希正确而自动批准。

## 1. 检查真实资产没有变化

在 PowerShell 中：

```powershell
cd D:\ZPDS
python -m scripts.manage_gold collect --data-root D:\datasets
```

期望：

```text
assets_changed=0
dry-run: add --write to update the Manifest
```

如果 `assets_changed` 大于 0，说明文件内容、大小或 Manifest 记录发生变化。先调查，
不要直接批准。确认确实要更新绑定时才运行：

```powershell
python -m scripts.manage_gold collect --data-root D:\datasets --write
```

资产发生变化时，工具会自动把对应样例重置为 `pending`，防止旧审核结论继续生效。

## 2. 查看待审状态

```powershell
python -m scripts.manage_gold status
```

初始状态应为 10 个 `pending`。审核人逐项打开 Manifest 中的 Evidence 和原始样例，
确认文件、问题、边界、Reason Code 与预期 Decision。

## 3. 记录单人审核

确认一个样例后运行：

```powershell
python -m scripts.manage_gold review `
  --sample guida_meta_imu_mismatch `
  --status approved `
  --notes "已核对 meta 路径差异和预期决策。" `
  --write
```

Manifest 已指定审核账号 `xiandongfu123-droid`，因此可以省略 `--reviewer`；
工具会自动使用该账号并记录带时区的 `reviewed_at`。如果显式传入其他账号，审核会失败。
如果证据不足：

```powershell
python -m scripts.manage_gold review `
  --sample umi_vio_reset `
  --status pending `
  --notes "缺少 VIO reset 的真实时间窗和 pose 跳变图。" `
  --write
```

如果确认样例定义错误：

```powershell
python -m scripts.manage_gold review `
  --sample <sample_id> `
  --status rejected `
  --notes "写明拒绝原因。" `
  --write
```

每个写操作不加 `--write` 时只是预演，不修改文件。

## 4. 冻结前检查

只有 10 个样例全部 `approved` 才能冻结：

```powershell
python -m scripts.manage_gold freeze --data-root D:\datasets
```

该命令会重新读取所有 Raw 资产并核对 SHA256。预演通过后才执行：

```powershell
python -m scripts.manage_gold freeze --data-root D:\datasets --write
```

最后运行：

```powershell
python -m scripts.validate_wp0
python -m pytest -q
python -m ruff check .
```

## 审核边界

程序可以证明文件存在、大小、哈希、结构字段和审核记录完整，但不能代替人判断：

- 某个 VIO 跳变是否是真正 reset；
- 数据是否适合业务训练目标；
- `quarantine` 是否比 `keep_with_flag` 更合适；
- 许可或隐私是否满足组织政策。

因此程序不会自动把 `pending` 改成 `approved`。
