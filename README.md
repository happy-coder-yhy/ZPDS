ZPDS（Ziki Physical AI Data Specification）数据清洗与标准化

多源自我中心数据（墨现头戴 / 遁甲机器狗 / UMI / EPIC / A2D 真机）→ QC 检测 → 场景分段 →
**Prepared Segment**（统一训练格式，segment.json + data/ + maps/ + calibration/ + reports/）。

## 新数据处理 Checklist（2026-08 复盘后固化）

处理一份新数据前照做，可避免约 20 分钟的常见错误重跑：

1. **数据放英文路径**：`E:/datasets/egos/<英文名>/`（数据源可含中文，**输出目录必须英文**，
   cv2 不支持中文路径，代码已 fail-fast）
2. **主流程**（QC + 可选手部/场景，场景/手部产物会自动复用）：
   ```bash
   cd /e/ZSPD/ZPDS && export PYTHONPATH="e:/ZSPD/ZPDS"
   python -m zpds_prepare.main "E:/datasets/egos/<数据>" --profile <guida|dunjia|umi> \
     --with-hands --with-scene --output output/<英文名>/
   ```
3. **批量 Prepared Segment + 脱敏**（`-d` 必填；`--output` 必须传 `.../prepared_segments`，
   与 hands/scene 产物目录对齐）：
   ```bash
   python batch_prepare.py -d "E:/datasets/egos/<数据>" --profile <guida|dunjia|umi> \
     --output output/<英文名>/prepared_segments --with-privacy
   ```
4. **验证**：`reports/validation.json` status=pass；`-d` 传错时帧数预检会立即中止并报错
5. **产物位置**：`output/<英文名>/prepared_segments/r0001/seg_000001/`
   （segment.json + data/ + maps/ + calibration/ + reports/ + hands/ + scene/）

已知环境约定：主代码 `.venv`（Python 3.11）；WiLoR 推理 `e:/ZSPD/wilor_env/`；
ffmpeg/ffprobe 必须在 PATH（转码、深度解码、libx264 重编码、预览均依赖）。
