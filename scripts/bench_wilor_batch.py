"""WiLoR 跨帧 batch vs 逐帧：输出一致性 + 加速实测。

用法（需在 wilor_env 中运行）::

    e:/ZSPD/wilor_env/Scripts/python.exe scripts/bench_wilor_batch.py \
        output/taodai2/prepared_segments/r0001/seg_000001/data/ego_rgb.mp4 [N=48]

对同一批帧分别走 infer_raw（逐帧）与 infer_batch（跨帧合并，bs=16），
验证每帧检测手数一致、3D 关节几乎一致，并统计每帧平均耗时。
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from zpds.hands.backends.wilor import WiLoRBackend
from zpds.hands.wilor_schema import WiLoRConfig


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    video_path = Path(sys.argv[1])
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 48

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    n_frames = min(n_frames, total)
    idxs = np.linspace(0, total - 1, n_frames).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"未能采样到帧: {video_path}")
    print(f"视频: {video_path}  采样 {len(frames)} 帧")

    config = WiLoRConfig(
        checkpoint_path="e:/ZSPD/WiLoR/pretrained_models/wilor_final.ckpt",
        expected_sha256="",
        wilor_source_path="e:/ZSPD/WiLoR",
        detector_path="e:/ZSPD/WiLoR/pretrained_models/detector.pt",
        model_config_path="e:/ZSPD/WiLoR/pretrained_models/model_config.yaml",
        device="cuda",
        precision="float16",
        inference_batch_size=16,
        model_version="wilor_cvpr2025",
    )
    backend = WiLoRBackend(config)
    try:
        # ---- 预热（模型加载后首帧编译 kernel） ----
        backend.infer_batch([frames[0]])

        # ---- 逐帧路径 ----
        t0 = time.perf_counter()
        single = [backend.infer_raw(f) for f in frames]
        single_ms = (time.perf_counter() - t0) * 1000 / len(frames)

        # ---- 跨帧 batch 路径 ----
        t0 = time.perf_counter()
        batched = backend.infer_batch(frames)
        batch_ms = (time.perf_counter() - t0) * 1000 / len(frames)

        # ---- 一致性 ----
        mismatched = 0
        max_joint_err = 0.0
        n_compared = 0
        for i, (a, b) in enumerate(zip(single, batched)):
            na = len(a["pred_keypoints_3d"] or [])
            nb = len(b["pred_keypoints_3d"] or [])
            if na != nb:
                mismatched += 1
                continue
            for n in range(na):
                err = float(
                    np.linalg.norm(
                        a["pred_keypoints_3d"][n].astype(np.float64)
                        - b["pred_keypoints_3d"][n].astype(np.float64),
                        axis=1,
                    ).mean()
                )
                max_joint_err = max(max_joint_err, err)
                n_compared += 1

        print("\n═══════════════ 结果 ═══════════════")
        print(f"手数不一致帧: {mismatched} / {len(frames)}")
        print(f"3D 关节平均 L2 差: max={max_joint_err * 1000:.3f} mm (n={n_compared})")
        print(f"\n每帧耗时 (ms):")
        print(f"  逐帧      mean={single_ms:8.1f}")
        print(f"  跨帧batch mean={batch_ms:8.1f}")
        print(f"  加速比: {single_ms / batch_ms:.2f}x")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
