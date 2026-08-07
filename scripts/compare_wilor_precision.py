"""WiLoR FP16 vs FP32 精度与速度对照。

用法（需在 wilor_env 中运行，有 torch）::

    e:/ZSPD/wilor_env/Scripts/python.exe scripts/compare_wilor_precision.py \
        output/taodai2/prepared_segments/r0001/seg_000001/data/ego_rgb.mp4 [N=30]

对视频均匀采样 N 帧，分别用 float32 / float16 两种 precision 推理，
比较检测到的手数量、3D 关节 L2 误差（mm）、相机参数差异与每帧耗时。
检测框由同一 YOLO 在同一帧上生成，顺序稳定，按序配对。
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


def build_config(precision: str) -> WiLoRConfig:
    return WiLoRConfig(
        checkpoint_path="e:/ZSPD/WiLoR/pretrained_models/wilor_final.ckpt",
        expected_sha256="",
        wilor_source_path="e:/ZSPD/WiLoR",
        detector_path="e:/ZSPD/WiLoR/pretrained_models/detector.pt",
        model_config_path="e:/ZSPD/WiLoR/pretrained_models/model_config.yaml",
        device="cuda",
        precision=precision,
        model_version="wilor_cvpr2025",
    )


def sample_frames(video_path: Path, n: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"无法打开视频: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    total = max(total, 1)
    n = min(n, total)
    idxs = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok:
            continue
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    if not frames:
        raise SystemExit(f"未能从视频采样到帧: {video_path}")
    return frames


def joint_error(a: np.ndarray, b: np.ndarray) -> float:
    """逐关节 L2 欧氏距离均值（mm）。"""
    return float(np.linalg.norm(a - b, axis=1).mean())


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    video_path = Path(sys.argv[1])
    n_frames = int(sys.argv[2]) if len(sys.argv) > 2 else 30

    frames = sample_frames(video_path, n_frames)
    print(f"视频: {video_path}  采样 {len(frames)} 帧")

    backends = {
        prec: WiLoRBackend(build_config(prec)) for prec in ("float32", "float16")
    }
    try:
        kp3d_errors: list[float] = []
        cam_errors: list[float] = []
        hand_counts = {"float32": [], "float16": []}
        timings = {"float32": [], "float16": []}
        mismatched_frames = 0
        no_hand_frames = 0

        for i, frame in enumerate(frames):
            results = {}
            for prec, backend in backends.items():
                t0 = time.perf_counter()
                results[prec] = backend.infer_raw(frame)
                timings[prec].append((time.perf_counter() - t0) * 1000)

            f32, f16 = results["float32"], results["float16"]
            hand_counts["float32"].append(len(f32["pred_keypoints_3d"] or []))
            hand_counts["float16"].append(len(f16["pred_keypoints_3d"] or []))

            if len(f32["pred_keypoints_3d"] or []) != len(f16["pred_keypoints_3d"] or []):
                mismatched_frames += 1
                continue
            n_hands = len(f32["pred_keypoints_3d"] or [])
            if n_hands == 0:
                no_hand_frames += 1
                continue

            for n in range(n_hands):
                kp3d_errors.append(
                    joint_error(f32["pred_keypoints_3d"][n], f16["pred_keypoints_3d"][n])
                )
                cam_errors.append(
                    float(np.abs(f32["pred_cam"][n] - f16["pred_cam"][n]).max())
                )
            if (i + 1) % 10 == 0:
                print(f"  {i + 1}/{len(frames)} 帧完成")

        print("\n═══════════════ 结果汇总 ═══════════════")
        print(f"  对照帧: {len(frames)}（关节样本 {len(kp3d_errors)} 个）")
        print(f"  检测手数不一致帧: {mismatched_frames} / {len(frames)}")
        print(f"  无手帧: {no_hand_frames} / {len(frames)}")

        def stats(name: str, values: list[float]) -> str:
            arr = np.asarray(values, dtype=np.float64)
            return (
                f"{name}: mean={arr.mean():.3f}  max={arr.max():.3f}  "
                f"p95={np.percentile(arr, 95):.3f}  (n={len(arr)})"
            )

        if kp3d_errors:
            print(f"\n3D 关节 L2 误差 (mm)  {stats('', kp3d_errors)}")
            print(f"pred_cam 最大绝对差   {stats('', cam_errors)}")

        print("\n每帧推理耗时 (ms):")
        for prec in ("float32", "float16"):
            arr = np.asarray(timings[prec], dtype=np.float64)
            print(
                f"  {prec:<8} mean={arr.mean():8.1f}  p50={np.median(arr):8.1f}  "
                f"p95={np.percentile(arr, 95):8.1f}"
            )
        ratio = np.asarray(timings["float32"]).mean() / np.asarray(timings["float16"]).mean()
        print(f"  加速比: {ratio:.2f}x")

        print(f"\n模型精度: float32 → {backends['float32'].model_info.precision}")
        print(f"           float16 → {backends['float16'].model_info.precision}")
    finally:
        for backend in backends.values():
            backend.close()


if __name__ == "__main__":
    main()
