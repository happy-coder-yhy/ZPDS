"""
人员 B — 手部检测示例（单图 + 可选 YAML 配置）。

用法:
    # 默认配置（auto 模式）
    python examples/hands/demo_image.py

    # 指定图片
    python examples/hands/demo_image.py image.jpg

    # 强制使用 Solutions 后端
    python examples/hands/demo_image.py image.jpg --backend solutions_hands

    # 从 YAML 加载配置
    python examples/hands/demo_image.py image.jpg --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zpds.hands.base import RawHandResult
from zpds.hands.mediapipe_adapter import MediaPipeHandEstimator


def generate_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def draw_results(frame: np.ndarray, results: list[RawHandResult]) -> np.ndarray:
    vis = frame.copy()

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17),
    ]

    for result in results:
        kp = result.keypoints
        for start_idx, end_idx in HAND_CONNECTIONS:
            sx, sy = kp.pixel[start_idx]
            ex, ey = kp.pixel[end_idx]
            cv2.line(vis, (int(sx), int(sy)), (int(ex), int(ey)),
                     (0, 255, 0), 2)
        for px, py in kp.pixel:
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)
        bbox = result.bbox
        cv2.rectangle(
            vis, (int(bbox.x1), int(bbox.y1)), (int(bbox.x2), int(bbox.y2)),
            (255, 0, 0), 2,
        )
        label = f"{result.handedness} {result.handedness_score:.2f}"
        cv2.putText(vis, label, (int(bbox.x1), int(bbox.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return vis


def main():
    parser = argparse.ArgumentParser(description="手部检测单图示例")
    parser.add_argument("image", nargs="?", help="图片路径（不传则用灰色测试图）")
    parser.add_argument("--config", "-c", help="YAML 配置文件路径")
    parser.add_argument("--backend", "-b",
                        choices=["auto", "tasks_hand_landmarker", "solutions_hands"],
                        default="auto",
                        help="后端选择（默认 auto）")
    parser.add_argument("--output", "-o", default="output/hands/demo_image_result.jpg",
                        help="可视化输出路径")
    args = parser.parse_args()

    # ---- 初始化 ----
    if args.config:
        print(f"从配置文件加载: {args.config}")
        estimator = MediaPipeHandEstimator.from_yaml(args.config)
    else:
        estimator = MediaPipeHandEstimator(backend=args.backend)

    # ---- 打印后端信息 ----
    info = estimator.backend_info
    print(f"\n后端信息:")
    print(f"  请求后端:     {info.requested_backend}")
    print(f"  活跃后端:     {info.active_backend}")
    print(f"  触发回退:     {info.fallback_used}")
    if info.fallback_reason:
        print(f"  回退原因:     {info.fallback_reason}")
    if info.delegate:
        print(f"  推理设备:     {info.delegate}")
    print(f"  最大手数:     {estimator.config.num_hands}")
    print(f"  BBox 边距:    {estimator.config.bbox_padding_ratio}")

    # ---- 打印模型信息 ----
    mi = estimator.model_info
    print(f"\n模型信息:")
    print(f"  文件:         {mi.path}")
    print(f"  存在:         {mi.exists}")
    if mi.exists:
        print(f"  大小:         {mi.size_bytes / 1024 / 1024:.1f} MB")
        print(f"  SHA-256:      {mi.sha256[:16]}...")
    else:
        print(f"  下载地址:     {mi.download_url}")
    print(f"  初始化耗时:   {estimator.session_stats.init_time_ms:.0f} ms")

    # ---- 加载/生成图片 ----
    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            print(f"无法读取图片: {args.image}")
            sys.exit(1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"\n图片: {args.image} ({frame_rgb.shape[1]}x{frame_rgb.shape[0]})")
    else:
        print("\n未指定图片，使用灰色测试图")
        frame_rgb = generate_test_image()

    # ---- 推理 ----
    print(f"\n执行推理 (timestamp_ms=0) ...")
    results = estimator.estimate(frame_rgb, timestamp_ms=0)
    print(f"检测到 {len(results)} 只手\n")

    for i, r in enumerate(results):
        print(f"--- 手 #{i} ---")
        print(f"  label:           {r.label}")
        print(f"  handedness:      {r.handedness}")
        print(f"  handedness_score:{r.handedness_score:.4f}")
        print(f"  BBox:            ({r.bbox.x1:.1f}, {r.bbox.y1:.1f}) -> "
              f"({r.bbox.x2:.1f}, {r.bbox.y2:.1f})")
        for j in range(min(3, 21)):
            nx, ny, nz = r.keypoints.normalized[j]
            px, py = r.keypoints.pixel[j]
            print(f"    kp[{j}]: norm=({nx:.4f}, {ny:.4f}, {nz:.4f}) "
                  f"pixel=({px:.1f}, {py:.1f})")

    # ---- 耗时 ----
    print(f"\n--- 推理耗时 ---")
    for t in estimator.timing_history:
        print(f"  前处理:   {t.preprocess_ms:.2f} ms")
        print(f"  推理:     {t.inference_ms:.2f} ms")
        print(f"  后处理:   {t.postprocess_ms:.2f} ms")
        print(f"  总计:     {t.total_ms:.2f} ms  ({t.fps:.1f} FPS)")

    # ---- 保存可视化 ----
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    vis_frame = draw_results(frame_rgb, results)
    vis_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(output_path), vis_bgr)
    print(f"\n可视化结果: {output_path}")

    # ---- 会话统计 ----
    stats = estimator.session_stats
    print(f"\n会话统计:")
    print(f"  总帧数:       {stats.total_frames}")
    print(f"  有手帧:       {stats.hand_frames}")
    print(f"  无手帧:       {stats.no_hand_frames}")
    print(f"  异常帧:       {stats.exception_frames}")
    print(f"  总推理耗时:   {stats.total_inference_ms:.1f} ms")
    print(f"  平均耗时:     {stats.avg_inference_ms:.1f} ms")

    estimator.close()
    print("完成。")


if __name__ == "__main__":
    main()
