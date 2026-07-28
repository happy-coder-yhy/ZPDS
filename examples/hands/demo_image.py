"""
人员 B — 单张图片手部检测示例。

用法:
    cd ZPDS/
    python examples/hands/demo_image.py [图片路径]

不传参数时生成一张纯色测试图。
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# 确保项目根目录在 sys.path 中
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from zpds.hands.base import RawHandResult, HandKeypoints, HandBBox
from zpds.hands.mediapipe_adapter import MediaPipeHandEstimator


def generate_test_image(width: int = 640, height: int = 480) -> np.ndarray:
    """生成一张纯灰色测试图。"""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def draw_results(frame: np.ndarray, results: list[RawHandResult]) -> np.ndarray:
    """在图像上绘制手部检测结果。"""
    vis = frame.copy()

    # MediaPipe 手部关键点连线拓扑
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
        (0, 5), (5, 6), (6, 7), (7, 8),       # 食指
        (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
        (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
        (0, 17), (17, 18), (18, 19), (19, 20), # 小指
        (5, 9), (9, 13), (13, 17),             # 手指根部横线
    ]

    for result in results:
        kp = result.keypoints

        # 绘制连线
        for start_idx, end_idx in HAND_CONNECTIONS:
            sx, sy = kp.pixel[start_idx]
            ex, ey = kp.pixel[end_idx]
            cv2.line(vis, (int(sx), int(sy)), (int(ex), int(ey)),
                     (0, 255, 0), 2)

        # 绘制关键点
        for px, py in kp.pixel:
            cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

        # 绘制 BBox
        bbox = result.bbox
        cv2.rectangle(
            vis,
            (int(bbox.x1), int(bbox.y1)),
            (int(bbox.x2), int(bbox.y2)),
            (255, 0, 0), 2,
        )

        # 绘制标签
        label = f"{result.handedness} {result.handedness_score:.2f}"
        cv2.putText(vis, label, (int(bbox.x1), int(bbox.y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    return vis


def main():
    image_path = sys.argv[1] if len(sys.argv) > 1 else None

    if image_path:
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"无法读取图片: {image_path}")
            sys.exit(1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        print(f"加载图片: {image_path}  ({frame_rgb.shape[1]}×{frame_rgb.shape[0]})")
    else:
        print("未指定图片，使用纯灰色测试图（不会检测到手）")
        frame_rgb = generate_test_image()

    print(f"\n初始化 MediaPipe Hand Landmarker ...")
    estimator = MediaPipeHandEstimator(
        model_path="models/mediapipe/hand_landmarker.task",
        num_hands=2,
    )
    print(f"  最大手数: {estimator.config.num_hands}")
    print(f"  检测阈值: {estimator.config.min_hand_detection_confidence}")
    print(f"  BBox 边距: {estimator.config.bbox_padding_ratio}")

    print(f"\n执行推理 (timestamp_ms=0) ...")
    results = estimator.estimate(frame_rgb, timestamp_ms=0)

    print(f"检测到 {len(results)} 只手\n")

    for i, r in enumerate(results):
        print(f"--- 手 #{i} ---")
        print(f"  label:           {r.label}")
        print(f"  handedness:      {r.handedness}")
        print(f"  handedness_score:{r.handedness_score:.4f}")
        print(f"  detection_score: {r.detection_score:.4f}")
        print(f"  BBox:            ({r.bbox.x1:.1f}, {r.bbox.y1:.1f}) → "
              f"({r.bbox.x2:.1f}, {r.bbox.y2:.1f}) "
              f"[padded={r.bbox.is_padded}, ratio={r.bbox.padding_ratio}]")
        print(f"  关键点数量:      {len(r.keypoints.normalized)}")
        print(f"  关键点 visibility:{r.keypoints.has_visibility}")

        # 打印前 3 个关键点的归一化坐标
        for j in range(min(3, 21)):
            nx, ny, nz = r.keypoints.normalized[j]
            px, py = r.keypoints.pixel[j]
            print(f"    kp[{j}]: norm=({nx:.4f}, {ny:.4f}, {nz:.4f}) "
                  f"pixel=({px:.1f}, {py:.1f})")

        # 检查关键点是否在有效范围
        out_of_bounds = 0
        for nx, ny, nz in r.keypoints.normalized:
            if nx < 0 or nx > 1 or ny < 0 or ny > 1:
                out_of_bounds += 1
        if out_of_bounds:
            print(f"  ⚠ 越界关键点数: {out_of_bounds}")
        else:
            print(f"  ✓ 所有关键点在 [0,1] 范围内")

    # 打印耗时
    print(f"\n--- 推理耗时 ---")
    for t in estimator.timing_history:
        print(f"  前处理:   {t.preprocess_ms:.2f} ms")
        print(f"  推理:     {t.inference_ms:.2f} ms")
        print(f"  后处理:   {t.postprocess_ms:.2f} ms")
        print(f"  总计:     {t.total_ms:.2f} ms  ({t.fps:.1f} FPS)")

    # 保存可视化结果
    output_dir = Path("output/hands")
    output_dir.mkdir(parents=True, exist_ok=True)

    vis_frame = draw_results(frame_rgb, results)
    vis_bgr = cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR)
    out_path = output_dir / "demo_image_result.jpg"
    cv2.imwrite(str(out_path), vis_bgr)
    print(f"\n可视化结果已保存: {out_path}")

    estimator.close()
    print("完成。")


if __name__ == "__main__":
    main()
