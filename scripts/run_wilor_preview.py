"""WiLoR Preview — 验证 21 点映射骨架连线。

用法:
    wilor_env/Scripts/python.exe scripts/run_wilor_preview.py image.jpg

输出 output/hands/wilor_preview.jpg
"""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np

from zpds.hands.backends.wilor import WiLoRBackend
from zpds.hands.wilor_adapter import WiLoRAdapter
from zpds.hands.wilor_joint_mapping import (
    WILOR_TO_HANDS_V1_V1,
    convert_wilor_to_raw_hand_result,
    is_mapping_ready,
)
from zpds.hands.wilor_schema import WiLoRConfig

# MediaPipe Hand Connections — 用于骨架绘制
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # 拇指
    (0, 5), (5, 6), (6, 7), (7, 8),        # 食指
    (0, 9), (9, 10), (10, 11), (11, 12),   # 中指
    (0, 13), (13, 14), (14, 15), (15, 16), # 无名指
    (0, 17), (17, 18), (18, 19), (19, 20), # 小指
    (5, 9), (9, 13), (13, 17),             # 掌部横向连接
]

# 每根手指的颜色
FINGER_COLORS = [
    (0, 255, 0),     # 拇指 - 绿
    (255, 0, 0),     # 食指 - 蓝
    (0, 0, 255),     # 中指 - 红
    (255, 255, 0),   # 无名指 - 青
    (255, 0, 255),   # 小指 - 紫
]

FINGER_RANGES = [
    (1, 5),    # 拇指: 1-4
    (5, 9),    # 食指: 5-8
    (9, 13),   # 中指: 9-12
    (13, 17),  # 无名指: 13-16
    (17, 21),  # 小指: 17-20
]


def draw_skeleton(
    frame: np.ndarray,
    keypoints: list[tuple[float, float]],
    bbox: tuple[float, float, float, float] | None = None,
) -> np.ndarray:
    """在图像上绘制 21 点骨架 + BBox。"""
    vis = frame.copy()

    # 手掌连线（白色）
    palm_connections = [(5, 9), (9, 13), (13, 17)]
    for si, ei in palm_connections:
        if si < len(keypoints) and ei < len(keypoints):
            sx, sy = keypoints[si]
            ex, ey = keypoints[ei]
            cv2.line(vis, (int(sx), int(sy)), (int(ex), int(ey)),
                     (200, 200, 200), 1)

    # 手指连线 + 到手腕
    for fi, (start, end) in enumerate(FINGER_RANGES):
        color = FINGER_COLORS[fi]
        # 手腕到手指根部
        if len(keypoints) > start:
            wx, wy = keypoints[0]
            rx, ry = keypoints[start]
            cv2.line(vis, (int(wx), int(wy)), (int(rx), int(ry)), color, 2)
        # 手指内部
        for i in range(start, min(end - 1, len(keypoints) - 1)):
            sx, sy = keypoints[i]
            ex, ey = keypoints[i + 1]
            cv2.line(vis, (int(sx), int(sy)), (int(ex), int(ey)), color, 2)

    # 关键点
    for px, py in keypoints:
        cv2.circle(vis, (int(px), int(py)), 3, (0, 0, 255), -1)

    # 手腕 — 大绿点
    if keypoints:
        cv2.circle(vis, (int(keypoints[0][0]), int(keypoints[0][1])),
                   6, (0, 255, 0), -1)

    # BBox
    if bbox is not None:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 0, 0), 2)

    return vis


def main():
    if len(sys.argv) < 2:
        print("用法: python scripts/run_wilor_preview.py <image_path>")
        sys.exit(1)

    image_path = Path(sys.argv[1])
    if not image_path.is_file():
        print(f"图片不存在: {image_path}")
        sys.exit(1)

    print(f"图片: {image_path}")
    frame_bgr = cv2.imread(str(image_path))
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    print(f"尺寸: {w}×{h}")

    # ---- 初始化 WiLoR ----
    print("\n初始化 WiLoR...")
    config = WiLoRConfig(
        checkpoint_path="e:/ZSPD/WiLoR/pretrained_models/wilor_final.ckpt",
        expected_sha256="",
        device="cuda",
        model_version="wilor_cvpr2025",
        wilor_source_path="e:/ZSPD/WiLoR",
        detector_path="e:/ZSPD/WiLoR/pretrained_models/detector.pt",
        model_config_path="e:/ZSPD/WiLoR/pretrained_models/model_config.yaml",
    )

    backend = WiLoRBackend(config)
    adapter = WiLoRAdapter(backend)
    print(f"  设备: {backend.device}")
    print(f"  模型版本: {backend.model_info.model_version}")
    print(f"  checkpoint SHA-256: {backend.model_info.checkpoint_sha256[:16]}...")

    # ---- 推理 ----
    print("\n执行推理...")
    detections = adapter.detect(frame_rgb, timestamp_ms=0)
    print(f"  检测到 {len(detections)} 只手")

    # ---- 映射 + 绘制 ----
    if is_mapping_ready(WILOR_TO_HANDS_V1_V1):
        print(f"  映射: {len(WILOR_TO_HANDS_V1_V1)} 点 (wilor-to-hands-v1-v1)")
    else:
        print("  ⚠ 映射未就绪！")

    vis_frame = frame_rgb.copy()

    for i, det in enumerate(detections):
        print(f"\n--- Hand {i}: {det.handedness} (score={det.handedness_score:.2f}) ---")
        print(f"  BBox: ({det.bbox_xyxy_px[0]:.0f}, {det.bbox_xyxy_px[1]:.0f}) → "
              f"({det.bbox_xyxy_px[2]:.0f}, {det.bbox_xyxy_px[3]:.0f})")
        print(f"  clipped: {det.clipped}")

        # 绘制 BBox
        x1, y1, x2, y2 = det.bbox_xyxy_px
        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)),
                      (255, 0, 0), 2)
        cv2.putText(vis_frame, f"{det.handedness} {det.detection_score:.2f}",
                    (int(x1), int(y1) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        # 尝试 21 点映射
        raw_result = convert_wilor_to_raw_hand_result(
            det,
            mapping=WILOR_TO_HANDS_V1_V1,
            image_width=w, image_height=h,
        )

        if raw_result is not None:
            kps = raw_result.keypoints.pixel
            print(f"  关键点: {len(kps)} 个")
            print(f"  clipped: {raw_result.keypoints.any_clipped} "
                  f"({raw_result.keypoints.clipped_count} 个)")
            # 打印手腕和指尖
            tips = [0, 4, 8, 12, 16, 20]
            tip_names = ["手腕", "拇指尖", "食指尖", "中指尖", "无名指尖", "小指尖"]
            for idx, name in zip(tips, tip_names):
                x, y = kps[idx]
                print(f"    {name}: ({x:.0f}, {y:.0f})")

            # 绘制骨架
            vis_frame = draw_skeleton(vis_frame, kps, det.bbox_xyxy_px)
        else:
            print("  ⚠ 21 点转换失败 — 映射未就绪或关节数不匹配")

    # ---- 保存 ----
    output_dir = Path("output/hands")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "wilor_preview.jpg"
    cv2.imwrite(str(output_path), cv2.cvtColor(vis_frame, cv2.COLOR_RGB2BGR))
    print(f"\n预览图: {output_path}")

    backend.close()
    print("完成。")


if __name__ == "__main__":
    main()
