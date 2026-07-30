"""WiLoR Pilot 测试运行器（阶段 6）。

对一批测试图片同时运行 WiLoR + MediaPipe（可选对照），
逐帧输出诊断 JSON + 完整 Run Report。

用途：
    python -m zpds.hands.wilor_pilot --images ./pilot_images/ --output ./pilot_output/

诊断输出每帧一行 JSON，包含：
    - 帧索引、时间戳
    - WiLoR 状态 / 手数 / BBox / 左右手 / 耗时
    - MediaPipe 对照（如启用）
    - 关键点数量 / clipped / failure_reason
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np

from zpds.hands.schemas import HandFrameResult
from zpds.hands.wilor_schema import WiLoRRunReport


def run_pilot(
    *,
    image_paths: list[Path],
    estimator,  # WiLoRHandEstimator
    mediapipe_estimator=None,  # MediaPipeHandEstimator | None (对照)
    output_dir: str | Path = "./pilot_output",
    print_diagnostics: bool = True,
) -> WiLoRRunReport:
    """对一批图片运行 Pilot 测试，输出逐帧诊断和 Run Report。

    Args:
        image_paths: 测试图片路径列表（20~50 帧）。
        estimator: WiLoRHandEstimator 实例。
        mediapipe_estimator: 可选 MediaPipe 对照估计器。
        output_dir: 输出目录。
        print_diagnostics: 是否打印每帧诊断到 stdout。

    Returns:
        WiLoRRunReport。
    """
    import cv2

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    diagnostics_path = output / "pilot_diagnostics.jsonl"
    report_path = output / "pilot_run_report.json"

    diagnostics_lines: list[dict] = []

    for idx, img_path in enumerate(image_paths):
        frame_bgr = cv2.imread(str(img_path))
        if frame_bgr is None:
            diag = {
                "frame_index": idx,
                "image_path": str(img_path),
                "error": "failed_to_read",
            }
            diagnostics_lines.append(diag)
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        timestamp_ms = idx * 33  # 假设 30fps

        diag: dict = {
            "frame_index": idx,
            "image_path": str(img_path),
            "image_size": [w, h],
            "timestamp_ms": timestamp_ms,
        }

        # ---- WiLoR ----
        t_wilor = time.perf_counter()
        frame_result: HandFrameResult | None = None
        try:
            frame_result = estimator.estimate_frame(frame_rgb, timestamp_ms)
        except Exception as exc:
            diag["wilor_error"] = f"{type(exc).__name__}: {exc}"

        wilor_ms = (time.perf_counter() - t_wilor) * 1000

        if frame_result is not None:
            p = frame_result.primary
            diag["wilor_status"] = p.status
            diag["wilor_hand_count"] = len(p.hands)
            diag["wilor_inference_ms"] = p.inference_ms
            diag["wilor_failure_reason"] = p.failure_reason
            diag["wilor_model"] = p.model_name
            diag["fallback_attempted"] = frame_result.fallback_attempted
            diag["fallback_used"] = frame_result.fallback_used
            diag["effective_model"] = frame_result.effective_model
            diag["effective_hand_count"] = len(frame_result.effective_hands)

        diag["wilor_total_ms"] = round(wilor_ms, 2)

        # ---- MediaPipe 对照 ----
        if mediapipe_estimator is not None:
            t_mp = time.perf_counter()
            try:
                mp_results = mediapipe_estimator.estimate(frame_rgb, timestamp_ms)
                mp_ms = (time.perf_counter() - t_mp) * 1000
                diag["mediapipe_hands"] = len(mp_results)
                diag["mediapipe_inference_ms"] = round(mp_ms, 2)
            except Exception as exc:
                diag["mediapipe_error"] = f"{type(exc).__name__}: {exc}"

        diagnostics_lines.append(diag)

        if print_diagnostics:
            print(json.dumps(diag, ensure_ascii=False, default=str))

    # 写入诊断文件
    with open(diagnostics_path, "w", encoding="utf-8") as f:
        for line in diagnostics_lines:
            f.write(json.dumps(line, ensure_ascii=False, default=str) + "\n")

    # 生成 Run Report
    report = estimator.build_run_report()
    report.coverage["decoded_frames"] = len([
        d for d in diagnostics_lines if "error" not in d
    ])

    # 写入报告
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    print(f"\nPilot diagnostics: {diagnostics_path}")
    print(f"Run report: {report_path}")

    return report


__all__ = ["run_pilot"]
