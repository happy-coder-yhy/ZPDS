"""
ZPDS Prepared Segment 生成 — 主入口点。

将墨现 Guida 原始数据转换为标准化 Prepared Segment：

    ① 确定有效时间区间（Span）
    ② 裁剪并转码 RGB 视频
    ③ 生成 rgb_sample_map.parquet
    ④ 规范化 IMU
    ⑤ 提取 calibration.json
    ⑥ 生成 segment.json
    ⑦ 写出后验证

用法:
    python prepare_segment.py
"""

import json
import time
import yaml
from pathlib import Path

import numpy as np

from segment.span_determiner import (
    load_config, load_index, determine_span,
)
from segment.video_transcoder import transcode_rgb
from segment.sample_map import generate_sample_map, write_sample_map
from segment.imu_normalizer import normalize_imu, write_imu
from segment.calibration import extract_calibration, write_calibration
from segment.segment_writer import build_segment_json, write_segment_json
from segment.validator import validate_segment, write_validation_report

# ---- QC 模块（使用新版统一检测器） ----
from zpds_prepare.detectors.black_frame import detect_black_frames

# ============================================================
# 配置
# ============================================================
DATASET = "E:/datasets/egos/墨现"
OUTPUT_DIR = "prepared_segments/seg_000001"
CONFIG_PATH = "config.yaml"
REVISION = "r0001"

# ============================================================
# 辅助：打印分隔
# ============================================================
def step_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 60}")


def main():
    start_time = time.time()

    # ---- 加载配置 ----
    cfg = load_config(CONFIG_PATH)
    target_fps = cfg["output"]["target_fps"]

    # ================================================================
    # Step 1: 确定有效时间区间
    # ================================================================
    step_header(1, "确定有效时间区间 (Span)")

    # 运行黑屏检测（使用新版统一检测器）
    color_mkv = str(Path(DATASET) / "color_000000.mkv")
    index_frames_all = load_index(DATASET)
    timestamps_all = [f["timestamp_ns"] for f in index_frames_all]
    black_issues = detect_black_frames(
        video_path=color_mkv,
        timestamps_ns=timestamps_all,
        mean_intensity_threshold=cfg["video"].get("black_detection", {}).get(
            "mean_intensity_threshold", cfg["video"].get("black_threshold", 5.0)
        ),
        min_duration_ns=int(cfg["video"].get("black_detection", {}).get(
            "min_duration_s", cfg["video"].get("min_black_duration_s", 0.5)
        ) * 1_000_000_000),
        edge_tolerance_ns=int(cfg["video"].get("black_detection", {}).get(
            "edge_tolerance_s", 1.0
        ) * 1_000_000_000),
    )

    span = determine_span(
        dataset_path=DATASET,
        black_frame_indices=None,
        black_issues=black_issues if black_issues else None,
        imu_gap_samples=None,   # 当前样本无 IMU 中断
        timestamp_gaps=None,    # 当前样本无时间戳跳变
        config_path=CONFIG_PATH,
    )

    print(f"  源起始:   {span['source_start_ns']:>20,} ns")
    print(f"  源结束:   {span['source_end_ns']:>20,} ns")
    print(f"  时长:     {span['duration_s']:.2f} s")
    print(f"  Span 帧数: {span['total_frames_in_span']}")
    print(f"  头部裁剪: {span['trimmed_head_frames']} 帧 ({span['reason']['start']})")
    print(f"  尾部裁剪: {span['trimmed_tail_frames']} 帧 ({span['reason']['end']})")

    # ================================================================
    # Step 2: 裁剪并转码 RGB 视频
    # ================================================================
    step_header(2, "裁剪转码 RGB 视频")

    index_frames = load_index(DATASET)
    source_mkv = str(Path(DATASET) / "color_000000.mkv")
    output_mp4 = str(Path(OUTPUT_DIR) / "data" / "ego_rgb.mp4")

    video_result = transcode_rgb(
        source_mkv=source_mkv,
        output_mp4=output_mp4,
        source_start_ns=span["source_start_ns"],
        source_end_ns=span["source_end_ns"],
        index_frames=index_frames,
        target_fps=target_fps,
    )
    print(f"  输出帧数:  {video_result['output_frames']}")
    print(f"  编码:      {video_result['codec']}")
    print(f"  分辨率:    {video_result['width']}×{video_result['height']}")
    print(f"  输出文件:  {video_result['output_path']}")

    # ================================================================
    # Step 3: 生成 rgb_sample_map.parquet
    # ================================================================
    step_header(3, "生成采样映射表 sample_map")

    sample_map = generate_sample_map(
        index_frames=index_frames,
        source_start_ns=span["source_start_ns"],
        source_end_ns=span["source_end_ns"],
        target_fps=target_fps,
    )
    sm_path = write_sample_map(sample_map, OUTPUT_DIR)
    print(f"  输出行数:  {len(sample_map)}")
    print(f"  映射方法:  nearest")
    print(f"  最大误差:  {sample_map['time_error_ns'].abs().max():,} ns")
    print(f"  输出文件:  {sm_path}")

    # ================================================================
    # Step 4: 规范化 IMU
    # ================================================================
    step_header(4, "规范化 IMU 数据")

    imu_path = str(Path(DATASET) / "imu" / "imu_000000.csv")
    imu = normalize_imu(
        imu_path=imu_path,
        source_start_ns=span["source_start_ns"],
        source_end_ns=span["source_end_ns"],
    )
    imu_out = write_imu(imu, OUTPUT_DIR)
    print(f"  输出行数:  {len(imu)}")
    print(f"  列名:      {list(imu.columns)}")
    print(f"  单位:      m/s^2 (accel), rad/s (gyro)")
    print(f"  输出文件:  {imu_out}")

    # ================================================================
    # Step 5: 提取 calibration.json
    # ================================================================
    step_header(5, "提取标定信息")

    meta_path = str(Path(DATASET) / "meta.json")
    calib = extract_calibration(meta_path)
    calib_path = write_calibration(calib, OUTPUT_DIR)
    print(f"  标定 ID:   {calib['calibration_id']}")
    print(f"  RGB 内参:  fx={calib['cameras'][0]['intrinsics']['fx']:.2f}, "
          f"fy={calib['cameras'][0]['intrinsics']['fy']:.2f}")
    print(f"  输出文件:  {calib_path}")

    # ================================================================
    # Step 6: 生成 segment.json
    # ================================================================
    step_header(6, "生成 segment.json")

    segment = build_segment_json(
        dataset_path=DATASET,
        span=span,
        video_result=video_result,
        sample_map_rows=len(sample_map),
        imu_rows=len(imu),
        revision=REVISION,
    )
    seg_path = write_segment_json(segment, OUTPUT_DIR)
    print(f"  Segment ID: {segment['segment_id']}")
    print(f"  Streams:    {[s['stream_id'] for s in segment['streams']]}")
    print(f"  输出文件:   {seg_path}")

    # ================================================================
    # Step 7: 写出后验证
    # ================================================================
    step_header(7, "写出后验证")

    validation = validate_segment(OUTPUT_DIR)
    val_path = write_validation_report(validation, OUTPUT_DIR)

    print(f"\n  状态: {validation['status'].upper()}")
    for check_name, result in validation["checks"].items():
        icon = "✓" if result == "pass" else ("⚠" if result == "warn" else "✗")
        print(f"  {icon} {check_name}: {result}")

    if validation["statistics"]:
        print(f"\n  统计:")
        for k, v in validation["statistics"].items():
            print(f"    {k}: {v:,}" if isinstance(v, int) else f"    {k}: {v}")

    if validation["errors"]:
        print(f"\n  错误:")
        for e in validation["errors"]:
            print(f"    ✗ {e}")

    print(f"\n  验证报告: {val_path}")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  Prepared Segment 生成{'完成' if validation['status'] == 'pass' else '失败'}")
    print(f"  耗时: {elapsed:.1f}s")
    print(f"  输出: {Path(OUTPUT_DIR).resolve()}")
    print(f"{'=' * 60}")

    return validation["status"] == "pass"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
