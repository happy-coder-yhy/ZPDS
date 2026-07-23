"""
ZPDS Prepare — 主入口。

从原始采集数据出发，运行检测器、生成统一 QualityIssue、
决定 trim/split/keep_with_flag、产出候选 Segment。

用法:
    python -m zpds_prepare.main E:/datasets/egos/墨现
    python -m zpds_prepare.main E:/datasets/egos/墨现 --output output/
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from zpds_prepare.readers.guida_reader import (
    read_meta,
    read_index_frames,
    read_index_timestamps,
    read_imu,
    get_color_mkv,
    get_session_id,
)
from zpds_prepare.detectors.black_frame import detect_black_frames
from zpds_prepare.detectors.timestamp_gap import detect_timestamp_gaps
from zpds_prepare.detectors.imu_gap import detect_imu_gaps
from zpds_prepare.decisions.segment_planner import (
    plan_segments,
    get_issue_summary,
)
from zpds_prepare.writers.quality_writer import write_quality_issues
from zpds_prepare.writers.candidate_writer import write_segment_candidates


CONFIG_PATH = "config.yaml"
OUTPUT_DIR = "output"
EXPECTED_VIDEO_FPS = 30.0
EXPECTED_IMU_HZ = 50.0


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="ZPDS Prepare — 从原始数据生成质量报告和候选分段方案"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default="E:/datasets/egos/墨现",
        help="数据集根目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        help="输出目录 (默认 output/)",
    )
    parser.add_argument(
        "--config", "-c",
        default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    args = parser.parse_args()

    dataset_path = args.dataset
    output_dir = Path(args.output)
    config_path = args.config

    start_time = time.time()

    # ================================================================
    # Step 0: 加载配置
    # ================================================================
    cfg = load_config(config_path)

    # 黑屏检测参数
    bd = cfg.get("video", {}).get("black_detection", {})
    black_threshold = bd.get(
        "mean_intensity_threshold",
        cfg.get("video", {}).get("black_threshold", 5.0),
    )
    min_black_duration_s = bd.get(
        "min_duration_s",
        cfg.get("video", {}).get("min_black_duration_s", 0.5),
    )
    edge_tolerance_s = bd.get("edge_tolerance_s", 1.0)

    # 视频时间戳缺口参数
    tv = cfg.get("timestamp", {}).get("video", {})
    video_gap_factor = tv.get(
        "gap_factor",
        cfg.get("timestamp", {}).get("video_gap_factor", 2.0),
    )
    video_split_gap_s = tv.get("split_gap_s", 0.5)

    # IMU 时间戳缺口参数
    ti = cfg.get("timestamp", {}).get("imu", {})
    imu_gap_factor = ti.get(
        "gap_factor",
        cfg.get("timestamp", {}).get("imu_gap_factor", 3.0),
    )
    imu_split_gap_s = ti.get("split_gap_s", 1.0)

    # Segment 约束参数
    seg = cfg.get("segment", {})
    min_duration_s = seg.get("min_duration_s", 1.0)
    max_duration_s = seg.get("max_duration_s", 120.0)

    # ================================================================
    # Step 1: 读取数据
    # ================================================================
    step_header(1, "读取原始数据")

    print(f"  数据集: {dataset_path}")
    meta = read_meta(dataset_path)
    print(f"  设备:       {meta['device']}")
    print(f"  标称帧率:   {meta['fps']} fps")
    print(f"  分辨率:     {meta['width']}×{meta['height']}")
    print(f"  标称帧数:   {meta['frame_count']}")

    index_frames = read_index_frames(dataset_path)
    timestamps_ns = [f["timestamp_ns"] for f in index_frames]
    print(f"  Index 帧数: {len(timestamps_ns)}")

    if len(timestamps_ns) >= 2:
        duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        median_interval_ns = int(np.median(np.diff(timestamps_ns)))
        print(f"  时长:       {duration_s:.2f} s")
        print(f"  帧间隔中位数: {median_interval_ns:,} ns (~{1e9/median_interval_ns:.1f} fps)")

    session_start_ns = timestamps_ns[0]
    session_end_ns = timestamps_ns[-1]
    session_id = get_session_id(dataset_path)
    print(f"  Session ID: {session_id}")

    imu = read_imu(dataset_path)
    print(f"  IMU 行数:   {len(imu)}")

    # ================================================================
    # Step 2: 运行检测器
    # ================================================================
    step_header(2, "运行检测器")

    all_issues = []

    # 2a. 黑屏检测
    print("\n  [2a] 黑屏检测...")
    color_mkv = get_color_mkv(dataset_path)
    min_black_duration_ns = int(min_black_duration_s * 1_000_000_000)
    edge_tolerance_ns = int(edge_tolerance_s * 1_000_000_000)

    black_issues = detect_black_frames(
        video_path=color_mkv,
        timestamps_ns=timestamps_ns,
        mean_intensity_threshold=black_threshold,
        min_duration_ns=min_black_duration_ns,
        edge_tolerance_ns=edge_tolerance_ns,
    )
    all_issues.extend(black_issues)
    print(f"    发现 {len(black_issues)} 个黑屏区间")
    for iss in black_issues:
        print(f"      [{iss.decision}] {iss.start_ns:,} → {iss.end_ns:,} "
              f"({(iss.end_ns - iss.start_ns)/1e9:.2f}s, {iss.details.get('frame_count', '?')} 帧)")

    # 2b. 视频时间戳缺口检测
    print("\n  [2b] 视频时间戳缺口检测...")
    expected_video_interval_ns = int(1_000_000_000 / EXPECTED_VIDEO_FPS)
    video_split_gap_ns = int(video_split_gap_s * 1_000_000_000)

    video_issues = detect_timestamp_gaps(
        timestamps_ns=timestamps_ns,
        expected_interval_ns=expected_video_interval_ns,
        gap_factor=video_gap_factor,
        split_gap_ns=video_split_gap_ns,
        stream_id="ego_rgb",
    )
    all_issues.extend(video_issues)
    print(f"    发现 {len(video_issues)} 个时间戳缺口")
    for iss in video_issues:
        print(f"      [{iss.decision}] Frame {iss.details.get('frame_index', '?')}: "
              f"gap={iss.details.get('gap_ms', '?')}ms, "
              f"est. missing={iss.details.get('estimated_missing_frames', '?')} 帧")

    # 2c. IMU 时间戳缺口检测
    print("\n  [2c] IMU 时间戳缺口检测...")
    expected_imu_interval_ns = int(1_000_000_000 / EXPECTED_IMU_HZ)
    imu_split_gap_ns = int(imu_split_gap_s * 1_000_000_000)

    imu_issues = detect_imu_gaps(
        imu=imu,
        expected_interval_ns=expected_imu_interval_ns,
        gap_factor=imu_gap_factor,
        split_gap_ns=imu_split_gap_ns,
        stream_id="ego_imu",
    )
    all_issues.extend(imu_issues)
    print(f"    发现 {len(imu_issues)} 个 IMU 缺口")
    for iss in imu_issues:
        print(f"      [{iss.decision}] Sample {iss.details.get('sample_index', '?')}: "
              f"gap={iss.details.get('gap_s', '?')}s, "
              f"est. missing={iss.details.get('estimated_missing_samples', '?')} 样本")

    # ================================================================
    # Step 3: 汇总分析
    # ================================================================
    step_header(3, "汇总分析")

    summary = get_issue_summary(all_issues)
    print(f"  总异常数: {summary['total']}")
    if summary["total"] > 0:
        print(f"  按类型: {summary['by_type']}")
        print(f"  按处置: {summary['by_decision']}")

    # ================================================================
    # Step 4: 写出 quality_issues.json
    # ================================================================
    step_header(4, "写出 quality_issues.json")

    qi_path = write_quality_issues(
        output_path=output_dir / "quality_issues.json",
        issues=all_issues,
        source_session_id=session_id,
    )
    print(f"  输出: {qi_path.resolve()}")

    # ================================================================
    # Step 5: 生成候选 Segment
    # ================================================================
    step_header(5, "生成候选 Segment")

    candidates = plan_segments(
        issues=all_issues,
        session_start_ns=session_start_ns,
        session_end_ns=session_end_ns,
        min_duration_ns=int(min_duration_s * 1_000_000_000),
        max_duration_ns=int(max_duration_s * 1_000_000_000),
    )

    print(f"  候选数: {len(candidates)}")
    for c in candidates:
        print(f"    {c.candidate_id}: "
              f"{c.source_start_ns:,} → {c.source_end_ns:,} "
              f"({c.duration_ns / 1e9:.2f}s, {c.reason})")
        for iss in c.issues_in_span:
            print(f"      ⚠ [{iss['decision']}] {iss['issue_type']}: "
                  f"{(iss['end_ns'] - iss['start_ns']) / 1e9:.2f}s")

    # ================================================================
    # Step 6: 写出 segment_candidates.json
    # ================================================================
    step_header(6, "写出 segment_candidates.json")

    sc_path = write_segment_candidates(
        output_path=output_dir / "segment_candidates.json",
        candidates=candidates,
        source_session_id=session_id,
        source_start_ns=session_start_ns,
        source_end_ns=session_end_ns,
    )
    print(f"  输出: {sc_path.resolve()}")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  完成")
    print(f"  耗时:        {elapsed:.1f}s")
    print(f"  发现异常:    {summary['total']}")
    print(f"  候选 Segment: {len(candidates)}")
    print(f"  输出目录:    {output_dir.resolve()}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
