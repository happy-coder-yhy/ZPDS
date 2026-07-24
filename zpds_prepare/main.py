"""
ZPDS Prepare — 主入口。

从原始采集数据出发，运行检测器、生成统一 QualityIssue、
决定 trim/split/keep_with_flag、产出候选 Segment。

用法:
    # 墨现 (默认)
    python -m zpds_prepare.main /path/to/dataset/
    python -m zpds_prepare.main /path/to/dataset/ --profile guida

    # 遁甲
    python -m zpds_prepare.main /path/to/session.mcap --profile dunjia
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from zpds_prepare.detectors.black_frame import detect_black_frames
from zpds_prepare.detectors.timestamp_gap import detect_timestamp_gaps
from zpds_prepare.detectors.imu_gap import detect_imu_gaps
from zpds_prepare.detectors.frame_count import detect_frame_count_mismatch
from zpds_prepare.detectors.bad_frame import detect_bad_frames
from zpds_prepare.decisions.segment_planner import (
    plan_segments,
    get_issue_summary,
)
from zpds_prepare.writers.quality_writer import write_quality_issues
from zpds_prepare.writers.candidate_writer import write_segment_candidates


CONFIG_PATH = "config.yaml"
OUTPUT_DIR = "output"


def _get_reader(profile: str) -> tuple[Any, float, float, str]:
    """返回 (reader_module, expected_video_fps, expected_imu_hz, get_video_fn_name)。

    各 profile 的 reader 模块导出相同的函数签名：
      - read_meta(path) -> dict
      - read_index_frames(path) -> list[dict]
      - read_index_timestamps(path) -> list[int]
      - read_imu(path) -> pd.DataFrame
      - get_color_*  (path) -> str   # 视频文件路径
      - get_session_id(path) -> str
    """
    if profile == "guida":
        from zpds_prepare.readers import guida_reader as rd
        return rd, 30.0, 50.0, "get_color_mkv"
    elif profile == "dunjia":
        from zpds_prepare.readers import dunjia_reader as rd
        return rd, 25.0, 196.0, "get_color_video"
    else:
        raise ValueError(f"未知 profile: {profile}，可选: guida, dunjia")


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
        default=None,
        help="数据集路径 (墨现: 目录; 遁甲: .mcap 文件)",
    )
    parser.add_argument(
        "--profile", "-p",
        default="guida",
        choices=["guida", "dunjia"],
        help="数据源 profile (默认: guida)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录 (默认: output/moxian/ 或 output/dunjia/)",
    )
    parser.add_argument(
        "--config", "-c",
        default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    args = parser.parse_args()

    # ---- 解析 profile 和 reader ----
    profile = args.profile
    rd, EXPECTED_VIDEO_FPS, EXPECTED_IMU_HZ, get_video_fn = _get_reader(profile)
    get_video = getattr(rd, get_video_fn)

    # ---- 默认数据集路径 ----
    if args.dataset is None:
        if profile == "dunjia":
            parser.error("遁甲模式必须指定 .mcap 文件路径")
        else:
            # 保持与旧版的兼容默认值
            dataset_path = "E:/datasets/egos/墨现"
    else:
        dataset_path = args.dataset

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        subdir = "moxian" if profile == "guida" else "dunjia"
        output_dir = Path("output") / subdir
    else:
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

    print(f"  Profile:     {profile}")
    print(f"  数据集:      {dataset_path}")
    meta = rd.read_meta(dataset_path)
    print(f"  设备:        {meta['device']}")
    print(f"  标称帧率:    {meta['fps']} fps")
    print(f"  分辨率:      {meta['width']}×{meta['height']}")
    print(f"  标称帧数:    {meta['frame_count']}")

    index_frames = rd.read_index_frames(dataset_path)
    timestamps_ns = [f["timestamp_ns"] for f in index_frames]
    print(f"  Index 帧数:  {len(timestamps_ns)}")

    if len(timestamps_ns) >= 2:
        duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        median_interval_ns = int(np.median(np.diff(timestamps_ns)))
        print(f"  时长:        {duration_s:.2f} s")
        print(f"  帧间隔中位数: {median_interval_ns:,} ns (~{1e9/median_interval_ns:.1f} fps)")

    session_start_ns = timestamps_ns[0]
    session_end_ns = timestamps_ns[-1]
    session_id = rd.get_session_id(dataset_path)
    print(f"  Session ID:  {session_id}")

    # 遁甲：显示双时间戳信息
    if profile == "dunjia" and index_frames:
        first = index_frames[0]
        print(f"  时间戳 (消息内):   {first['timestamp_ns']}")
        print(f"  log_time (MCAP):   {first.get('log_time_ns', 'N/A')}")
        print(f"  publish_time:      {first.get('publish_time_ns', 'N/A')}")

    imu = rd.read_imu(dataset_path)
    print(f"  IMU 行数:    {len(imu)}")

    # ================================================================
    # Step 2: 运行检测器
    # ================================================================
    step_header(2, "运行检测器")

    all_issues = []

    # 2a. 帧数一致性检查
    print("\n  [2a] 帧数一致性检查...")
    fc_issues = detect_frame_count_mismatch(
        index_frame_count=len(timestamps_ns),
        meta_frame_count=meta["frame_count"],
        timestamps_ns=timestamps_ns,
        stream_id="ego_rgb",
    )
    all_issues.extend(fc_issues)
    print(f"    发现 {len(fc_issues)} 个不一致")
    for iss in fc_issues:
        print(f"      [{iss.decision}] Index={iss.details.get('index_frame_count')} vs "
              f"Meta={iss.details.get('meta_frame_count')}, diff={iss.details.get('difference')}")

    # 2b. 获取视频路径 (遁甲: 重构 .h264; 墨现: 直接读 .mkv)
    color_video = get_video(dataset_path)

    # 2c. 坏帧检测
    print("\n  [2b] 坏帧检测...")
    bad_issues = detect_bad_frames(
        video_path=color_video,
        timestamps_ns=timestamps_ns,
        stream_id="ego_rgb",
    )
    all_issues.extend(bad_issues)
    print(f"    发现 {len(bad_issues)} 个坏帧区间")
    for iss in bad_issues:
        print(f"      [{iss.decision}] {iss.details.get('bad_frame_count')} 帧 "
              f"({iss.details.get('bad_ratio', 0)*100:.1f}%)")

    # 2d. 黑屏检测
    print("\n  [2c] 黑屏检测...")
    min_black_duration_ns = int(min_black_duration_s * 1_000_000_000)
    edge_tolerance_ns = int(edge_tolerance_s * 1_000_000_000)

    black_issues = detect_black_frames(
        video_path=color_video,
        timestamps_ns=timestamps_ns,
        mean_intensity_threshold=black_threshold,
        min_duration_ns=min_black_duration_ns,
        edge_tolerance_ns=edge_tolerance_ns,
    )
    all_issues.extend(black_issues)
    print(f"    发现 {len(black_issues)} 个黑屏区间")
    for iss in black_issues:
        print(f"      [{iss.decision}] {iss.start_ns:,} → {iss.end_ns:,} "
              f"({(iss.end_ns - iss.start_ns)/1e9:.2f}s, "
              f"{iss.details.get('frame_count', '?')} 帧)")

    # 2e. 视频时间戳缺口检测
    print("\n  [2d] 视频时间戳缺口检测...")
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

    # 2f. IMU 时间戳缺口检测
    print("\n  [2e] IMU 时间戳缺口检测...")
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
