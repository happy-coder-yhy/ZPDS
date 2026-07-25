"""
A2D Segment 候选生成 — CLI 入口。

读取 A2D Episode 目录，运行全部机器人质量检测器，
计算公共有效范围，生成 segment_candidates.json。

用法:
    python scripts/segment_a2d.py E:/datasets/真机/A2D/8032/
    python scripts/segment_a2d.py E:/datasets/真机/A2D/8032/ -o output/a2d/8032/
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from zpds_prepare.readers.a2d_reader import read_session
from zpds_prepare.segmentation.a2d_segmenter import run_and_write


def main():
    parser = argparse.ArgumentParser(
        description="A2D Segment 候选生成 — 质量检测 + 分段规划"
    )
    parser.add_argument(
        "episode",
        help="A2D Episode 根目录路径",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录 (默认: output/a2d/{episode_id}/)",
    )
    parser.add_argument(
        "--min-duration", "-m",
        type=float,
        default=1.0,
        help="最短有效 Segment 秒数 (默认: 1.0)",
    )
    parser.add_argument(
        "--max-duration", "-M",
        type=float,
        default=300.0,
        help="最长有效 Segment 秒数 (默认: 300.0)",
    )
    parser.add_argument(
        "--alignment-dir",
        default=None,
        help="camera_robot_alignment.parquet 所在 maps/ 目录 (默认自动推测)",
    )
    args = parser.parse_args()

    episode_root = Path(args.episode)
    if not episode_root.is_dir():
        print(f"错误: 目录不存在: {episode_root}", file=sys.stderr)
        return 1

    # ---- 自动推断输出目录 ----
    if args.output is None:
        # 从 episode 路径提取 ID (如 8032)
        ep_id = episode_root.name
        output_dir = Path(f"output/a2d/{ep_id}")
    else:
        output_dir = Path(args.output)

    # ---- 自动推断 alignment 目录 ----
    alignment_dir = None
    if args.alignment_dir:
        alignment_dir = Path(args.alignment_dir)
    else:
        # 默认: output/a2d/{ep_id}/prepared/seg_000001/maps/
        ep_id = episode_root.name
        default_alignment = Path(f"output/a2d/{ep_id}/prepared/seg_000001/maps")
        if default_alignment.is_dir():
            alignment_dir = default_alignment

    # ---- 配置 ----
    config = {
        "min_duration_s": args.min_duration,
        "max_duration_s": args.max_duration,
    }

    # ---- 执行 ----
    start_time = time.time()

    print(f"读取 A2D Episode: {episode_root}")
    session = read_session(episode_root)

    print(f"  Session ID:  {session.session_id}")
    print(f"  视频流:      {list(session.video_streams.keys())}")
    print(f"  时序流:      {list(session.time_series_streams.keys())}")
    print(f"  相机帧数:    {session.meta.get('camera_frame_count', '?')}")
    print(f"  aligned 样本: {session.meta.get('aligned_samples', '?')}")
    print(f"  输出目录:    {output_dir.resolve()}")

    sc_path = run_and_write(
        session=session,
        output_dir=output_dir,
        config=config,
        alignment_dir=alignment_dir,
    )

    elapsed = time.time() - start_time
    print(f"\n完成，耗时 {elapsed:.1f}s")
    print(f"输出: {sc_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
