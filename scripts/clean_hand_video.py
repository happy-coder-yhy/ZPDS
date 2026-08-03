#!/usr/bin/env python3
"""从视频和既有 hands_2d.parquet 生成可追溯的手部清洗产物。"""

from __future__ import annotations

import argparse
import json
import sys

from zpds.hands.cleaning import HandVideoCleaningConfig, clean_hand_video


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, help="输入视频，只读")
    parser.add_argument("--hands", required=True, help="已有 hands_2d.parquet")
    parser.add_argument(
        "--frame-status",
        help="可选全帧推理状态 Parquet，用于区分 no_hand 与模型失败",
    )
    parser.add_argument("--output-dir", required=True, help="派生清洗产物目录")
    parser.add_argument(
        "--config",
        default="configs/hands/cleaning_default.yaml",
        help="手部视频清洗 YAML 配置",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = HandVideoCleaningConfig.load(args.config)
        result = clean_hand_video(
            args.video,
            args.hands,
            args.output_dir,
            config,
            frame_status_path=args.frame_status,
        )
    except (FileNotFoundError, TypeError, ValueError, OSError) as error:
        print(f"hand video cleaning failed: {error}", file=sys.stderr)
        return 2
    summary = {
        "report": str(result.report_path),
        "cleaned_video": str(result.cleaned_video_path) if result.cleaned_video_path else None,
        "frame_metrics": str(result.frame_metrics_path),
        "sample_map": str(result.sample_map_path),
        "summary": result.report["summary"],
        "quality_metrics": result.report["quality_metrics"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
