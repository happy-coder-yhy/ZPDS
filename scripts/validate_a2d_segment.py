"""
A2D Prepared Segment 验证 — CLI 入口。

对已生成的 Prepared Segment 执行完整性验证。

用法:
    python scripts/validate_a2d_segment.py output/a2d/8032/prepared_segments/seg_000001/
    python scripts/validate_a2d_segment.py output/a2d/8032/prepared_segments/  # 验证全部 segment
    python scripts/validate_a2d_segment.py output/a2d/8032/                     # 自动发现 prepared_segments/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from segment.a2d_validator import validate_segment, write_validation_report


def main():
    parser = argparse.ArgumentParser(
        description="A2D Prepared Segment 验证 — 检查视频/时序/对齐完整性"
    )
    parser.add_argument(
        "source",
        help="segment 目录路径，或 prepared_segments 父目录（验证全部 segment）",
    )
    parser.add_argument(
        "--output-report", "-o",
        default=None,
        help="汇总报告输出路径（JSON），仅在验证多个 segment 时有效",
    )
    args = parser.parse_args()

    source_path = Path(args.source)

    # ---- 发现要验证的 segment 目录 ----
    if (source_path / "segment.json").is_file():
        # 单个 segment 目录
        segment_dirs = [source_path]
    elif (source_path / "prepared_segments").is_dir():
        # output 目录 → 进入 prepared_segments/
        segment_dirs = sorted(
            d for d in (source_path / "prepared_segments").iterdir()
            if d.is_dir() and (d / "segment.json").is_file()
        )
    elif source_path.is_dir():
        # prepared_segments 父目录 → 扫描所有 seg_* 子目录
        segment_dirs = sorted(
            d for d in source_path.iterdir()
            if d.is_dir() and (d / "segment.json").is_file()
        )
    else:
        print(f"错误: 找不到 segment.json: {source_path}", file=sys.stderr)
        return 1

    if not segment_dirs:
        print("错误: 没有找到可验证的 Segment 目录", file=sys.stderr)
        return 1

    print(f"验证 {len(segment_dirs)} 个 Segment...")
    print()

    all_reports: list[dict] = []
    overall_fail = 0
    overall_warn = 0
    overall_pass = 0

    for seg_dir in segment_dirs:
        print(f"--- {seg_dir.name} ---")

        try:
            report = validate_segment(seg_dir)
        except Exception as e:
            print(f"  ✗ 验证异常: {e}")
            overall_fail += 1
            all_reports.append({
                "segment_id": seg_dir.name,
                "status": "fail",
                "error": str(e),
            })
            print()
            continue

        # 写出验证报告到 segment 目录
        report_path = write_validation_report(report, seg_dir)
        all_reports.append({"segment_dir": str(seg_dir), **report})

        status = report["status"]
        if status == "fail":
            overall_fail += 1
        elif status == "pass_with_warning":
            overall_warn += 1
        else:
            overall_pass += 1

        # 打印摘要
        print(f"  Status: {status}")
        for check_id, result in report["checks"].items():
            icon = {"pass": "✓", "warning": "⚠", "fail": "✗"}.get(result, "?")
            print(f"  {icon} {check_id}: {result}")
        if report.get("statistics"):
            print(f"  Stats: {json.dumps(report['statistics'], ensure_ascii=False)}")
        print(f"  报告 → {report_path}")
        print()

    # ---- 汇总 ----
    print("=" * 50)
    print(f"总计: {len(segment_dirs)} Segment")
    print(f"  Pass:            {overall_pass}")
    print(f"  Pass with Warn:  {overall_warn}")
    print(f"  Fail:            {overall_fail}")
    print("=" * 50)

    # 写出汇总报告
    if args.output_report and len(segment_dirs) > 1:
        summary = {
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "total": len(segment_dirs),
            "pass": overall_pass,
            "pass_with_warning": overall_warn,
            "fail": overall_fail,
            "segments": all_reports,
        }
        with open(args.output_report, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\n汇总报告 → {args.output_report}")

    return 0 if overall_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
