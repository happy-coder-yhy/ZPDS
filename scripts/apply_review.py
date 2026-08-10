"""平台审核结果应用 CLI（人工审核闭环联调用）。

用法:
    python scripts/apply_review.py --original output/xxx/quality_issues.json \\
        --reviewed reviewed.json [--output merged.json]

读取原始 quality_issues.json（0.2.0）与平台审核返回的 json，按 review.status
（approved/rejected/modified/added）合并出新的 issues 列表；带 --output 时
写回 0.2.0 格式（审核区清空为 pending），供再次审核或直接作为切分输入。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.writers.quality_writer import write_quality_issues
from zpds_prepare.writers.review_applier import apply_review


def _load_issues(path: Path) -> list[QualityIssue]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    for raw in payload.get("issues", []):
        issues.append(QualityIssue(
            issue_type=str(raw["issue_type"]),
            stream_id=str(raw.get("stream_id", "")),
            start_ns=int(raw.get("start_ns", 0)),
            end_ns=int(raw.get("end_ns", raw.get("start_ns", 0))),
            severity=str(raw.get("severity", "warning")),
            decision=str(raw["decision"]),
            details=dict(raw.get("details") or {}),
        ))
    return issues


def main() -> int:
    parser = argparse.ArgumentParser(description="应用平台审核结果并输出合并后的 issues")
    parser.add_argument("--original", required=True, help="原始 quality_issues.json 路径")
    parser.add_argument("--reviewed", required=True, help="平台审核返回的 json 路径")
    parser.add_argument("--output", "-o", default=None,
                        help="合并结果输出路径（0.2.0 格式）；省略则只打印统计")
    args = parser.parse_args()

    original_path = Path(args.original)
    reviewed_path = Path(args.reviewed)
    if not original_path.is_file():
        print(f"原始报告不存在: {original_path}", file=sys.stderr)
        return 1
    if not reviewed_path.is_file():
        print(f"审核版不存在: {reviewed_path}", file=sys.stderr)
        return 1

    original = _load_issues(original_path)
    reviewed = json.loads(reviewed_path.read_text(encoding="utf-8"))
    merged, stats = apply_review(original, reviewed)

    print(f"原 issues:   {len(original)}")
    print(f"审核结果:    approved={stats.approved} rejected={stats.rejected} "
          f"modified={stats.modified} added={stats.added} kept={stats.kept}")
    print(f"合并后:      {len(merged)} 条（供重新切分）")

    if args.output:
        source_session_id = reviewed.get("source_session_id", original_path.stem)
        out = write_quality_issues(
            Path(args.output), merged, source_session_id=source_session_id
        )
        print(f"输出:        {out.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
