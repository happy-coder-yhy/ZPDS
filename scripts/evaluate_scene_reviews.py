"""校验 Scene 双人复核，并生成 provisional 评估报告。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from zpds.scene.review_evaluation import evaluate_reviews, load_review_jsonl


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="评估 Scene 边界双人复核完成度")
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    report = evaluate_reviews(load_review_jsonl(args.reviews))
    output = Path(args.output_json).expanduser().resolve()
    _write_json_atomic(output, report)
    print(f"复核完成: {report['effective_label_count']}/{report['review_item_count']}")
    print(f"待填写: {len(report['incomplete_item_ids'])}")
    print(f"待仲裁: {len(report['unresolved_item_ids'])}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"scene review evaluation failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
