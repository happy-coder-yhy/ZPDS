"""Import normalized existing Prepared annotations into an Experience version."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from zpds.annotation.importer import import_segment_annotations


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="将 Prepared Segment 的既有标注导入 Experience",
    )
    parser.add_argument(
        "--segment",
        action="append",
        required=True,
        help="Prepared Segment 目录；可重复指定",
    )
    parser.add_argument("--experience-dir", required=True, help="Experience 输出目录")
    parser.add_argument("--experience-version", help="默认使用 Experience 目录名")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    imported = 0
    try:
        for segment in args.segment:
            manifest = import_segment_annotations(
                Path(segment),
                Path(args.experience_dir),
                experience_version=args.experience_version,
            )
            if manifest is None:
                print(f"Skipped (no declared annotations): {Path(segment).resolve()}")
            else:
                imported += 1
                print(f"Imported: {Path(segment).resolve()} -> {manifest}")
    except Exception as error:  # noqa: BLE001 - CLI boundary reports a stable failure.
        print(f"Annotation import failed: {error}", file=sys.stderr)
        return 1
    print(f"Imported segments: {imported}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
