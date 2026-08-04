"""从 provisional 阈值回归报告生成双人边界复核包。"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from zpds.scene.review import (
    build_review_items,
    export_evidence_frames,
    safe_case_name,
    write_review_jsonl,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="生成 Scene 边界人工双人复核包")
    parser.add_argument("--regression-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--negative-per-case", type=int, default=8)
    parser.add_argument("--context-s", type=float, default=0.5)
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    regression_path = Path(args.regression_json).expanduser().resolve()
    document = json.loads(regression_path.read_text(encoding="utf-8"))
    if document.get("schema_version") != "zpds.scene.threshold_regression.v1":
        raise ValueError("不支持的 scene regression schema")
    output_dir = Path(args.output_dir).expanduser().resolve()
    all_items = []
    case_summaries = []

    for case in document["cases"]:
        video_path = Path(case["input"]).resolve()
        before_hash = _sha256(video_path)
        if before_hash != case["input_sha256"]:
            raise RuntimeError(f"视频哈希与回归报告不一致: {video_path}")
        items = build_review_items(
            case,
            negative_count=args.negative_per_case,
            context_s=args.context_s,
        )
        case_dir_name = safe_case_name(str(case["name"]))
        relative_dir = f"evidence/{case_dir_name}"
        export_evidence_frames(
            video_path,
            items,
            evidence_dir=output_dir / relative_dir,
            uri_prefix=relative_dir,
        )
        if _sha256(video_path) != before_hash:
            raise RuntimeError(f"视频在证据提取过程中发生变化: {video_path}")
        all_items.extend(items)
        case_summaries.append(
            {
                "name": case["name"],
                "profile": case.get("profile"),
                "candidate_count": sum(
                    item["sample_source"] == "detected_candidate" for item in items
                ),
                "negative_audit_count": sum(
                    item["sample_source"] == "negative_audit" for item in items
                ),
                "raw_unchanged": True,
            }
        )

    annotations_path = output_dir / "boundary_reviews.jsonl"
    write_review_jsonl(annotations_path, all_items)
    manifest = {
        "schema_version": "zpds.scene.boundary_review_pack.v1",
        "source_regression": str(regression_path),
        "calibration_status": "provisional",
        "requires_two_independent_reviewers": True,
        "requires_adjudication": True,
        "allowed_decisions": ["true_boundary", "no_boundary", "uncertain"],
        "allowed_boundary_types": ["hard_cut", "gradual", "semantic", "other"],
        "precision_eligible_after_adjudication": True,
        "recall_eligible": False,
        "recall_blocker": "必须由复核人员补录检测器未提出的真实边界",
        "automatic_threshold_install": False,
        "annotation_file": annotations_path.name,
        "review_item_count": len(all_items),
        "cases": case_summaries,
    }
    _write_json_atomic(output_dir / "review_manifest.json", manifest)
    print(f"复核包已生成: {output_dir}")
    print(f"复核项: {len(all_items)}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"scene review pack failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
