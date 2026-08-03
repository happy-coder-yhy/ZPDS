"""Standalone CLI for the isolated UMI provisional pipeline."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from zpds_prepare.detectors.umi.provisional_pipeline import (
    run_umi_provisional_dataset,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache", required=True, type=Path)
    parser.add_argument("--minimum-gap-ns", type=int, default=500_000_000)
    parser.add_argument("--max-residual-ns", type=int, default=None)
    parser.add_argument("--encoder-freeze-min-samples", type=int, default=10)
    parser.add_argument("--producer-version", default="dev")
    parser.add_argument(
        "--bimanual-without-vio",
        action="store_true",
        help="Do not require dual VIO alignment for the candidate bimanual view.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_umi_provisional_dataset(
        args.dataset,
        args.output,
        cache_dir=args.cache,
        config={
            "minimum_gap_ns": args.minimum_gap_ns,
            "alignment_max_residual_ns": args.max_residual_ns,
            "encoder_freeze_min_samples": args.encoder_freeze_min_samples,
            "require_vio_for_bimanual": not args.bimanual_without_vio,
        },
        producer_version=args.producer_version,
    )
    print(
        json.dumps(
            {
                "session_id": result.session_id,
                "formal_manifest_written": result.formal_manifest_written,
                "human_hand_model_invoked": result.human_hand_model_invoked,
                "raw_unchanged": (
                    result.source_sha256_before == result.source_sha256_after
                ),
                "candidate_views": {
                    name: view.status
                    for name, view in result.candidate_views.items()
                },
                "evidence_artifact_count": len(
                    result.evidence_index.artifacts
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main"]
