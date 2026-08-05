"""人员 B：SceneProposal、VLM 复核与运行摘要的 parquet/JSON 写出。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from zpds.scene.schemas import SceneProposal, VLMReviewResult

SCENE_COLUMNS = (
    "scene_id",
    "start_ns",
    "end_ns",
    "confidence",
    "sources",
    "boundary_scores",
    "evidence_uris",
    "short_span",
    "producer",
    "version",
    "config_hash",
)

VLM_COLUMNS = (
    "scene_id",
    "scene_label",
    "task_label",
    "decision",
    "confidence",
    "reasons",
    "evidence_frame_uris",
    "producer",
    "version",
    "config_hash",
)


@dataclass(frozen=True)
class SceneWriteResult:
    output_dir: Path
    scene_file: Path
    vlm_file: Path
    summary_file: Path


def _scene_rows(
    scenes: Sequence[SceneProposal],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for scene in scenes:
        row = asdict(scene)
        row["sources"] = ",".join(scene.sources)
        row["boundary_scores"] = json.dumps(
            dict(scene.boundary_scores),
            ensure_ascii=False,
            sort_keys=True,
        )
        row["evidence_uris"] = ",".join(scene.evidence_uris)
        rows.append(row)
    return rows


def _vlm_rows(
    results: Sequence[VLMReviewResult],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for result in results:
        row = asdict(result)
        row["evidence_frame_uris"] = ",".join(
            result.evidence_frame_uris
        )
        rows.append(row)
    return rows


def _write_parquet_atomic(
    dataframe: pd.DataFrame,
    path: Path,
    *,
    columns: tuple[str, ...],
) -> None:
    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"parquet 缺少必需列: {sorted(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    dataframe[list(columns)].to_parquet(temporary, index=False)
    temporary.replace(path)


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def write_scene_run(
    output_dir: str | Path,
    *,
    input_path: str | Path,
    config_hash: str,
    profile: str | None,
    fps: float,
    frame_count: int,
    start_ns: int,
    end_ns: int,
    scenes: Sequence[SceneProposal],
    vlm_results: Sequence[VLMReviewResult],
    review_queue: Sequence[VLMReviewResult],
    skipped: bool = False,
    skip_reason: str | None = None,
    extra: dict[str, Any] | None = None,
) -> SceneWriteResult:
    """写出 scene_proposals.parquet、vlm_review.parquet 与 run_summary.json。"""

    root = Path(output_dir).expanduser().resolve()
    scene_file = root / "scene_proposals.parquet"
    vlm_file = root / "vlm_review.parquet"
    summary_file = root / "run_summary.json"

    if skipped:
        summary_document: dict[str, object] = {
            "input": str(input_path),
            "profile": profile,
            "config_hash": config_hash,
            "fps": fps,
            "frame_count": frame_count,
            "start_ns": start_ns,
            "end_ns": end_ns,
            "skipped": True,
            "skip_reason": skip_reason,
            "scene_count": 0,
            "vlm_reviewed": 0,
            "review_queue_scene_ids": [],
        }
        if extra:
            summary_document.update(extra)
        _write_json_atomic(summary_file, summary_document)
        return SceneWriteResult(
            output_dir=root,
            scene_file=scene_file,
            vlm_file=vlm_file,
            summary_file=summary_file,
        )

    scene_rows = _scene_rows(scenes)
    vlm_rows = _vlm_rows(vlm_results)
    if scene_rows:
        _write_parquet_atomic(
            pd.DataFrame(scene_rows),
            scene_file,
            columns=SCENE_COLUMNS,
        )
    else:
        empty = pd.DataFrame(columns=SCENE_COLUMNS)
        _write_parquet_atomic(empty, scene_file, columns=SCENE_COLUMNS)
    _write_parquet_atomic(
        pd.DataFrame(vlm_rows) if vlm_rows else pd.DataFrame(columns=VLM_COLUMNS),
        vlm_file,
        columns=VLM_COLUMNS,
    )

    document: dict[str, object] = {
        "input": str(input_path),
        "profile": profile,
        "config_hash": config_hash,
        "fps": fps,
        "frame_count": frame_count,
        "start_ns": start_ns,
        "end_ns": end_ns,
        "skipped": skipped,
        "skip_reason": skip_reason,
        "scene_count": len(scenes),
        "vlm_reviewed": len(vlm_results),
        "review_queue_scene_ids": [
            result.scene_id for result in review_queue
        ],
    }
    if extra:
        document.update(extra)
    _write_json_atomic(summary_file, document)
    return SceneWriteResult(
        output_dir=root,
        scene_file=scene_file,
        vlm_file=vlm_file,
        summary_file=summary_file,
    )


__all__ = [
    "SCENE_COLUMNS",
    "VLM_COLUMNS",
    "SceneWriteResult",
    "write_scene_run",
]
