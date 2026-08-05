"""人员 B：scene 产物回读校验与 Raw 哈希不变校验。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import pandas as pd

from zpds.scene.writer import VLM_COLUMNS


def sha256_file(path: str | Path) -> str:
    """分块计算文件 SHA-256，避免大文件整读内存。"""

    digest = hashlib.sha256()
    with Path(path).expanduser().resolve().open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class SceneValidationReport:
    ok: bool
    issues: tuple[str, ...]


def _check_scene_intervals(path: Path) -> list[str]:
    issues: list[str] = []
    try:
        frame = pd.read_parquet(path)
    except Exception as error:  # noqa: BLE001 - 校验报告需要聚合任意读取错误
        return [f"scene_proposals.parquet 无法回读: {error}"]
    if frame.empty:
        return ["scene_proposals.parquet 为空（场景数为 0）"]
    required = {"scene_id", "start_ns", "end_ns"}
    missing = required - set(frame.columns)
    if missing:
        return [f"scene_proposals.parquet 缺少列: {sorted(missing)}"]
    starts = frame["start_ns"].tolist()
    ends = frame["end_ns"].tolist()
    if any(start >= end for start, end in zip(starts, ends)):
        issues.append("存在 end_ns <= start_ns 的场景")
    ordered = sorted(zip(starts, ends))
    for (_, previous_end), (next_start, _) in pairwise(ordered):
        if next_start < previous_end:
            issues.append("场景区间存在重叠")
    return issues


def validate_scene_outputs(
    output_dir: str | Path,
    *,
    raw_path: str | Path | None = None,
    raw_sha256_before: str | None = None,
    expected_scene_count: int | None = None,
    expect_artifacts: bool = True,
) -> SceneValidationReport:
    """校验 parquet 可回读、区间合法、Raw 哈希未变。"""

    root = Path(output_dir).expanduser().resolve()
    issues: list[str] = []
    scene_file = root / "scene_proposals.parquet"
    vlm_file = root / "vlm_review.parquet"
    summary_file = root / "run_summary.json"

    if expect_artifacts:
        if not scene_file.is_file():
            issues.append("scene_proposals.parquet 缺失")
        if not vlm_file.is_file():
            issues.append("vlm_review.parquet 缺失")
        if not summary_file.is_file():
            issues.append("run_summary.json 缺失")

    if scene_file.is_file():
        issues.extend(_check_scene_intervals(scene_file))
    if vlm_file.is_file():
        try:
            frame = pd.read_parquet(vlm_file)
            missing = set(VLM_COLUMNS) - set(frame.columns)
            if missing:
                issues.append(f"vlm_review.parquet 缺少列: {sorted(missing)}")
        except Exception as error:  # noqa: BLE001
            issues.append(f"vlm_review.parquet 无法回读: {error}")
    if summary_file.is_file():
        try:
            document = json.loads(summary_file.read_text(encoding="utf-8"))
            if not isinstance(document, dict):
                issues.append("run_summary.json 顶层必须是对象")
            elif "config_hash" not in document:
                issues.append("run_summary.json 缺少 config_hash")
        except (OSError, json.JSONDecodeError) as error:
            issues.append(f"run_summary.json 无法解析: {error}")
    else:
        issues.append("run_summary.json 缺失")

    if expected_scene_count is not None and scene_file.is_file():
        try:
            frame = pd.read_parquet(scene_file)
        except Exception as error:  # noqa: BLE001
            issues.append(f"scene_proposals.parquet 无法回读: {error}")
        else:
            if len(frame) != expected_scene_count:
                issues.append(
                    f"场景数不一致: 期望 {expected_scene_count}，"
                    f"实际 {len(frame)}"
                )

    if raw_path is not None and raw_sha256_before is not None:
        raw = Path(raw_path).expanduser().resolve()
        if not raw.is_file():
            issues.append(f"Raw 文件不存在: {raw}")
        elif sha256_file(raw) != raw_sha256_before:
            issues.append("Raw 文件 SHA-256 在运行前后发生变化")

    return SceneValidationReport(ok=not issues, issues=tuple(issues))


__all__ = [
    "SceneValidationReport",
    "sha256_file",
    "validate_scene_outputs",
]
