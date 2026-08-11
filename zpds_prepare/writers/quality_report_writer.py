"""统一预清洗质检报告写入器。

该报告是审核平台唯一接收和返回的 JSON。检测结果、候选切分、审核空位、
源数据身份和完整性信息均封装在 ``zpds.preclean_quality_report.v1.9`` 中。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import CandidateSegment
from zpds_prepare.quality_report_contract import (
    ALLOWED_ACTIONS,
    ALLOWED_ASSET_FORMATS,
    ALLOWED_CHECK_APPLICABILITY,
    ALLOWED_CHECK_STATUSES,
    ALLOWED_CLOCK_IDS,
    ALLOWED_EVIDENCE_TYPES,
    ALLOWED_LOCATOR_METHODS,
    ALLOWED_MODALITIES,
    ALLOWED_OVERALL_STATUSES,
    ALLOWED_SEGMENT_DISPOSITIONS,
    ALLOWED_SEGMENT_STATUSES,
    ALLOWED_SEVERITIES,
    ALLOWED_STREAM_STATUSES,
    CHECK_IDS,
    ISSUE_SCHEMA_VERSION,
    PROFILE_SOURCE_TYPE,
    SCHEMA_VERSION,
    SYNCHRONIZATION_RULE,
    TOP_LEVEL_KEYS,
    normalize_action,
)


def sha256_path(path: str | Path) -> str:
    """计算文件或目录的稳定 SHA256。

    目录哈希包含相对路径和每个文件的内容，避免只比较目录名称。
    """

    source = Path(path).resolve()
    digest = hashlib.sha256()
    if source.is_file():
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if source.is_dir():
        for member in sorted(p for p in source.rglob("*") if p.is_file()):
            digest.update(member.relative_to(source).as_posix().encode("utf-8"))
            digest.update(b"\0")
            with member.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            digest.update(b"\n")
        return digest.hexdigest()
    raise FileNotFoundError(f"源数据不存在: {source}")


def compute_immutable_hash(document: dict[str, Any]) -> str:
    """计算平台不得修改部分的哈希。

    问题级和报告级审核字段、Segment 审核状态以及 integrity 本身不参与计算。
    平台填写这些字段后，哈希仍应保持一致。
    """

    stable = json.loads(json.dumps(document, ensure_ascii=False))
    stable.pop("integrity", None)
    stable.pop("report_review", None)
    for issue in stable.get("issues", []):
        issue.pop("review", None)
    for segment in stable.get("proposed_cleaning", {}).get("segments", []):
        segment.pop("status", None)
    encoded = json.dumps(
        stable, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())


def _source_format(path: Path) -> str:
    if not path.is_file():
        return "directory"
    suffix = path.suffix.lower().lstrip(".")
    return {"h5": "hdf5", "pkl": "pickle"}.get(suffix, suffix)


def _build_source_assets(
    source: Path,
    session: Any,
) -> tuple[list[dict[str, Any]], dict[Path, str]]:
    """Build physical source assets and a resolved-path lookup."""

    paths = [source]
    for stream in session.annotation_streams.values():
        annotation_path = Path(stream.source_path).resolve()
        if source.is_dir() and annotation_path.is_relative_to(source):
            continue
        if annotation_path not in paths:
            paths.append(annotation_path)

    assets: list[dict[str, Any]] = []
    by_path: dict[Path, str] = {}
    for index, path in enumerate(paths, start=1):
        asset_id = f"source_{index:06d}"
        uri = source.name if path == source else _relative_locator(path, source)
        assets.append({
            "asset_id": asset_id,
            "uri": uri,
            "format": _source_format(path),
            "readable": path.exists(),
            "size_bytes": _source_size(path),
            "sha256": sha256_path(path),
        })
        by_path[path] = asset_id
    return assets, by_path


def _relative_locator(path: str | Path, source: Path) -> str:
    candidate = Path(path).resolve()
    root = source if source.is_dir() else source.parent
    try:
        return candidate.relative_to(root).as_posix()
    except ValueError:
        return candidate.as_posix()


def _stream_locator(profile: str, kind: str, stream: Any, source: Path) -> str:
    """Return a source-side locator, never a generated cache URI."""

    stream_id = str(stream.stream_id)
    topic_maps = {
        "dunjia": {
            "camera0": "/robot0/sensor/camera0/compressed",
            "camera1": "/robot0/sensor/camera1/compressed",
            "camera2": "/robot0/sensor/camera2/compressed",
            "robot0_imu": "/robot0/sensor/imu",
            "ego_audio": "/robot0/sensor/audio",
            "ego_depth": "/robot0/sensor/depth/compressed",
        },
        "umi": {
            "robot0_camera0": "/robot0/sensor/camera0/compressed",
            "robot1_camera0": "/robot1/sensor/camera0/compressed",
            "robot0_imu": "/robot0/sensor/imu",
            "robot1_imu": "/robot1/sensor/imu",
            "robot0_magnetic_encoder": "/robot0/sensor/magnetic_encoder",
            "robot1_magnetic_encoder": "/robot1/sensor/magnetic_encoder",
            "robot0_vio_pose": "/robot0/vio/eef_pose",
            "robot1_vio_pose": "/robot1/vio/eef_pose",
        },
    }
    if stream_id in topic_maps.get(profile, {}):
        return topic_maps[profile][stream_id]

    metadata = getattr(stream, "metadata", {}) or {}
    if metadata.get("source_topic"):
        return str(metadata["source_topic"])
    if metadata.get("topic"):
        return str(metadata["topic"])

    if kind == "annotation":
        return _relative_locator(stream.source_path, source)
    if kind == "timeseries":
        base = _relative_locator(stream.source_path, source)
        a2d_groups = {
            "robot_state": "/state/robot",
            "robot_action": "/action/robot",
            "gripper_state": "/state/gripper",
            "gripper_action": "/action/gripper",
        }
        suffix = a2d_groups.get(stream_id)
        return f"{base}#{suffix}" if suffix else base
    if kind == "depth" and getattr(stream, "source_files", None):
        return _relative_locator(stream.source_files[0], source)
    if kind == "video":
        return _relative_locator(stream.video_path, source)
    if profile == "guida" and kind == "imu":
        return "imu/imu_000000.csv"
    return stream_id


def _stream_inventory(
    session: Any,
    asset_id: str,
    *,
    profile: str,
    source: Path,
    asset_by_path: dict[Path, str],
) -> list[dict[str, Any]]:
    streams: list[dict[str, Any]] = []

    def add(
        stream_id: str,
        modality: str,
        timestamps: Iterable[int],
        count: int,
        rate: float | None,
        source_locator: str,
        row_asset_id: str | None = None,
    ) -> None:
        values = list(timestamps)
        normalized_modality = {
            "gripper_state": "joint_state",
            "gripper_command": "joint_command",
        }.get(modality, modality)
        streams.append({
            "stream_id": stream_id,
            "modality": normalized_modality,
            "asset_id": row_asset_id or asset_id,
            "source_locator": source_locator,
            "required": True,
            "status": "available" if count > 0 else "missing",
            "start_ns": int(values[0]) if values else None,
            "end_ns": int(values[-1]) if values else None,
            "sample_count": int(count),
            "nominal_rate_hz": float(rate) if rate is not None else None,
        })

    for stream in session.video_streams.values():
        add(
            stream.stream_id,
            "rgb",
            stream.timestamps_ns,
            stream.frame_count,
            stream.fps,
            _stream_locator(profile, "video", stream, source),
        )
    for stream in session.depth_streams.values():
        add(
            stream.stream_id,
            "depth",
            stream.timestamps_ns,
            stream.frame_count,
            stream.fps,
            _stream_locator(profile, "depth", stream, source),
        )
    for stream in session.imu_streams.values():
        timestamps = (
            stream.dataframe["timestamp_ns"].tolist()
            if "timestamp_ns" in stream.dataframe.columns
            else []
        )
        add(
            stream.stream_id,
            "imu",
            timestamps,
            len(stream.dataframe),
            stream.sample_rate_hz,
            _stream_locator(profile, "imu", stream, source),
        )
    for stream in session.audio_streams.values():
        timestamps = [int(packet["timestamp_ns"]) for packet in stream.packets]
        add(
            stream.stream_id,
            "audio",
            timestamps,
            stream.num_packets,
            stream.sample_rate_hz,
            _stream_locator(profile, "audio", stream, source),
        )
    for stream in session.time_series_streams.values():
        add(
            stream.stream_id,
            stream.modality,
            stream.timestamps_ns,
            stream.num_samples,
            stream.expected_rate_hz,
            _stream_locator(profile, "timeseries", stream, source),
        )
    for stream in session.annotation_streams.values():
        source_video = session.video_streams.get(stream.source_video_stream_id)
        timestamps = source_video.timestamps_ns if source_video is not None else []
        add(
            stream.stream_id,
            stream.annotation_type if stream.annotation_type in {"mask"} else "annotation",
            timestamps,
            len(stream.records),
            None,
            _stream_locator(profile, "annotation", stream, source),
            asset_by_path.get(Path(stream.source_path).resolve()),
        )
    return streams


def _checks(
    issues: list[QualityIssue],
    stream_ids: list[str],
    *,
    cascade_executed: bool,
    scene_executed: bool,
) -> list[dict[str, Any]]:
    issue_ids = [f"iss_{index:06d}" for index in range(1, len(issues) + 1)]
    failed = any(issue.severity in {"error", "critical"} for issue in issues)
    status = "failed" if failed else ("warning" if issues else "passed")
    return [
        {
            "check_id": "source_readability",
            "name": "源数据可读性",
            "scope": stream_ids,
            "applicability": "applicable",
            "status": "passed",
            "detector": {"name": "session_reader", "version": "1"},
            "parameters": {},
            "metrics": {"readable_stream_count": len(stream_ids)},
            "issue_ids": [],
        },
        {
            "check_id": "quality_detection",
            "name": "预清洗质量检测",
            "scope": stream_ids,
            "applicability": "applicable",
            "status": status,
            "detector": {"name": "zpds_prepare", "version": "1"},
            "parameters": {},
            "metrics": {"issue_count": len(issues)},
            "issue_ids": issue_ids,
        },
        {
            "check_id": "qc_cascade",
            "name": "QC 级联检查",
            "scope": stream_ids,
            "applicability": "applicable",
            "status": "passed" if cascade_executed else "not_run",
            "detector": {"name": "zpds_qc_cascade", "version": "1"},
            "parameters": {},
            "metrics": {},
            "issue_ids": [],
            **({} if cascade_executed else {"reason": "命令行指定跳过 QC 级联。"}),
        },
        {
            "check_id": "scene_segmentation",
            "name": "场景分割",
            "scope": [],
            "applicability": "applicable",
            "status": "passed" if scene_executed else "not_run",
            "detector": {"name": "zpds_scene", "version": "1"},
            "parameters": {},
            "metrics": {},
            "issue_ids": [],
            **({} if scene_executed else {"reason": "按当前流程约定不执行场景分割。"}),
        },
    ]


def _issue_and_evidence(
    issues: list[QualityIssue],
    inventory: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    issue_rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    streams = {row["stream_id"]: row for row in inventory}
    fallback_stream = inventory[0] if inventory else None
    for index, issue in enumerate(issues, start=1):
        issue_id = f"iss_{index:06d}"
        evidence_id = f"ev_{index:06d}"
        # The QC cascade uses the internal values ``warn``/``info``, while the
        # public quality-report contract exposes ``warning``/``error``/``critical``.
        # Informational findings are retained as non-blocking warnings because
        # v1.9 intentionally has no public ``info`` severity.
        severity = {
            "warn": "warning",
            "warning": "warning",
            "info": "warning",
            "error": "error",
            "critical": "critical",
        }.get(str(issue.severity).lower(), str(issue.severity).lower())
        external_action = normalize_action(issue.decision)
        issue_start_ns = int(issue.start_ns)
        issue_end_ns = int(issue.end_ns)
        if issue_end_ns < issue_start_ns:
            issue_start_ns, issue_end_ns = issue_end_ns, issue_start_ns
        if issue.stream_id == "all" and issue_start_ns == issue_end_ns == 0:
            starts = [row["start_ns"] for row in inventory if row["start_ns"] is not None]
            ends = [row["end_ns"] for row in inventory if row["end_ns"] is not None]
            if starts and ends:
                issue_start_ns = min(starts)
                issue_end_ns = max(ends)
        affected_streams = (
            list(streams)
            if issue.stream_id == "all"
            else [issue.stream_id]
        )
        issue_rows.append({
            "issue_id": issue_id,
            "issue_type": issue.issue_type,
            "stream_id": issue.stream_id,
            "start_ns": issue_start_ns,
            "end_ns": issue_end_ns,
            "duration_ns": issue_end_ns - issue_start_ns,
            "severity": severity,
            "decision": external_action,
            "details": issue.details,
            "evidence_refs": [evidence_id],
            "proposed_action": {
                "action": external_action,
                "affected_streams": affected_streams,
                "start_ns": issue_start_ns,
                "end_ns": issue_end_ns,
                "reason": issue.details.get("message", issue.issue_type),
            },
            "review": {
                "status": "pending",
                "reviewer_id": None,
                "reviewed_at": None,
                "decision": None,
                "modified_action": None,
                "reason_code": None,
                "note": "",
                "evidence_checked": False,
            },
        })
        evidence_stream = streams.get(issue.stream_id, fallback_stream)
        if evidence_stream is None:
            raise ValueError(f"{issue_id} 无法关联任何数据流")
        point_evidence = issue_start_ns == issue_end_ns
        evidence: dict[str, Any] = {
            "evidence_id": evidence_id,
            "issue_id": issue_id,
            "type": "timestamp_point" if point_evidence else "timestamp_range",
            "source_asset_id": evidence_stream["asset_id"],
            "stream_id": evidence_stream["stream_id"],
            "clock_id": "source_device_clock",
            "context_before_ns": 1_000_000_000,
            "context_after_ns": 1_000_000_000,
            "details_ref": f"#issues/{index - 1}/details",
            "description": (
                "前端根据源资产、数据流和时间字段定位原始数据，"
                "不提供单帧图片。"
            ),
        }
        if point_evidence:
            rate = evidence_stream.get("nominal_rate_hz")
            tolerance = round(1_000_000_000 / rate) if rate and rate > 0 else 0
            evidence.update({
                "timestamp_ns": issue_start_ns,
                "locator_method": "nearest_timestamp",
                "tolerance_ns": int(tolerance),
            })
        else:
            evidence.update({
                "start_ns": issue_start_ns,
                "end_ns": issue_end_ns,
                "locator_method": "range_overlap",
            })
        evidence_rows.append(evidence)
    return issue_rows, evidence_rows


def _segments(
    candidates: list[CandidateSegment], issues: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded = [
        row["issue_id"]
        for row in issues
        if row["decision"] in {"split", "quarantine"}
    ]
    for candidate in candidates:
        rows.append({
            "candidate_id": candidate.candidate_id,
            "start_ns": candidate.source_start_ns,
            "end_ns": candidate.source_end_ns,
            "duration_ns": candidate.duration_ns,
            "included_streams": [],
            "excluded_issue_ids": excluded,
            "recommended_disposition": "keep",
            "reason": candidate.reason,
            "issues_in_span": candidate.issues_in_span,
            "status": "pending_review",
        })
    return rows


def validate_generated_report(document: dict[str, Any]) -> None:
    """Validate the v1.9 instance before it is written or signed."""

    errors: list[str] = []
    if set(document) != TOP_LEVEL_KEYS:
        errors.append(
            "顶层字段不一致: "
            f"缺少={sorted(TOP_LEVEL_KEYS - set(document))}, "
            f"多余={sorted(set(document) - TOP_LEVEL_KEYS)}"
        )
    if document.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version 必须为 {SCHEMA_VERSION}")

    dataset = document.get("dataset") or {}
    profile = dataset.get("profile")
    if profile not in PROFILE_SOURCE_TYPE:
        errors.append(f"未知 profile: {profile!r}")
    elif dataset.get("source_type") != PROFILE_SOURCE_TYPE[profile]:
        errors.append(f"{profile} 的 source_type 不正确")
    if dataset.get("units") != {
        "time": "ns",
        "length": "m",
        "angle": "rad",
        "temperature": "degC",
    }:
        errors.append("dataset.units 不符合固定单位")
    time_range = dataset.get("time_range") or {}
    session_start = time_range.get("start_ns")
    session_end = time_range.get("end_ns")
    if (
        not isinstance(session_start, int)
        or not isinstance(session_end, int)
        or session_start >= session_end
        or time_range.get("duration_ns") != session_end - session_start
    ):
        errors.append("dataset.time_range 非法")

    assets = document.get("source_assets") or []
    asset_ids = {row.get("asset_id") for row in assets}
    if not assets or None in asset_ids or len(asset_ids) != len(assets):
        errors.append("source_assets 缺失或 asset_id 重复")
    for asset in assets:
        if not asset.get("uri") or not asset.get("format") or not asset.get("sha256"):
            errors.append(f"Source Asset 不完整: {asset.get('asset_id')!r}")
        if asset.get("format") not in ALLOWED_ASSET_FORMATS:
            errors.append(f"{asset.get('asset_id')} format 非法")

    inventory = document.get("stream_inventory") or []
    stream_ids = {row.get("stream_id") for row in inventory}
    streams_by_id = {row.get("stream_id"): row for row in inventory}
    if not inventory or None in stream_ids or len(stream_ids) != len(inventory):
        errors.append("stream_inventory 缺失或 stream_id 重复")
    for stream in inventory:
        if stream.get("asset_id") not in asset_ids:
            errors.append(f"{stream.get('stream_id')} 引用了不存在的 Asset")
        if not stream.get("source_locator"):
            errors.append(f"{stream.get('stream_id')} 缺少 source_locator")
        if stream.get("modality") not in ALLOWED_MODALITIES:
            errors.append(f"{stream.get('stream_id')} modality 非法")
        if stream.get("status") not in ALLOWED_STREAM_STATUSES:
            errors.append(f"{stream.get('stream_id')} status 非法")

    issues = document.get("issues") or []
    issue_ids = {row.get("issue_id") for row in issues}
    if None in issue_ids or len(issue_ids) != len(issues):
        errors.append("Issue ID 缺失或重复")
    evidence = document.get("evidence_index") or []
    evidence_ids = {row.get("evidence_id") for row in evidence}
    if None in evidence_ids or len(evidence_ids) != len(evidence):
        errors.append("Evidence ID 缺失或重复")

    for issue in issues:
        issue_id = issue.get("issue_id")
        start_ns = issue.get("start_ns")
        end_ns = issue.get("end_ns")
        if issue.get("severity") not in ALLOWED_SEVERITIES:
            errors.append(f"{issue_id} severity 非法")
        if issue.get("decision") not in ALLOWED_ACTIONS:
            errors.append(f"{issue_id} decision 非法")
        proposed = issue.get("proposed_action") or {}
        if proposed.get("action") not in ALLOWED_ACTIONS:
            errors.append(f"{issue_id} proposed_action.action 非法")
        if (
            not isinstance(start_ns, int)
            or not isinstance(end_ns, int)
            or start_ns > end_ns
            or issue.get("duration_ns") != end_ns - start_ns
        ):
            errors.append(f"{issue_id} 时间范围非法")
        else:
            stream = streams_by_id.get(issue.get("stream_id"))
            if stream is not None:
                stream_start_ns = stream.get("start_ns")
                stream_end_ns = stream.get("end_ns")
                if (
                    isinstance(stream_start_ns, int)
                    and isinstance(stream_end_ns, int)
                    and (
                        start_ns < stream_start_ns
                        or end_ns > stream_end_ns
                    )
                ):
                    errors.append(f"{issue_id} 时间范围超出对应 Stream")
        if not all(ref in evidence_ids for ref in issue.get("evidence_refs", [])):
            errors.append(f"{issue_id} Evidence 引用不存在")

    for row in evidence:
        evidence_id = row.get("evidence_id")
        evidence_type = row.get("type")
        if row.get("issue_id") not in issue_ids:
            errors.append(f"{evidence_id} Issue 引用不存在")
        if row.get("source_asset_id") not in asset_ids:
            errors.append(f"{evidence_id} Asset 引用不存在")
        if row.get("stream_id") not in stream_ids:
            errors.append(f"{evidence_id} Stream 引用不存在")
        if row.get("clock_id") not in ALLOWED_CLOCK_IDS:
            errors.append(f"{evidence_id} clock_id 非法")
        if evidence_type not in ALLOWED_EVIDENCE_TYPES:
            errors.append(f"{evidence_id} type 非法")
        if row.get("locator_method") not in ALLOWED_LOCATOR_METHODS:
            errors.append(f"{evidence_id} locator_method 非法")
        if evidence_type == "timestamp_point":
            if "timestamp_ns" not in row or "start_ns" in row or "end_ns" in row:
                errors.append(f"{evidence_id} 单点 Evidence 字段组合非法")
            if row.get("locator_method") != "nearest_timestamp":
                errors.append(f"{evidence_id} 单点定位方式非法")
        elif evidence_type == "timestamp_range":
            if "start_ns" not in row or "end_ns" not in row or "timestamp_ns" in row:
                errors.append(f"{evidence_id} 范围 Evidence 字段组合非法")
            if row.get("locator_method") != "range_overlap":
                errors.append(f"{evidence_id} 范围定位方式非法")

    checks = (document.get("check_coverage") or {}).get("checks") or []
    for check in checks:
        if check.get("check_id") not in CHECK_IDS:
            errors.append(f"未知 check_id: {check.get('check_id')!r}")
        if check.get("status") not in ALLOWED_CHECK_STATUSES:
            errors.append(f"{check.get('check_id')} status 非法")
        if check.get("applicability") not in ALLOWED_CHECK_APPLICABILITY:
            errors.append(f"{check.get('check_id')} applicability 非法")
        if not all(ref in issue_ids for ref in check.get("issue_ids", [])):
            errors.append(f"{check.get('check_id')} Issue 引用不存在")

    cleaning = document.get("proposed_cleaning") or {}
    if cleaning.get("synchronization_rule") != SYNCHRONIZATION_RULE:
        errors.append("synchronization_rule 非法")
    if cleaning.get("master_clock") not in ALLOWED_CLOCK_IDS:
        errors.append("master_clock 非法")
    for segment in cleaning.get("segments") or []:
        if segment.get("status") not in ALLOWED_SEGMENT_STATUSES:
            errors.append(f"{segment.get('candidate_id')} status 非法")
        if segment.get("recommended_disposition") not in ALLOWED_SEGMENT_DISPOSITIONS:
            errors.append(f"{segment.get('candidate_id')} disposition 非法")

    overall = document.get("overall_result") or {}
    if overall.get("status") not in ALLOWED_OVERALL_STATUSES:
        errors.append("overall_result.status 非法")
    if overall.get("recommended_action") not in ALLOWED_ACTIONS:
        errors.append("overall_result.recommended_action 非法")

    summary = document.get("summary") or {}
    if summary.get("issue_count") != len(issues):
        errors.append("summary.issue_count 与 Issues 不一致")
    if summary.get("proposed_segment_count") != len(cleaning.get("segments") or []):
        errors.append("summary.proposed_segment_count 与 Segments 不一致")

    if errors:
        raise ValueError("quality_report v1.9 校验失败:\n- " + "\n- ".join(errors))


def write_quality_report(
    output_path: Path,
    *,
    issues: list[QualityIssue],
    candidates: list[CandidateSegment],
    session: Any,
    dataset_path: str,
    profile: str,
    cascade_executed: bool,
    scene_executed: bool,
    pipeline_version: str = "0.1.0",
    config_version: str = "config.yaml",
) -> Path:
    """写出审核平台唯一使用的统一质检报告。"""

    if profile not in PROFILE_SOURCE_TYPE:
        raise ValueError(f"不支持的 profile: {profile!r}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = Path(dataset_path).resolve()
    start_ns = int(session.session_start_ns)
    end_ns = int(session.session_end_ns)
    source_asset_id = "source_000001"
    source_assets, asset_by_path = _build_source_assets(source, session)
    inventory = _stream_inventory(
        session,
        source_asset_id,
        profile=profile,
        source=source,
        asset_by_path=asset_by_path,
    )
    stream_ids = [stream["stream_id"] for stream in inventory]
    checks = _checks(
        issues,
        stream_ids,
        cascade_executed=cascade_executed,
        scene_executed=scene_executed,
    )
    issue_rows, evidence_rows = _issue_and_evidence(issues, inventory)
    segment_rows = _segments(candidates, issue_rows)
    for segment in segment_rows:
        segment["included_streams"] = stream_ids

    by_decision = Counter(row["decision"] for row in issue_rows)
    by_severity = Counter(row["severity"] for row in issue_rows)
    by_stream = Counter(row["stream_id"] for row in issue_rows)
    by_type = Counter(row["issue_type"] for row in issue_rows)
    by_check = Counter(row["status"] for row in checks)
    keep_duration = sum(row["duration_ns"] for row in segment_rows)
    duration_ns = end_ns - start_ns
    blocking = sum(row["severity"] in {"error", "critical"} for row in issue_rows)

    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "quality_issue_schema_version": ISSUE_SCHEMA_VERSION,
        "report_metadata": {
            "report_id": f"qcr_{profile}_{session.session_id}",
            "report_type": "preclean_quality_report",
            "title": f"{profile.upper()} 预清洗质检报告",
            "generated_at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "generator": {
                "name": "zpds_prepare",
                "pipeline_version": pipeline_version,
                "code_revision": "unknown",
                "config_version": config_version,
            },
            "profile_note": "平台审核后返回同一 JSON，仅填写 review/status 字段。",
        },
        "dataset": {
            "dataset_id": source.stem,
            "data_id": session.session_id,
            "source_session_id": session.session_id,
            "profile": profile,
            "source_type": PROFILE_SOURCE_TYPE[profile],
            "time_range": {
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_ns": duration_ns,
            },
            "units": {"time": "ns", "length": "m", "angle": "rad", "temperature": "degC"},
        },
        "source_assets": source_assets,
        "stream_inventory": inventory,
        "check_coverage": {
            "required_check_count": sum(row["applicability"] == "applicable" for row in checks),
            "executed_check_count": sum(row["applicability"] == "applicable" and row["status"] != "not_run" for row in checks),
            "all_required_checks_executed": not any(row["applicability"] == "applicable" and row["status"] in {"not_run", "unavailable"} for row in checks),
            "checks": checks,
        },
        "issues": issue_rows,
        "evidence_index": evidence_rows,
        "proposed_cleaning": {
            "strategy": "reviewed_issue_planning",
            "synchronization_rule": SYNCHRONIZATION_RULE,
            "synchronization_description": (
                "所有关联数据流按同一主时间轴同步裁剪或切分。"
            ),
            "master_clock": "source_device_clock",
            "segments": segment_rows,
        },
        "summary": {
            "issue_count": len(issue_rows),
            "by_decision": dict(by_decision),
            "by_severity": dict(by_severity),
            "by_stream": dict(by_stream),
            "by_issue_type": dict(by_type),
            "checks_by_status": dict(by_check),
            "issues_by_review_status": {"pending": len(issue_rows)} if issue_rows else {},
            "source_duration_ns": duration_ns,
            "recommended_keep_duration_ns": keep_duration,
            "recommended_keep_ratio": round(keep_duration / duration_ns, 6) if duration_ns else 0,
            "proposed_segment_count": len(segment_rows),
        },
        "overall_result": {
            "status": "review_required",
            "recommended_action": "manual_review",
            "blocking_issue_count": blocking,
            "warning_issue_count": sum(row["severity"] == "warning" for row in issue_rows),
            "reasons": [f"{row['issue_id']}: {row['issue_type']}" for row in issue_rows],
            "limitations": ["场景分割未启用。"] if not scene_executed else [],
        },
        "report_review": {
            "status": "pending",
            "reviewer_id": None,
            "reviewed_at": None,
            "final_result": None,
            "note": "",
        },
        "integrity": {
            "report_content_sha256": None,
            "asset_hash_algorithm": "sha256",
            "validation_status": "verified",
            "schema_validation": "passed",
            "count_validation": "passed",
        },
    }
    validate_generated_report(document)
    document["integrity"]["report_content_sha256"] = compute_immutable_hash(document)
    output_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return output_path


__all__ = [
    "ISSUE_SCHEMA_VERSION",
    "SCHEMA_VERSION",
    "compute_immutable_hash",
    "sha256_path",
    "validate_generated_report",
    "write_quality_report",
]
