"""
生成 segment.json — Prepared Segment 的核心控制文件。

streams 列表根据传入的 video_results 和 imu_results 动态生成，
文件名由各流的 stream_id 决定，不再硬编码。
"""

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

_PROFILE_REGISTRY_NAMES = {
    "guida": "guida_ego",
    "dunjia": "dunjia_ego",
    "umi": "jianzhi_umi",
    "epic": "epic100",
    "a2d": "a2d_robot",
}


def _primary_stream_id(profile: str) -> str | None:
    """按 profile 声明的主相机 stream_id；无声明（单相机）返回 None。"""
    from zpds.profiles.registry import get

    registered = get(_PROFILE_REGISTRY_NAMES.get(profile, profile))
    if registered is None:
        return None
    return registered.primary_stream_id


def sha256_hex(path: str) -> str:
    """计算文件 SHA-256。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _utc_now_iso() -> str:
    """当前 UTC 时间，ISO-8601 格式（Z 后缀）。"""
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z")
    )


def _git_commit_short() -> str:
    """读取当前 git commit 短哈希；失败时返回 "unknown"。"""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=5,
            check=False,
        )
        commit = result.stdout.strip()
        if result.returncode == 0 and commit:
            return commit
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def package_version() -> str:
    """读取 zpds 包版本；安装元数据缺失时回退到 ``zpds.__version__``。"""
    try:
        installed = version("zpds")
    except PackageNotFoundError:
        installed = ""
    if installed:
        return installed
    try:
        import zpds
    except Exception:  # noqa: BLE001 - 版本探测不应阻断产物生成
        return "unknown"
    return str(getattr(zpds, "__version__", "unknown"))


def build_dataset_json(
    *,
    dataset_id: str,
    prep_revision: str,
    name: str | None = None,
    description: str | None = None,
    source_types: list[str] | None = None,
    dataset_version: str | None = None,
    zpds_version: str = "0.1.0",
    default_experience_version: str | None = None,
) -> dict:
    """构建 dataset.json（ZPDS 数据标准最小字段）。"""
    if dataset_version is None:
        dataset_version = package_version()
    document: dict[str, object] = {
        "zpds_version": zpds_version,
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "name": name or dataset_id,
        "description": description or "",
        "created_at": _utc_now_iso(),
        "source_types": source_types or [],
        "default_prep_revision": prep_revision,
    }
    if default_experience_version is not None:
        document["default_experience_version"] = default_experience_version
    return document


def build_revision_json(
    *,
    prep_revision: str,
    pipeline_name: str,
    pipeline_version: str | None = None,
    config_hash: str,
    code_commit: str | None = None,
    changes: list[str] | None = None,
    parent_revision: str | None = None,
    run_stats: dict[str, object] | None = None,
) -> dict:
    """构建 prepared_segments/<prep_revision>/revision.json。

    Prepared 层长度单位以 ``zpds.prepared.conventions.LENGTH_UNIT`` 为
    权威来源，并统一为米（m）。源资产采用其他单位时，由具体 Stream
    或标定来源字段显式记录，不通过修改标签伪装成已完成数值换算。
    """
    from zpds.prepared.conventions import LENGTH_UNIT

    if pipeline_version is None:
        pipeline_version = package_version()
    document: dict[str, object] = {
        "zpds_version": "0.1.0",
        "prep_revision": prep_revision,
        "parent_revision": parent_revision,
        "created_at": _utc_now_iso(),
        "pipeline": {
            "name": pipeline_name,
            "version": pipeline_version,
            "code_commit": code_commit or _git_commit_short(),
            "config_hash": config_hash,
        },
        "conventions": {
            "time_unit": "ns",
            "time_interval": "[start_ns,end_ns)",
            "segment_time_origin_ns": 0,
            "length_unit": LENGTH_UNIT,
            "length_unit_source": "zpds/prepared/conventions.py",
            "angle_unit": "rad",
            "quaternion_order": "xyzw",
            "pose_notation": "T_parent_child",
        },
        "changes": changes or [],
    }
    if run_stats is not None:
        document["run_stats"] = run_stats
    return document


def _write_json_atomic(document: dict, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)
    return str(path)


def write_dataset_json(document: dict, dataset_root: str | Path) -> str:
    """写出 dataset.json 到 dataset 根目录。"""
    return _write_json_atomic(document, Path(dataset_root) / "dataset.json")


def write_revision_json(document: dict, revision_dir: str | Path) -> str:
    """写出 revision.json 到 prepared_segments/<prep_revision>/。"""
    return _write_json_atomic(document, Path(revision_dir) / "revision.json")


def build_segment_json(
    dataset_path: str,
    span: dict,
    video_results: list[dict] | None = None,
    imu_results: list[dict] | None = None,
    calibration_id: str = "calib_guida_001",
    prep_revision: str = "r0001",
    segment_id: str = "seg_000001",
    session_id: str = "guida_session_001",
    quality_issues: list[dict] | None = None,
    source_assets: list[dict] | None = None,
    profile: str = "guida",
    depth_npz_path: str | None = None,
    depth_results: list[dict] | None = None,
    calibrations: dict | None = None,
    annotation_results: list[dict] | None = None,
    time_series_results: list[dict] | None = None,
    audio_results: list[dict] | None = None,
) -> dict:
    """构建 segment.json 内容。

    每个 video_result 应包含:
      - stream_id, width, height, output_fps, output_frames
      - sample_map_uri (相对于 segment 根目录)
      - frame_id (可选), role (可选, 默认 "observation")

    每个 imu_result 应包含:
      - stream_id, uri (相对于 segment 根目录), rows

    每个 annotation_result 应包含:
      - stream_id, uri, modality, source_asset_id
      - ground_truth_status (可选), operation (可选), sample_map_uri (可选)

    每个 depth_result 应包含:
      - stream_id, uri, dtype, unit, width, height, frames
      - sample_map_uri, source_asset_id, operation

    每个 time_series_result 应包含:
      - stream_id, uri, modality, role, rows
      - source_asset_id, source_topic, operation, fields
    """
    data_dir = Path(dataset_path)
    index_path = data_dir / "index.jsonl"
    meta_path = data_dir / "meta.json"

    duration_ns = span["source_end_ns"] - span["source_start_ns"]

    # source_assets — 由调用方传入或按 guida 默认生成
    if source_assets is None:
        color_path = data_dir / "color_000000.mkv"
        imu_path = data_dir / "imu" / "imu_000000.csv"
        source_assets = [
            {
                "source_asset_id": "raw_color_0",
                "uri": "color_000000.mkv",
                "sha256": sha256_hex(str(color_path)) if color_path.exists() else "",
            },
            {
                "source_asset_id": "raw_index",
                "uri": "index.jsonl",
                "sha256": sha256_hex(str(index_path)) if index_path.exists() else "",
            },
            {
                "source_asset_id": "raw_imu_0",
                "uri": "imu/imu_000000.csv",
                "sha256": sha256_hex(str(imu_path)) if imu_path.exists() else "",
            },
            {
                "source_asset_id": "raw_meta",
                "uri": "meta.json",
                "sha256": sha256_hex(str(meta_path)) if meta_path.exists() else "",
            },
        ]

    source_asset_ids = {
        asset.get("source_asset_id")
        for asset in source_assets
        if asset.get("source_asset_id")
    }

    # ---- 构建 streams 列表 ----
    streams: list[dict] = []

    # RGB 视频流 — 每个 video_result 生成一个 stream entry
    primary_stream_id = _primary_stream_id(profile)
    for vr in (video_results or []):
        stream_id = vr["stream_id"]
        entry = {
            "stream_id": stream_id,
            "role": vr.get("role", "observation"),
            "modality": "rgb",
            "uri": f"data/{stream_id}.mp4",
            "format": "mp4",
            "encoding": "h264",
            "shape": [vr["height"], vr["width"], 3],
            "dtype": "uint8",
            "frame_id": vr.get("frame_id", stream_id),
            "time": {
                "clock_id": "segment",
                "sampling": "cfr",
                "rate_hz": vr["output_fps"],
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": source_assets[0]["source_asset_id"] if source_assets else "raw_color_0",
                "operation": (
                    "trim_transcode_resample_undistort"
                    if vr.get("undistorted")
                    else "trim_transcode_resample"
                ),
                "sample_map_uri": vr.get("sample_map_uri", f"maps/{stream_id}_sample_map.parquet"),
                "undistortion": vr.get(
                    "undistortion",
                    {"status": "not_requested", "detail": "no calibration evaluation"},
                ),
            },
        }
        if primary_stream_id is not None:
            entry["is_primary"] = (
                "true" if stream_id == primary_stream_id else "false"
            )
        if vr.get("preview_uri"):
            entry["preview_uri"] = vr["preview_uri"]
        if vr.get("redacted"):
            entry["redaction"] = {
                "status": "applied",
                "operation": (
                    "face_blur_text_redact"
                    if vr.get("redaction_face") and vr.get("redaction_text")
                    else "face_blur"
                    if vr.get("redaction_face")
                    else "text_redact"
                ),
                "manifest_uri": vr.get("redaction_manifest_uri", ""),
                "stats": vr.get("redaction_stats", {}),
            }
        elif vr.get("redaction_skipped"):
            entry["redaction"] = {
                "status": "skipped",
                "reason": vr.get("redaction_skip_reason", ""),
            }
        streams.append(entry)

    # Guida 等按原始频率无损写出的深度流
    for dr in (depth_results or []):
        streams.append({
            "stream_id": dr["stream_id"],
            "role": "observation",
            "modality": "depth",
            "uri": dr["uri"],
            "format": dr.get("format", "png_sequence"),
            "encoding": dr.get("encoding", "png"),
            "shape": [dr["height"], dr["width"]],
            "dtype": dr["dtype"],
            "unit": dr.get("unit", "unknown"),
            "unit_status": dr.get("unit_status", "unverified"),
            "invalid_value": dr.get("invalid_value"),
            "frame_id": dr.get("frame_id", "depth_optical_frame"),
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
                "rate_hz": dr.get("rate_hz"),
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": dr.get("source_asset_id", "raw_depth_0"),
                "operation": dr.get("operation", "trim_decode_lossless_png"),
                "sample_map_uri": dr["sample_map_uri"],
            },
            "quality_summary": {
                "zero_ratio": dr.get("zero_ratio"),
                "invalid_ratio": dr.get("invalid_ratio"),
            },
        })

    # 兼容现有 Dunjia FFV1 深度输出
    if depth_npz_path is not None and not depth_results:
        streams.append({
            "stream_id": "ego_depth",
            "role": "observation",
            "modality": "depth",
            "uri": "data/ego_depth.mp4",
            "format": "mp4",
            "encoding": "ffv1",
            "dtype": "uint16",
            "frame_id": "depth_optical_frame",
            "time": {
                "clock_id": "segment",
                "sampling": "cfr",
                "rate_hz": 30.0,
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": "raw_mcap" if profile != "guida" else "raw_depth_0",
                "operation": "trim_decode_ffv1",
            },
        })

    # 音频流（遁甲 MCAP 有 audio topic 时）
    for ar in (audio_results or []):
        if not ar.get("uri"):
            continue  # 音频写出失败，跳过
        streams.append({
            "stream_id": ar["stream_id"],
            "role": "observation",
            "modality": "audio",
            "uri": ar["uri"],
            "format": ar.get("format", "wav"),
            "encoding": "pcm_s16le",
            "source_format": ar.get("source_format", "opus"),
            "sample_rate": ar.get("sample_rate", 16000),
            "channels": ar.get("channels", 1),
            "packets": ar.get("packets", 0),
            "duration_s": ar.get("duration_s", 0.0),
            "time": {
                "clock_id": "segment",
                "sampling": "packet",
                "rate_hz": 50.0,  # Opus 20ms 包
                "start_ns": 0,
                "end_ns": duration_ns,
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": ar.get("source_asset_id", "raw_mcap_audio"),
                "source_topic": ar.get("source_topic", "/robot0/sensor/audio"),
                "operation": ar.get("operation", "decode_opus_to_wav_and_sample_map"),
                "sample_map_uri": ar["sample_map_uri"],
            },
        })

    # IMU 流 — 每个 imu_result 生成一个 stream entry
    for ir in (imu_results or []):
        streams.append({
            "stream_id": ir["stream_id"],
            "role": "state",
            "modality": "imu",
            "uri": ir["uri"],
            "format": "parquet",
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
                "timestamp_column": "timestamp_ns",
            },
            "fields": [
                {
                    "name": "linear_acceleration",
                    "shape": [3],
                    "dtype": "float32",
                    "unit": "m/s^2",
                    "frame_id": "imu",
                },
                {
                    "name": "angular_velocity",
                    "shape": [3],
                    "dtype": "float32",
                    "unit": "rad/s",
                    "frame_id": "imu",
                },
            ],
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": ir.get(
                    "source_asset_id",
                    (
                        "raw_imu_0"
                        if "raw_imu_0" in source_asset_ids
                        else source_assets[0]["source_asset_id"]
                        if source_assets
                        else "raw_imu_0"
                    ),
                ),
                "operation": "trim_and_unit_normalize",
            },
        })

    # 通用时序流（UMI 磁编码器、后续 VIO 等）
    for tr in (time_series_results or []):
        stream_entry = {
            "stream_id": tr["stream_id"],
            "role": tr.get("role", "sensor"),
            "modality": tr["modality"],
            "uri": tr["uri"],
            "format": "parquet",
            "frame_id": tr.get("frame_id"),
            "unit": tr.get("unit", "unknown"),
            "semantic_status": tr.get(
                "semantic_status",
                "raw_unverified",
            ),
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
                "rate_hz": tr.get("rate_hz"),
                "timestamp_column": "timestamp_ns",
            },
            "fields": tr.get("fields", []),
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": tr.get(
                    "source_asset_id",
                    "raw_mcap",
                ),
                "source_topic": tr["source_topic"],
                "source_field": tr.get("source_field"),
                "operation": tr.get(
                    "operation",
                    "trim_preserve_raw_value",
                ),
            },
            "sample_count": tr.get("rows", 0),
        }
        for contract_key in (
            "coordinate_contract",
            "continuity",
            "continuity_group_count",
        ):
            if contract_key in tr:
                stream_entry[contract_key] = tr[contract_key]
        streams.append(stream_entry)

    # 标注流 — 每个 annotation_result 生成一个 stream entry
    for ar in (annotation_results or []):
        streams.append({
            "stream_id": ar["stream_id"],
            "role": "annotation",
            "modality": ar.get("modality", "hand_object_detection"),
            "uri": ar["uri"],
            "format": "parquet",
            "ground_truth_status": ar.get("ground_truth_status", "model_generated"),
            "time": {
                "clock_id": "segment",
                "sampling": "sparse",
                "timestamp_column": "timestamp_ns",
            },
            "origin": {
                "kind": "imported_model_annotation",
                "source_asset_id": ar.get("source_asset_id", "raw_hand_object_pkl"),
                "operation": ar.get("operation", "safe_pickle_parse_and_frame_remap"),
                "sample_map_uri": ar.get("sample_map_uri", "maps/ego_rgb_sample_map.parquet"),
            },
        })

    segment = {
        "zpds_version": "0.1.0",
        "prep_revision": prep_revision,
        "segment_id": segment_id,
        "source_type": "ego",

        "source_session": {
            "session_id": session_id,
            "session_uri": str(data_dir.resolve()),
        },

        "source_assets": source_assets,

        "timeline": {
            "start_ns": 0,
            "end_ns": duration_ns,
            "continuous": True,
        },

        "source_span": {
            "source_clock_id": "device_clock",
            "start_ns": span["source_start_ns"],
            "end_ns": span["source_end_ns"],
        },

        "streams": streams,

        "calibration_uri": "calibration/calibration.json",

        "quality": {
            "status": "warn" if (quality_issues and len(quality_issues) > 0) else "pass",
            "issues": quality_issues or [],
        },
    }

    return segment


def write_segment_json(segment: dict, output_dir: str) -> str:
    """写出 segment.json。

    Returns:
        输出文件路径
    """
    seg_dir = Path(output_dir)
    seg_dir.mkdir(parents=True, exist_ok=True)
    output_path = seg_dir / "segment.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(segment, f, indent=2, ensure_ascii=False)
    return str(output_path)
