"""
ZPDS 批量 Prepared Segment 生成。

读取 segment_candidates.json，对每个候选区间：
  ① 裁剪并转码 RGB 视频
  ② 生成 {stream_id}_sample_map.parquet
  ③ 规范化 IMU
  ④ 提取 calibration.json（共享，只做一次）
  ⑤ 生成 segment.json
  ⑥ 写出后验证

用法:
    # 墨现 (默认)
    python batch_prepare.py
    python batch_prepare.py --candidates output/segment_candidates.json

    # 遁甲
    python batch_prepare.py --profile dunjia --dataset session.mcap \
        --candidates output_dunjia/segment_candidates.json \
        --output prepared_segments_dunjia/
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from segment.annotation_normalizer import normalize_hand_objects, write_annotation_parquet
from segment.audio_writer import write_audio_stream
from segment.calibration import (
    extract_calibration,
    extract_calibration_from_mcap,
    write_calibration,
)
from segment.depth_writer import write_depth_stream
from segment.epic_fields_calibration import (
    load_epic_fields_calibration,
    missing_epic_fields_calibration,
)
from segment.image_undistorter import plan_undistortion
from segment.imu_normalizer import normalize_imu_df, write_imu
from segment.magnetic_encoder_writer import write_magnetic_encoder_stream
from segment.mask_normalizer import normalize_masks, write_mask_parquet
from segment.preview import create_preview
from segment.sample_map import (
    generate_sample_map,
    generate_sample_map_from_timestamps,
    write_sample_map,
)
from segment.segment_writer import (
    build_dataset_json,
    build_revision_json,
    build_segment_json,
    package_version,
    sha256_hex,
    write_dataset_json,
    write_revision_json,
    write_segment_json,
)
from segment.validator import (
    validate_segment,
    write_annotation_validation_report,
    write_validation_report,
)
from segment.video_transcoder import transcode_rgb
from segment.vio_pose_writer import write_vio_pose_stream

# ============================================================
# 配置
# ============================================================
DATASET = "E:/datasets/egos/墨现"
CANDIDATES_PATH = "output/segment_candidates.json"
CONFIG_PATH = "config.yaml"
OUTPUT_ROOT = "output"  # 子目录在运行时按 profile 拼接 prepared_segments
REVISION = "r0001"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_dotenv(path: str | Path = ".env") -> None:
    """把 .env 的 KEY=VALUE 注入进程环境；已存在的变量不覆盖。

    与 zpds_prepare.main 的 _load_dotenv 保持一致（LLM API key 等）。
    """
    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _sha256_manifest(paths: list[Path], root: Path) -> str:
    """计算文件集合的确定性 manifest 哈希。"""
    digest = hashlib.sha256()
    for path in sorted(paths):
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_hex(str(path)).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _build_guida_source_assets(dataset_path: str, session) -> list[dict]:
    """构建 Guida Raw 资产清单，包含正式深度来源。"""
    root = Path(dataset_path)
    assets: list[dict] = []
    standard_assets = [
        ("raw_color_0", root / "color_000000.mkv", "color_000000.mkv"),
        ("raw_index", root / "index.jsonl", "index.jsonl"),
        ("raw_imu_0", root / "imu" / "imu_000000.csv", "imu/imu_000000.csv"),
        ("raw_meta", root / "meta.json", "meta.json"),
    ]
    for asset_id, path, uri in standard_assets:
        assets.append({
            "source_asset_id": asset_id,
            "uri": uri,
            "sha256": sha256_hex(str(path)) if path.is_file() else "",
        })

    depth_stream = session.depth_streams.get("ego_depth")
    if depth_stream is None:
        return assets

    source_files = list(depth_stream.source_files)
    if depth_stream.source_kind == "image_sequence":
        parent = source_files[0].parent
        try:
            uri = parent.relative_to(root).as_posix()
        except ValueError:
            uri = str(parent)
        assets.append({
            "source_asset_id": "raw_depth_0",
            "uri": uri,
            "media_type": "image/png-sequence",
            "sha256": _sha256_manifest(source_files, root),
            "hash_kind": "sha256_path_content_manifest",
            "member_count": len(source_files),
        })
    else:
        depth_hash = (
            sha256_hex(str(source_files[0]))
            if len(source_files) == 1
            else _sha256_manifest(source_files, root)
        )
        assets.append({
            "source_asset_id": "raw_depth_0",
            "uri": (
                source_files[0].relative_to(root).as_posix()
                if source_files[0].is_relative_to(root)
                else str(source_files[0])
            ),
            "media_type": "video/x-matroska",
            "sha256": depth_hash,
            "hash_kind": (
                "sha256"
                if len(source_files) == 1
                else "sha256_path_content_manifest"
            ),
            "members": [
                path.relative_to(root).as_posix()
                if path.is_relative_to(root)
                else str(path)
                for path in source_files
            ],
        })
    return assets


def _crop_hands_artifacts(
    source_dir: Path,
    target_dir: Path,
    *,
    start_ns: int,
    end_ns: int,
    start_frame: int,
    end_frame: int,
    source_fps: float,
) -> dict | None:
    """把 main.py 的整段 Hands 产物裁切为 segment 级产物（引用式接入）。

    main.py 在 ``{output}/prepared_segments/r0001/seg_000001/hands`` 产出整段
    Hands 数据；这里按候选区间 [start_ns, end_ns) 裁切后写入 ``{seg_dir}/hands/``。
    时间轴说明：

    - ``hands_2d.parquet``：timestamp_ns 是真实 MKV 时间戳，与 candidates 同轴，
      直接按 [start_ns, end_ns) 过滤（同帧多手行都保留）。
    - ``hand_cleaning_frames.parquet``：timestamp_ns 是帧号/fps 重算的
      （cleaning.py 定义），不能按时间戳过滤，只能按 source_frame_index（源解码
      帧号）裁切；裁切后 timestamp_ns 重算为相对 segment 轴（与输出视频对齐）。
    - ``hand_cleaning_report.json``：重建 segment-level 统计；原始整段报告保留
      为 ``hand_cleaning_report.original.json``。

    返回 segment.json 登记信息；源目录或产物缺失时返回 None。
    """
    required_files = [
        "hands_2d.parquet",
        "hand_cleaning_frames.parquet",
        "hand_cleaning_report.json",
    ]
    if not source_dir.is_dir() or any(
        not (source_dir / name).is_file() for name in required_files
    ):
        return None

    target_dir.mkdir(parents=True, exist_ok=True)

    # ---- hands_2d.parquet：按真实 MKV 时间戳过滤（与 candidates 同轴）----
    # 过滤后 rebase 到 segment-relative 轴（segment.json timeline.start_ns=0），
    # 与 hand_cleaning_frames 的时间戳约定一致；消费者无需再换算。
    hands = pd.read_parquet(source_dir / "hands_2d.parquet")
    hands = hands[
        (hands["timestamp_ns"] >= start_ns) & (hands["timestamp_ns"] < end_ns)
    ]
    hands["timestamp_ns"] = (hands["timestamp_ns"] - start_ns).astype("int64")
    hands.to_parquet(target_dir / "hands_2d.parquet", index=False)

    # ---- hand_cleaning_frames.parquet：按源解码帧号裁切，时间戳重算为相对轴 ----
    frames = pd.read_parquet(source_dir / "hand_cleaning_frames.parquet")
    frames = frames[
        (frames["source_frame_index"] >= start_frame)
        & (frames["source_frame_index"] < end_frame)
    ].copy()
    frames["timestamp_ns"] = (
        (frames["source_frame_index"] - start_frame)
        / source_fps
        * 1_000_000_000
    ).astype("int64")
    frames.to_parquet(target_dir / "hand_cleaning_frames.parquet", index=False)

    # ---- hand_cleaning_report.json：重建 segment-level，原始报告保留 ----
    report = json.loads(
        (source_dir / "hand_cleaning_report.json").read_text(encoding="utf-8")
    )
    frame_count = len(frames)
    excluded = (
        int(frames["is_excluded"].sum()) if "is_excluded" in frames.columns else 0
    )
    kept = frame_count - excluded

    def _crop_spans(spans: list[dict]) -> list[dict]:
        cropped: list[dict] = []
        for span in spans:
            lo = max(int(span["start_frame"]), start_frame)
            hi = min(int(span["end_frame"]), end_frame)
            if lo < hi:
                cropped.append({
                    **span,
                    "start_frame": lo - start_frame,
                    "end_frame": hi - start_frame,
                    "start_timestamp_ns": int(
                        (lo - start_frame) / source_fps * 1_000_000_000
                    ),
                    "end_timestamp_ns": int(
                        (hi - start_frame) / source_fps * 1_000_000_000
                    ),
                    "duration_s": round((hi - lo) / source_fps, 6),
                })
        return cropped

    cropped_report = dict(report)
    cropped_report["source"] = {
        **report.get("source", {}),
        "advertised_frame_count": frame_count,
        "decoded_frame_count": frame_count,
        "duration_s": round(frame_count / source_fps, 6),
        "fps": source_fps,
    }
    cropped_report["summary"] = {
        **report.get("summary", {}),
        "input_frames": frame_count,
        "excluded_frames": excluded,
        "kept_frames": kept,
        "input_duration_s": round(frame_count / source_fps, 6),
        "kept_duration_s": round(kept / source_fps, 6),
        "overall_disposition": (
            "reject" if not kept else "split" if excluded else "keep"
        ),
    }
    cropped_report["excluded_spans"] = _crop_spans(report.get("excluded_spans", []))
    cropped_report["kept_spans"] = _crop_spans(report.get("kept_spans", []))
    if "artifacts" in cropped_report:
        cropped_report["artifacts"] = {
            **cropped_report["artifacts"],
            "frame_metrics_uri": "hand_cleaning_frames.parquet",
        }
    provenance = cropped_report.get("provenance", {})
    provenance["hands_parquet_uri"] = "hands_2d.parquet"
    provenance["hands_parquet_sha256"] = sha256_hex(
        str(target_dir / "hands_2d.parquet")
    )
    cropped_report["provenance"] = provenance
    cropped_report["segmentization"] = {
        "applied": True,
        "crop": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
        "source_report_uri": "hand_cleaning_report.original.json",
    }
    (target_dir / "hand_cleaning_report.json").write_text(
        json.dumps(cropped_report, ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    shutil.copy2(
        source_dir / "hand_cleaning_report.json",
        target_dir / "hand_cleaning_report.original.json",
    )

    return {
        "uri": "hands/hands_2d.parquet",
        "schema": "zpds.hands.v1",
        "video_stream_id": (
            str(hands["video_stream_id"].iloc[0]) if len(hands) else ""
        ),
        "rows": len(hands),
        "frames_uri": "hands/hand_cleaning_frames.parquet",
        "frames": frame_count,
        "report_uri": "hands/hand_cleaning_report.json",
        "original_report_uri": "hands/hand_cleaning_report.original.json",
        "excluded_frames": excluded,
        "kept_frames": kept,
        "overall_disposition": cropped_report["summary"]["overall_disposition"],
        "cropped": True,
        "timebase": "segment_clock",
        "source_span": {"start_ns": start_ns, "end_ns": end_ns},
        "crop": {
            "start_ns": start_ns,
            "end_ns": end_ns,
            "start_frame": start_frame,
            "end_frame": end_frame,
        },
    }


def generate_segment(
    dataset_path: str,
    source_start_ns: int,
    source_end_ns: int,
    segment_id: str,
    output_dir: str,
    session,
    calibration: dict,
    cfg: dict,
    session_id: str = "guida_session_001",
    prep_revision: str = "r0001",
    quality_issues: list[dict] | None = None,
    profile: str = "guida",
    source_assets: list[dict] | None = None,
    depth_npz_path: str | None = None,
    experience_dir: str | None = None,
    experience_version: str | None = None,
    privacy_enabled: bool = False,
    privacy_reset_frames: set[int] | None = None,
    hands_source: dict | None = None,
) -> dict:
    """为单个候选区间生成完整 Prepared Segment。

    遍历 session 中所有 video_streams / imu_streams，
    文件名由各流的 stream_id 决定，不再硬编码。
    """
    target_fps = cfg["output"]["target_fps"]
    duration_ns = source_end_ns - source_start_ns

    pv = session.primary_video
    span = {
        "source_start_ns": source_start_ns,
        "source_end_ns": source_end_ns,
        "duration_s": duration_ns / 1_000_000_000,
        "total_frames_in_span": sum(
            1 for f in pv.index_frames
            if source_start_ns <= f["timestamp_ns"] <= source_end_ns
        ),
        "reason": {"start": "from_candidate", "end": "from_candidate"},
        "trimmed_head_frames": 0,
        "trimmed_tail_frames": 0,
    }

    # ---- ① sample_map (无需等转码，直接用 session 生成) ----
    video_meta: list[dict] = []
    for stream_id, vs in session.video_streams.items():
        if profile in ("dunjia", "umi", "epic"):
            sample_map = generate_sample_map_from_timestamps(
                timestamps_ns=vs.timestamps_ns,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
                target_fps=target_fps,
            )
        else:
            sample_map = generate_sample_map(
                index_frames=vs.index_frames,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
                target_fps=target_fps,
            )
        write_sample_map(sample_map, output_dir, stream_id)

        output_mp4 = str(Path(output_dir) / "data" / f"{stream_id}.mp4")
        video_meta.append({
            "stream_id": stream_id,
            "width": vs.width,
            "height": vs.height,
            "output_fps": target_fps,
            "output_mp4": output_mp4,
            "sample_map_uri": f"maps/{stream_id}_sample_map.parquet",
        })

    # ---- ② IMU ----
    imu_results = []
    for stream_id, imu_s in session.imu_streams.items():
        imu = normalize_imu_df(
            imu=imu_s.dataframe,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
        )
        write_imu(imu, output_dir, stream_id)
        imu_results.append({
            "stream_id": stream_id,
            "uri": f"data/{stream_id}.parquet",
            "rows": len(imu),
        })

    # ---- ③ UMI 机器人时序：保留原始值、位姿和 MCAP 时钟 ----
    time_series_results = []
    for stream in session.time_series_streams.values():
        if stream.modality == "magnetic_encoder":
            time_series_results.append(
                write_magnetic_encoder_stream(
                    stream=stream,
                    output_dir=output_dir,
                    source_start_ns=source_start_ns,
                    source_end_ns=source_end_ns,
                )
            )
        elif stream.modality == "vio_pose":
            time_series_results.append(
                write_vio_pose_stream(
                    stream=stream,
                    output_dir=output_dir,
                    source_start_ns=source_start_ns,
                    source_end_ns=source_end_ns,
                )
            )

    # ---- ④ 深度流: 保留原始频率并无损写出 ----
    depth_results = []
    for depth_stream in session.depth_streams.values():
        depth_results.append(
            write_depth_stream(
                stream=depth_stream,
                output_dir=output_dir,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
            )
        )

    # ---- ④b 音频流: Opus → WAV 标准化写出（遁甲有音频 topic 时）----
    audio_results = []
    for audio_stream in session.audio_streams.values():
        try:
            audio_results.append(
                write_audio_stream(
                    packets=audio_stream.packets,
                    output_dir=output_dir,
                    source_start_ns=source_start_ns,
                    source_end_ns=source_end_ns,
                    stream_id=audio_stream.stream_id,
                )
            )
        except Exception as exc:  # noqa: BLE001 - 音频失败不阻断整段
            audio_results.append({
                "stream_id": audio_stream.stream_id,
                "uri": None,
                "error": str(exc),
            })

    # ---- ⑤ 标注: 标准化 + 写出 ----
    annotation_results = []
    for stream_id, ann_s in session.annotation_streams.items():
        if not ann_s.records:
            continue

        # 使用第一个视频流的 sample_map
        first_vm = video_meta[0] if video_meta else None
        if first_vm is None:
            continue

        sm_path = Path(output_dir) / first_vm.get("sample_map_uri", "maps/ego_rgb_sample_map.parquet")
        if not sm_path.exists():
            continue
        sample_map = pd.read_parquet(str(sm_path))

        vs = session.video_streams[ann_s.source_video_stream_id]

        if ann_s.annotation_type == "hand_object_detection":
            df = normalize_hand_objects(
                annotation_stream=ann_s,
                video_timestamps_ns=vs.timestamps_ns,
                sample_map=sample_map,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
                video_width=vs.width,
                video_height=vs.height,
            )
            write_annotation_parquet(df, output_dir, stream_id)
            annotation_results.append({
                "stream_id": stream_id,
                "uri": f"annotations/{stream_id}.parquet",
                "modality": "hand_object_detection",
                "source_asset_id": f"raw_{stream_id}_pkl",
                "ground_truth_status": "model_generated",
                "operation": "safe_pickle_parse_and_frame_remap",
                "sample_map_uri": first_vm.get("sample_map_uri", "maps/ego_rgb_sample_map.parquet"),
                "rows": len(df),
            })

        elif ann_s.annotation_type == "instance_segmentation":
            df = normalize_masks(
                annotation_stream=ann_s,
                video_timestamps_ns=vs.timestamps_ns,
                sample_map=sample_map,
                source_start_ns=source_start_ns,
                source_end_ns=source_end_ns,
                video_width=vs.width,
                video_height=vs.height,
            )
            write_mask_parquet(df, output_dir, stream_id)
            annotation_results.append({
                "stream_id": stream_id,
                "uri": f"annotations/{stream_id}.parquet",
                "modality": "instance_segmentation",
                "source_asset_id": f"raw_{stream_id}_pkl",
                "ground_truth_status": "model_generated",
                "operation": "safe_pickle_parse_rle_normalize",
                "sample_map_uri": first_vm.get("sample_map_uri", "maps/ego_rgb_sample_map.parquet"),
                "rows": len(df),
            })

    # ---- ⑥ 转码 + G16 去畸变 ----
    video_results = []
    undistortion_coverage = calibration.setdefault("undistortion", {"streams": {}})
    for vm in video_meta:
        vs = session.video_streams[vm["stream_id"]]
        undistortion = plan_undistortion(
            calibration,
            vm["stream_id"],
            width=vs.width,
            height=vs.height,
        )
        undistortion_detail = {
            "status": undistortion.status,
            "detail": undistortion.detail,
            "operation": {
                "applied": "undistort",
                "identity": "identity",
            }.get(undistortion.status, "preserve_original"),
            "calibration_source": calibration.get("source", {}).get(
                "reference_url",
                calibration.get("source", {}).get("uri", ""),
            ),
        }
        undistortion_coverage["streams"][vm["stream_id"]] = undistortion_detail
        vr = transcode_rgb(
            source_video=vs.video_path,
            output_mp4=vm["output_mp4"],
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
            index_frames=vs.index_frames,
            target_fps=target_fps,
            frame_transform=undistortion.frame_transform,
            # 脱敏必须基于干净转码（已有产物可能是上次脱敏的重编码版）
            use_cache=not privacy_enabled,
        )
        vr["stream_id"] = vm["stream_id"]
        vr["sample_map_uri"] = vm["sample_map_uri"]
        vr["undistorted"] = undistortion.status == "applied"
        vr["undistortion"] = undistortion_detail
        video_results.append(vr)

    # ---- ⑥.8 隐私脱敏：对转码产物原地脱敏（人脸模糊 + 文本遮挡） ----
    # 训练集只出脱敏版；QC 在 main.py 阶段用原始视频判断，不受影响。
    # 脱敏先于预览执行，确保 preview 也是脱敏后的画面。
    redacted_count = 0
    if privacy_enabled:
        from zpds.privacy.segment_redaction import redact_segment_videos

        redacted_count = redact_segment_videos(
            video_meta,
            video_results,
            Path(output_dir),
            profile,
            reset_frames=privacy_reset_frames,
        )
        print(f"  隐私脱敏: {redacted_count} 个视频流")

    # ---- ⑥.5 前端预览压缩：保留原视频，另存 <stream_id>_preview.mp4 ----
    preview_cfg = cfg.get("preview", {})
    preview_count = 0
    if preview_cfg.get("enabled", True):
        for vm, vr in zip(video_meta, video_results):
            stream_id = vr["stream_id"]
            preview_path = (
                Path(output_dir) / "data" / f"{stream_id}_preview.mp4"
            )
            create_preview(
                vm["output_mp4"],
                preview_path,
                max_width=int(preview_cfg.get("max_width", 1280)),
                crf=int(preview_cfg.get("crf", 28)),
                preset=str(preview_cfg.get("preset", "veryfast")),
            )
            vr["preview_uri"] = f"data/{stream_id}_preview.mp4"
            preview_count += 1

    # ---- ⑥.7 Hands 辅助产物：main 整段产物 → segment 级裁切 ----
    hands_results = None
    if hands_source is not None:
        hands_results = _crop_hands_artifacts(
            source_dir=hands_source["source_dir"],
            target_dir=Path(output_dir) / "hands",
            start_ns=hands_source["start_ns"],
            end_ns=hands_source["end_ns"],
            start_frame=hands_source["start_frame"],
            end_frame=hands_source["end_frame"],
            source_fps=hands_source["source_fps"],
        )
        if hands_results:
            print(
                f"  Hands: {hands_results['rows']} 手行 / "
                f"{hands_results['frames']} 帧, "
                f"excluded={hands_results['excluded_frames']} "
                f"({hands_results['overall_disposition']})"
            )
        else:
            print("  Hands: 源产物不完整，跳过")

    # ---- ⑦ 标定与 segment.json ----
    write_calibration(calibration, output_dir)
    segment = build_segment_json(
        dataset_path=dataset_path,
        span=span,
        video_results=video_results,
        imu_results=imu_results,
        calibration_id=calibration["calibration_id"],
        prep_revision=prep_revision,
        segment_id=segment_id,
        session_id=session_id,
        quality_issues=quality_issues,
        source_assets=source_assets,
        profile=profile,
        depth_npz_path=depth_npz_path,
        depth_results=depth_results,
        calibrations=calibration.get("calibrations", None),
        annotation_results=annotation_results if annotation_results else None,
        time_series_results=time_series_results if time_series_results else None,
        audio_results=audio_results if audio_results else None,
        hands_results=hands_results,
    )
    write_segment_json(segment, output_dir)

    # ---- ⑧ 写出后验证 ----
    validation = validate_segment(output_dir)
    write_validation_report(validation, output_dir)
    write_annotation_validation_report(validation, output_dir)

    experience_manifest = None
    if experience_dir is not None and validation["status"] != "fail":
        from zpds.annotation.importer import import_segment_annotations

        experience_manifest = import_segment_annotations(
            output_dir,
            experience_dir,
            experience_version=experience_version,
        )

    return {
        "segment_id": segment_id,
        "status": validation["status"],
        "duration_s": duration_ns / 1_000_000_000,
        "video_streams": [vm["stream_id"] for vm in video_meta],
        "rgb_frames": video_results[0]["output_frames"] if video_results else 0,
        "preview_count": preview_count,
        "imu_samples": sum(ir["rows"] for ir in imu_results),
        "depth_frames": sum(dr["frames"] for dr in depth_results),
        "audio_packets": sum(
            ar.get("packets", 0) for ar in audio_results
        ),
        "time_series_samples": sum(
            result["rows"]
            for result in time_series_results
        ),
        "annotation_rows": sum(
            ar.get("rows", 0) for ar in annotation_results
        ) if annotation_results else 0,
        "experience_manifest": experience_manifest,
        "checks": validation["checks"],
        "errors": validation["errors"],
    }


def _verify_source_frames(
    *,
    primary_video,
    candidates_doc: dict,
    dataset_path: str,
) -> list[str]:
    """比对源视频实际帧数与 candidates 声明的时长（duration_s × fps）。

    防止 -d 传错数据源（例如默认旧数据 983 帧 vs 新数据 1278 帧），
    这类错误跑完整轮后 validation 才暴露。容差 ±2 帧。
    返回错误信息列表，空列表表示一致。
    """
    errors: list[str] = []
    duration_s = float(candidates_doc.get("source_duration_s") or 0.0)
    fps = getattr(primary_video, "fps", None) or 30.0
    if duration_s <= 0 or fps <= 0:
        return errors
    expected = round(duration_s * fps)
    actual = int(primary_video.frame_count)
    if abs(actual - expected) > 2:
        errors.append(
            f"源视频实际 {actual} 帧（{actual / fps:.1f}s @{fps:.1f}fps）"
            f"与候选声明时长 {duration_s:.3f}s（≈{expected} 帧）不一致，"
            f"疑似 --dataset/-d 传错数据源: {dataset_path}"
        )
    return errors


def step_header(title: str):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description="ZPDS 批量 Prepared Segment 生成"
    )
    parser.add_argument(
        "--candidates", "-c",
        default=None,
        help="segment_candidates.json 路径 (默认: output/moxian/ 或 output/dunjia/)",
    )
    parser.add_argument(
        "--reviewed-report",
        default=None,
        help="平台审核完成后返回的统一 quality_report.json；提供后将忽略默认候选文件",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=None,
        help="数据集路径 (必填; 墨现: 目录; 遁甲/UMI: .mcap 文件)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出根目录 (默认: prepared_segments/moxian/ 或 prepared_segments/dunjia/)",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="[Dunjia] H264 重建缓存目录（默认: 输出目录/.cache）",
    )
    parser.add_argument(
        "--epic-fields-root",
        default=None,
        help="[EPIC] EPIC-Fields JSON 根目录；未提供或未覆盖的视频保持原 RGB",
    )
    parser.add_argument(
        "--profile", "-p",
        default="guida",
        choices=["guida", "dunjia", "umi", "epic"],
        help="数据源 profile (默认: guida)",
    )
    parser.add_argument(
        "--with-privacy",
        action="store_true",
        help="对转码后的视频执行隐私脱敏（人脸模糊 + 文本遮挡），"
             "训练集只出脱敏版",
    )
    parser.add_argument(
        "--experience-dir",
        default=None,
        help="可选：将已声明的 Prepared 标注导入此 Experience 目录",
    )
    parser.add_argument(
        "--experience-version",
        default=None,
        help="Experience 版本（默认使用 --experience-dir 的目录名）",
    )
    args = parser.parse_args()
    if args.reviewed_report and args.candidates:
        parser.error("--reviewed-report 与 --candidates 不能同时使用")

    # 项目根绝对路径（batch_prepare 可能从任意 cwd 启动）
    _load_dotenv(Path(__file__).resolve().parent / ".env")
    profile = args.profile

    # ---- 加载配置和候选方案 ----
    cfg = load_config(args.config)

    # 默认 candidates 路径按 profile 分子目录；正式审核流程直接使用平台
    # 返回的同一份统一报告。
    if args.reviewed_report:
        candidates_path = Path(args.reviewed_report)
    elif args.candidates is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi", "epic": "epic"}
        subdir = profile_subdirs.get(profile, profile)
        candidates_path = Path("output") / subdir / "segment_candidates.json"
    else:
        candidates_path = Path(args.candidates)

    # ---- 数据源路径必填（防止静默读默认旧数据，跑完整轮才发现） ----
    if not args.dataset:
        print("错误: --dataset/-d 必填（数据源路径）")
        print("  墨现 guida: 数据目录（含 color/ 与 index.jsonl）")
        print("  遁甲 dunjia / UMI: .mcap 文件")
        return 1
    dataset_path = args.dataset

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi"}
        subdir = profile_subdirs.get(profile, profile)
        output_root = Path("output") / subdir / "prepared_segments"
    else:
        output_root = Path(args.output)

    # 中文路径 fail-fast：深度 PNG / MP4 由 cv2.imread/imwrite 处理，
    # 不支持非 ASCII 路径（静默失败），命中立即报错而不是跑一段再挂。
    if not output_root.as_posix().isascii():
        print(f"错误: 输出目录必须为纯 ASCII 路径（cv2 不支持中文路径）: {output_root}")
        print("请改用英文目录，例如: --output output/taodai2/prepared_segments")
        return 1

    # ZPDS dataset 结构：prepared_segments/<prep_revision>/<segment_id>/
    prep_revision = REVISION
    revision_root = output_root / prep_revision

    if not candidates_path.exists():
        print(f"错误: 审核报告或候选文件不存在: {candidates_path}")
        print(f"请先运行质检并完成平台审核: {dataset_path} ({profile})")
        return 1

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)
    analysis_artifacts: dict = candidates_doc.get("analysis_artifacts", {}) or {}

    # ---- 隐私脱敏：场景边界（scene_proposals.parquet）作为强制检测帧 ----
    # 场景切换后画面布局剧变，KLT 传播必然失效，在这些帧重新完整检测。
    # 注意：scene run 的时间戳是"帧号/fps 重算"的（run_summary 中 start_ns=0），
    # 与 candidates 的原始 MKV 时间戳不同轴 → 边界直接换算为完整视频帧号。
    # scene 产物位置与 main.py 一致：{output_root}/{prep_revision}/seg_000001/scene
    scene_boundary_frames: list[int] = []
    if args.with_privacy:
        # 优先 candidates JSON 顶层 analysis_artifacts.scene（main.py 新布局
        # {--output}/analysis/scene/，uri 相对 candidates 所在目录）；
        # 无声明时回退旧布局
        # （{output_root}/{prep_revision}/seg_000001/scene 与 {output_root.parent}/scene）。
        scene_art = analysis_artifacts.get("scene", {}) or {}
        scene_dir: Path | None = None
        if scene_art.get("uri"):
            candidate_scene_dir = (candidates_path.parent / scene_art["uri"]).parent
            if (candidate_scene_dir / "scene_proposals.parquet").is_file():
                scene_dir = candidate_scene_dir
        if scene_dir is None:
            legacy_dir = output_root / prep_revision / "seg_000001" / "scene"
            if (legacy_dir / "scene_proposals.parquet").is_file():
                scene_dir = legacy_dir
            else:
                legacy_dir2 = output_root.parent / "scene"
                if (legacy_dir2 / "scene_proposals.parquet").is_file():
                    scene_dir = legacy_dir2
        if scene_dir is not None:
            scene_file = scene_dir / "scene_proposals.parquet"
            if scene_file.is_file():
                try:
                    scene_fps = 30.0
                    summary_file = scene_dir / "run_summary.json"
                    if summary_file.is_file():
                        scene_fps = float(
                            json.loads(
                                summary_file.read_text(encoding="utf-8")
                            ).get("fps", 30.0)
                        )
                    scene_df = pd.read_parquet(str(scene_file))
                    if not scene_df.empty and "start_ns" in scene_df.columns:
                        scene_boundary_frames = sorted(
                            int(round(v / 1_000_000_000 * scene_fps))
                            for v in scene_df["start_ns"].tolist()
                        )
                        print(f"场景边界: {len(scene_boundary_frames)} 个帧号"
                              f"（脱敏强制检测帧，来自 {scene_file.name}）")
                except Exception as exc:  # noqa: BLE001 - 边界读取失败不阻断脱敏
                    print(f"[warn] 读取场景边界失败，脱敏退化为纯间隔采样: {exc}")
            else:
                print("[warn] 未找到场景产物 (scene_proposals.parquet)，"
                      "脱敏使用纯间隔采样")

    reviewed_report = None
    if args.reviewed_report:
        from zpds_prepare.review import (
            ReviewValidationError,
            build_reviewed_candidates_document,
            load_reviewed_report,
            validate_reviewed_report,
        )

        try:
            reviewed_report = load_reviewed_report(candidates_path)
            source_for_validation = dataset_path
            if profile == "epic" and dataset_path.endswith(".json"):
                record = json.loads(Path(dataset_path).read_text(encoding="utf-8"))
                source_for_validation = record.get("video_uri") or dataset_path
            validate_reviewed_report(
                reviewed_report,
                profile=profile,
                dataset_path=source_for_validation,
            )
            segment_cfg = cfg.get("segment", {})
            candidates_doc = build_reviewed_candidates_document(
                reviewed_report,
                min_duration_ns=int(
                    float(segment_cfg.get("min_duration_s", 1.0)) * 1_000_000_000
                ),
                max_duration_ns=int(
                    float(segment_cfg.get("max_duration_s", 120.0)) * 1_000_000_000
                ),
            )
        except (ReviewValidationError, OSError, ValueError) as exc:
            print(f"错误: 审核报告不可执行:\n{exc}")
            return 1

        output_root.mkdir(parents=True, exist_ok=True)
        approved_candidates_path = output_root / "approved_segment_candidates.json"
        approved_candidates_path.write_text(
            json.dumps(candidates_doc, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"审核后候选方案: {approved_candidates_path}")
    else:
        with open(candidates_path, "r", encoding="utf-8") as f:
            candidates_doc = json.load(f)

    candidates = candidates_doc.get("segments", [])
    default_session_ids = {
        "guida": "guida_session_001",
        "dunjia": "dunjia_session_001",
        "umi": "umi_session_001",
    }
    source_session_id = candidates_doc.get(
        "source_session_id",
        default_session_ids.get(profile, f"{profile}_session_001"),
    )

    if not candidates:
        print("没有候选 Segment，退出。")
        return 0

    # ---- Hands 辅助产物（main.py source-level staging，引用式接入）----
    # main.py 写入 {--output}/analysis/hands/ 并在 segment_candidates.json 顶层
    # analysis_artifacts.hands 声明 uri（相对 candidates 所在目录）；
    # 无 manifest 的旧 candidates 回退到旧布局
    # （{candidates.parent}/prepared_segments/r0001/seg_000001/hands）。
    hands_source_dir: Path | None = None
    hands_art = analysis_artifacts.get("hands", {}) or {}
    if hands_art.get("uri"):
        hands_source_dir = (candidates_path.parent / hands_art["uri"]).parent
    else:
        legacy_hands_dir = (
            candidates_path.parent
            / "prepared_segments"
            / REVISION
            / "seg_000001"
            / "hands"
        )
        if legacy_hands_dir.is_dir():
            hands_source_dir = legacy_hands_dir
    if hands_source_dir is not None and hands_source_dir.is_dir():
        print(f"Hands 源产物: {hands_source_dir}")
    else:
        hands_source_dir = None
        print("Hands 源产物: 未找到（跳过；需 main.py --with-hands 先产出）")

    print(f"Profile:      {profile}")
    print(f"数据源:       {dataset_path}")
    print(f"候选方案:     {candidates_path}")
    print(f"候选数量:     {len(candidates)}")
    print(f"Session ID:   {source_session_id}")

    # ---- 预加载共享资源 ----
    step_header("预加载共享资源")

    if profile == "dunjia":
        from zpds_prepare.readers import dunjia_reader as dr

        # 统一读取 Session
        print("  读取 MCAP Session ...")
        dunjia_depth = cfg.get("dunjia", {}).get("depth", {})
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir
            else output_root / ".cache"
        )
        session = dr.read_session(
            dataset_path,
            cache_dir=cache_dir,
            include_depth=bool(dunjia_depth.get("enabled", True)),
            require_depth=bool(dunjia_depth.get("required", True)),
        )
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps_ns = pv.timestamps_ns
        _source_errors = _verify_source_frames(
            primary_video=pv, candidates_doc=candidates_doc, dataset_path=dataset_path
        )
        if _source_errors:
            for _err in _source_errors:
                print(f"错误: {_err}")
            print("中止处理，请核对 --dataset/-d。")
            return 1
        print(f"  camera0: {pv.frame_count} 帧, "
              f"时间范围: {timestamps_ns[0]:,} → {timestamps_ns[-1]:,}")
        print(f"  摄像头: {len(session.video_streams)} 个")

        # 显示所有视频流信息
        for stream_id, vs in session.video_streams.items():
            try:
                calib = dr.read_calibration(
                    dataset_path, dr.CALIB_TOPICS.get(stream_id, "")
                )
                w, h = calib["width"], calib["height"]
            except (ValueError, KeyError):
                w, h = vs.width, vs.height
            print(f"    {stream_id}: {vs.frame_count} 帧, {w}×{h}, {vs.video_path}")

        # 加载 IMU
        print("  读取 MCAP IMU ...")
        imu_df = session.primary_imu.dataframe
        print(f"  IMU 样本: {len(imu_df)}")

        # 提取所有相机标定
        print("  提取多相机标定 ...")
        calibrations = {}
        for cam_name in ["camera0", "camera1", "camera2", "depth"]:
            try:
                calib_data = dr.read_calibration(
                    dataset_path, dr.CALIB_TOPICS[cam_name]
                )
                calibrations[cam_name] = calib_data
                print(f"    {cam_name}: {calib_data['width']}×{calib_data['height']}")
            except (ValueError, KeyError):
                pass
        calibration = extract_calibration_from_mcap(
            calibrations.get("camera0", {}), multi_cam=calibrations
        )
        print(f"  标定 ID: {calibration['calibration_id']}")

        # 深度已作为正式 DepthStream 进入统一 Session
        for stream_id, depth_stream in session.depth_streams.items():
            print(
                f"  深度流 {stream_id}: {depth_stream.frame_count} 帧, "
                f"{depth_stream.width}×{depth_stream.height}, "
                f"dtype={depth_stream.dtype}, unit={depth_stream.unit}, "
                f"rate={depth_stream.fps} Hz"
            )

        # 构建 source_assets
        mcap_path_obj = Path(dataset_path)
        source_assets = [
            {
                "source_asset_id": "raw_mcap",
                "uri": mcap_path_obj.name,
                "sha256": sha256_hex(dataset_path),
            },
        ]
    elif profile == "umi":
        from zpds_prepare.readers import umi_reader as ur

        # 统一读取 Session
        print("  读取 UMI MCAP Session ...")
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir
            else output_root.parent / ".cache"
        )
        session = ur.read_session(dataset_path, cache_dir=cache_dir)
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps_ns = pv.timestamps_ns
        _source_errors = _verify_source_frames(
            primary_video=pv, candidates_doc=candidates_doc, dataset_path=dataset_path
        )
        if _source_errors:
            for _err in _source_errors:
                print(f"错误: {_err}")
            print("中止处理，请核对 --dataset/-d。")
            return 1
        print(f"  {pv.stream_id}: {pv.frame_count} 帧, "
              f"时间范围: {timestamps_ns[0]:,} → {timestamps_ns[-1]:,}")
        print(f"  摄像头: {len(session.video_streams)} 个, "
              f"IMU: {len(session.imu_streams)} 个")

        # 显示所有流信息
        for stream_id, vs in session.video_streams.items():
            try:
                calib = ur.read_calibration(
                    dataset_path, ur.CALIB_TOPICS.get(
                        stream_id.replace("_camera0", ""), ""
                    )
                )
                w, h = calib["width"], calib["height"]
                dmodel = calib.get("distortion_model", "?")
            except (ValueError, KeyError):
                w, h = vs.width, vs.height
                dmodel = "?"
            print(f"    [{stream_id}] {vs.frame_count} 帧, {w}×{h}, "
                  f"{vs.fps} fps, {dmodel}")

        for stream_id, imu_s in session.imu_streams.items():
            print(f"    [{stream_id}] {len(imu_s.dataframe)} 样本, "
                  f"{imu_s.sample_rate_hz} Hz")
        for stream_id, ts_s in session.time_series_streams.items():
            print(
                f"    [{stream_id}] {ts_s.num_samples} 样本, "
                f"{ts_s.expected_rate_hz} Hz, "
                f"unit={ts_s.metadata.get('unit')}, "
                f"semantic={ts_s.metadata.get('semantic_status')}"
            )

        # 提取双端相机标定
        print("  提取双端相机标定 ...")
        calibrations = {}
        for robot_id in ["robot0", "robot1"]:
            try:
                calib_data = ur.read_calibration(
                    dataset_path, ur.CALIB_TOPICS[robot_id]
                )
                calibrations[robot_id] = calib_data
                print(f"    {robot_id}: {calib_data['width']}×{calib_data['height']}, "
                      f"{calib_data['distortion_model']}, "
                      f"T_b_c={len(calib_data.get('T_b_c', []))} 元")
            except (ValueError, KeyError):
                pass
        calibration = extract_calibration_from_mcap(
            calibrations.get("robot0", {}),
            calibration_id="calib_umi_001",
            multi_cam=calibrations,
        )
        print(f"  标定 ID: {calibration['calibration_id']}")

        # 构建 source_assets
        mcap_path_obj = Path(dataset_path)
        source_assets = [
            {
                "source_asset_id": "raw_mcap",
                "uri": mcap_path_obj.name,
                "sha256": sha256_hex(dataset_path),
            },
        ]
    elif profile == "epic":
        from zpds_prepare.readers import epic_reader as er

        # 从 record JSON 或视频路径加载
        epic_config = {}
        if dataset_path.endswith(".json"):
            with open(dataset_path, "r", encoding="utf-8") as _f:
                _record = json.load(_f)
            video_path = _record.get("video_uri")
            if not video_path:
                print(f"错误: record JSON 缺少 video_uri: {dataset_path}")
                return 1
            if _record.get("hand_object_uri"):
                epic_config["hand_object_path"] = _record["hand_object_uri"]
            if _record.get("mask_uri"):
                epic_config["mask_path"] = _record["mask_uri"]
            print(f"  从 record JSON 加载: {dataset_path}")
        else:
            video_path = dataset_path

        print(f"  读取 EPIC Session: {video_path}")
        session = er.read_session(video_path, config=epic_config if epic_config else None)
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps_ns = pv.timestamps_ns
        _source_errors = _verify_source_frames(
            primary_video=pv, candidates_doc=candidates_doc, dataset_path=dataset_path
        )
        if _source_errors:
            for _err in _source_errors:
                print(f"错误: {_err}")
            print("中止处理，请核对 --dataset/-d。")
            return 1
        print(f"  ego_rgb: {pv.frame_count} 帧, {pv.width}×{pv.height}, {pv.fps} fps")
        print(f"  时间范围: {timestamps_ns[0]:,} → {timestamps_ns[-1]:,}")
        print(f"  标注流: {len(session.annotation_streams)} 个")

        for stream_id, ann_s in session.annotation_streams.items():
            print(f"    [{stream_id}] {ann_s.annotation_type}, "
                  f"{len(ann_s.records)} 标注帧, bbox={ann_s.bbox_format}")

        video_id = session.meta.get("video_id", "")
        try:
            if args.epic_fields_root is None:
                raise FileNotFoundError("未提供 --epic-fields-root")
            calibration = load_epic_fields_calibration(args.epic_fields_root, video_id)
        except FileNotFoundError:
            calibration = missing_epic_fields_calibration(video_id, args.epic_fields_root)
        coverage = calibration["coverage"]["status"]
        print(f"  EPIC-Fields 标定覆盖: {coverage} ({video_id})")

        # source_assets: 视频 + pickle
        source_assets = []
        video_path_obj = Path(video_path)
        source_assets.append({
            "source_asset_id": "raw_color_0",
            "uri": video_path_obj.name,
            "sha256": sha256_hex(video_path) if video_path_obj.exists() else "",
        })
        for stream_id, ann_s in session.annotation_streams.items():
            asset_id = f"raw_{stream_id}_pkl"
            pkl_path = ann_s.source_path
            source_assets.append({
                "source_asset_id": asset_id,
                "uri": str(pkl_path),
                "media_type": "application/python-pickle",
                "sha256": sha256_hex(str(pkl_path)) if pkl_path.exists() else "",
                "ground_truth_status": "model_generated",
            })

    else:
        # Guida 默认模式
        from zpds_prepare.readers import guida_reader as gr

        print("  读取 Session ...")
        depth_cfg = cfg.get("guida", {}).get("depth", {})
        depth_enabled = bool(depth_cfg.get("enabled", True))
        depth_required = bool(depth_cfg.get("required", False))
        session = gr.read_session(
            dataset_path,
            include_depth=depth_enabled,
            require_depth=depth_required,
        )
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps = pv.timestamps_ns
        _source_errors = _verify_source_frames(
            primary_video=pv, candidates_doc=candidates_doc, dataset_path=dataset_path
        )
        if _source_errors:
            for _err in _source_errors:
                print(f"错误: {_err}")
            print("中止处理，请核对 --dataset/-d。")
            return 1
        print(f"  总帧数: {len(index_frames)}, "
              f"时间范围: {timestamps[0]:,} → {timestamps[-1]:,}")

        print("  提取标定信息 ...")
        meta_path = str(Path(dataset_path) / "meta.json")
        calibration = extract_calibration(meta_path)
        print(f"  标定 ID: {calibration['calibration_id']}")

        if session.depth_streams:
            depth_stream = session.depth_streams["ego_depth"]
            print(
                f"  深度流: {depth_stream.source_kind}, "
                f"{depth_stream.frame_count} 帧, "
                f"dtype={depth_stream.dtype}, unit={depth_stream.unit}"
            )
        else:
            print("  深度流: 未发现（本次保持兼容，不写出 Depth Stream）")

        source_assets = _build_guida_source_assets(dataset_path, session)

    # 主视频首帧绝对时间戳：candidates / Hands 时间轴都是原始 MKV 时间轴，
    # 换算为完整视频帧号时以首帧为基准（scene 段沿用其内部逻辑）。
    pv_first_ts = 0
    if pv is not None and getattr(pv, "index_frames", None):
        pv_first_ts = int(pv.index_frames[0]["timestamp_ns"])

    if reviewed_report is not None:
        expected_session_id = reviewed_report["dataset"]["source_session_id"]
        if session.session_id != expected_session_id:
            print(
                "错误: 重新读取源数据后的 Session ID 与审核报告不一致: "
                f"actual={session.session_id!r}, expected={expected_session_id!r}"
            )
            return 1

    # ---- 逐个生成 Prepared Segment ----
    step_header(f"生成 {len(candidates)} 个 Prepared Segment")

    results = []
    total_start = time.time()

    for idx, cand in enumerate(candidates):
        seg_id = f"seg_{idx + 1:06d}"
        seg_dir = revision_root / seg_id

        source_start = cand["source_start_ns"]
        source_end = cand["source_end_ns"]
        duration_s = cand["duration_s"]
        reason = cand.get("reason", "?")
        span_issues = cand.get("issues_in_span", [])

        print(f"\n  [{idx + 1}/{len(candidates)}] {seg_id}")
        print(f"    区间: {source_start:,} → {source_end:,} "
              f"({duration_s:.2f}s)")
        print(f"    原因: {reason}")
        if span_issues:
            print(f"    包含 {len(span_issues)} 个质量问题 (已标记)")

        t0 = time.time()

        # 场景边界帧号 → 相对本 segment 的帧号（只保留区间内的）
        reset_frames: set[int] = set()
        if scene_boundary_frames:
            target_fps = float(cfg["output"]["target_fps"])
            # candidate 起点对应的完整视频帧：candidate 时间戳是原始
            # MKV 时间轴，session 首帧时间戳即完整视频第 0 帧
            first_ts = 0
            if hasattr(session, "primary_video"):
                pv_local = session.primary_video
                if pv_local is not None and getattr(pv_local, "index_frames", None):
                    first_ts = int(pv_local.index_frames[0]["timestamp_ns"])
            cand_start_frame = round(
                (source_start - first_ts) / 1_000_000_000 * target_fps
            ) if first_ts else 0
            for boundary_frame in scene_boundary_frames:
                rel = int(boundary_frame) - cand_start_frame
                if 0 <= rel <= duration_s * target_fps + 1:
                    reset_frames.add(rel)

        # Hands 裁切帧号：hands 帧轴是源解码帧（0-based），用源 fps 换算
        #（scene 用 target_fps 是因为 scene 产物按 run_summary fps 定义，
        #  两者轴各自独立；guida 源 30fps 时与 target_fps 相等）
        hands_crop_spec = None
        if hands_source_dir is not None:
            hands_fps = float(getattr(pv, "fps", 0) or cfg["output"]["target_fps"])
            hands_crop_spec = {
                "source_dir": hands_source_dir,
                "start_ns": source_start,
                "end_ns": source_end,
                "start_frame": (
                    round((source_start - pv_first_ts) / 1_000_000_000 * hands_fps)
                    if pv_first_ts else 0
                ),
                "end_frame": (
                    round((source_end - pv_first_ts) / 1_000_000_000 * hands_fps)
                    if pv_first_ts else 0
                ),
                "source_fps": hands_fps,
            }

        try:
            result = generate_segment(
                dataset_path=dataset_path,
                source_start_ns=source_start,
                source_end_ns=source_end,
                segment_id=seg_id,
                output_dir=str(seg_dir),
                session=session,
                calibration=calibration,
                cfg=cfg,
                session_id=source_session_id,
                prep_revision=REVISION,
                quality_issues=span_issues if span_issues else None,
                profile=profile,
                source_assets=source_assets,
                experience_dir=args.experience_dir,
                experience_version=args.experience_version,
                privacy_enabled=args.with_privacy,
                privacy_reset_frames=reset_frames if reset_frames else None,
                hands_source=hands_crop_spec,
            )
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)

            status_icon = "✓" if result["status"] == "pass" else "✗"
            print(f"    {status_icon} 状态: {result['status'].upper()}")
            print(f"    RGB 帧: {result['rgb_frames']}, "
                  f"IMU 样本: {result['imu_samples']}, "
                  f"时序样本: {result['time_series_samples']}, "
                  f"耗时: {elapsed:.1f}s")

            if result["errors"]:
                for e in result["errors"]:
                    print(f"    ⚠ {e}")

        except Exception as exc:  # noqa: BLE001 - 单个 Segment 失败不能中止整批
            elapsed = time.time() - t0
            result = {
                "segment_id": seg_id,
                "status": "fail",
                "duration_s": duration_s,
                "error": str(exc),
                "elapsed_s": round(elapsed, 1),
            }
            print(f"    ✗ FAIL: {exc}")

        results.append(result)

    # ---- 汇总 ----
    total_elapsed = time.time() - total_start
    step_header("批量生成完成")

    pass_count = sum(1 for r in results if r["status"] == "pass")
    fail_count = sum(1 for r in results if r["status"] == "fail")
    total_rgb = sum(r.get("rgb_frames", 0) for r in results)
    total_imu = sum(r.get("imu_samples", 0) for r in results)
    total_time_series = sum(
        r.get("time_series_samples", 0)
        for r in results
    )

    print(f"  总数:        {len(results)}")
    print(f"  ✓ 通过:      {pass_count}")
    if fail_count > 0:
        print(f"  ✗ 失败:      {fail_count}")
    print(f"  RGB 总帧:    {total_rgb}")
    print(f"  IMU 总样本:  {total_imu}")
    print(f"  时序总样本:  {total_time_series}")
    print(f"  总耗时:      {total_elapsed:.1f}s")
    print(f"  输出目录:    {output_root.resolve()}")

    # ---- 写出批量汇总 ----
    summary_path = output_root / "batch_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema_version": "0.1.0",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_session_id": source_session_id,
        "candidates_path": str(candidates_path.resolve()),
        "profile": profile,
        "total_segments": len(results),
        "pass": pass_count,
        "fail": fail_count,
        "total_rgb_frames": total_rgb,
        "total_imu_samples": total_imu,
        "total_time_series_samples": total_time_series,
        "total_elapsed_s": round(total_elapsed, 1),
        "segments": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  汇总文件:    {summary_path.resolve()}")

    # ---- 写出 dataset.json 与 revision.json（ZPDS dataset 结构） ----
    dataset_root = output_root.parent
    dataset_id = dataset_root.name
    config_path = Path(args.config).expanduser().resolve()
    config_hash = "sha256:" + sha256_hex(str(config_path))
    source_type = {"guida": "ego", "dunjia": "ego", "epic": "ego"}.get(
        profile, "teleoperation"
    )
    dataset_doc = build_dataset_json(
        dataset_id=dataset_id,
        prep_revision=prep_revision,
        name=dataset_id,
        description=f"{profile} 清洗后的 Prepared Segment 数据集",
        source_types=[source_type],
    )

    # changes 从实际执行的配置与清洗步骤生成，不写死。
    output_cfg = cfg.get("output", {})
    target_fps = output_cfg.get("target_fps", 30)
    codec = output_cfg.get("rgb_codec", "avc1")
    changes: list[str] = ["按候选区间裁剪无效首尾"]
    changes.append(f"RGB 转 {codec} CFR {target_fps}fps")
    depth_enabled = bool(
        cfg.get(profile, {}).get("depth", {}).get("enabled", False)
    )
    if depth_enabled:
        changes.append("深度保留原频率无损写出")
    if profile == "umi":
        changes.append("磁编码器/VIO 保留原始值与双时钟")
    if cfg.get("preview", {}).get("enabled", True):
        changes.append("生成 <stream_id>_preview.mp4 压缩预览（保留原视频）")
    changes.append("Prepared 层长度单位统一为 m")
    revision_doc = build_revision_json(
        prep_revision=prep_revision,
        pipeline_name=f"zpds.{Path(__file__).stem}",
        pipeline_version=package_version(),
        config_hash=config_hash,
        changes=changes,
        run_stats={
            "profile": profile,
            "source_session_id": source_session_id,
            "segment_count": len(results),
            "video_stream_ids": sorted(
                {
                    stream_id
                    for result in results
                    for stream_id in result.get("video_streams", [])
                }
            ),
            "rgb_frames_total": total_rgb,
            "imu_samples_total": total_imu,
            "depth_frames_total": sum(
                result.get("depth_frames", 0) for result in results
            ),
            "preview_count_total": sum(
                result.get("preview_count", 0) for result in results
            ),
        },
    )
    dataset_path_out = write_dataset_json(dataset_doc, dataset_root)
    revision_path_out = write_revision_json(revision_doc, revision_root)
    print(f"  dataset.json:  {dataset_path_out}")
    print(f"  revision.json: {revision_path_out}")
    print(f"  修订目录:      {revision_root.resolve()}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
