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
import sys
import time
from pathlib import Path

import pandas as pd
import yaml

from segment.annotation_normalizer import normalize_hand_objects, write_annotation_parquet
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


def _overlay_redactions(
    records,
    video_path: Path,
    *,
    face_method: str = "blur",
    text_method: str = "black_rect",
    blur_ksize: int = 41,
    blur_sigma: int = 15,
) -> None:
    """按逐帧遮挡区域重渲染视频并原地覆盖（等长，无丢帧）。

    ``zpds.privacy.writer.write_redacted_video`` 会跳过无遮挡帧（丢帧）
    且要求首帧存在遮挡区域，不适合 segment 转码产物的等长覆盖。
    这里重读原视频逐帧渲染：无区域的帧写原帧，保证帧数与时序一致。
    """
    import cv2

    from zpds.privacy.redaction import FrameRedactor

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"无法打开视频: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    tmp_path = video_path.with_name(f"{video_path.stem}.redacting.mp4")

    # OpenH264 不可用（CLAUDE.md），直接用 mp4v；avc1 在部分 opencv 版本
    # 构造成功但写帧时才失败，不能用 isOpened() 判断，直接 mp4v 最稳。
    writer = cv2.VideoWriter(
        str(tmp_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    redactor = FrameRedactor(
        face_method=face_method,
        text_method=text_method,
        blur_ksize=blur_ksize,
        blur_sigma=blur_sigma,
    )
    try:
        for record in records:
            ok, frame = capture.read()
            if not ok:
                break
            if record.regions:
                frame = redactor.apply(frame, record.regions)
            writer.write(frame)
    finally:
        capture.release()
        writer.release()
    os.replace(tmp_path, video_path)


def _redact_segment_videos(
    video_meta: list[dict],
    video_results: list[dict],
    output_dir: Path,
    profile: str,
) -> int:
    """对转码后的 segment 视频做隐私脱敏，原地覆盖 ``output_mp4``。

    按 Profile 路由（face/text 适用性）：
    - 两者都不适用（如遁甲/UMI/A2D 无操作者人脸时仍可能拍文本，text 默认 applicable）
    - 人脸模糊 + 文本遮挡后覆盖原转码产物，脱敏版即训练用产物

    Returns:
        完成脱敏的视频流数量。
    """
    from zpds.privacy.backend_router import PrivacyBackendPolicy
    from zpds.privacy.config import PrivacyConfig
    from zpds.privacy.pipeline import PrivacyPipeline
    from zpds.privacy.writer import write_manifest

    policy = PrivacyBackendPolicy.from_profile(profile)
    if not policy.face_enabled and not policy.text_enabled:
        for vr in video_results:
            vr["redaction_skipped"] = True
            vr["redaction_skip_reason"] = (
                f"profile {profile}: face={policy.face_applicability}, "
                f"text={policy.text_applicability}"
            )
        return 0

    # 项目根绝对路径（batch_prepare 可能从任意 cwd 启动）
    config_path = Path(__file__).resolve().parent / "configs/privacy/default.yaml"
    try:
        pcfg = PrivacyConfig.load(config_path)
    except FileNotFoundError:
        print(f"[warn] 隐私配置不存在 ({config_path})，使用默认值")
        pcfg = PrivacyConfig.defaults()

    redacted_count = 0
    for vm, vr in zip(video_meta, video_results):
        stream_id = vr["stream_id"]
        mp4_path = Path(vm["output_mp4"])
        if not mp4_path.is_file():
            vr["redaction_skipped"] = True
            vr["redaction_skip_reason"] = f"转码产物不存在: {mp4_path}"
            continue
        print(
            f"  脱敏 [{stream_id}]: "
            f"人脸={'启用' if policy.face_enabled else '跳过'} "
            f"文本={'启用' if policy.text_enabled else '跳过'}"
        )
        pipeline = PrivacyPipeline(
            mp4_path,
            config=pcfg,
            policy=policy,
            profile=profile,
            session_id=stream_id,
        )
        records = pipeline.run_to_list()
        manifest = pipeline.build_manifest()

        # 原地覆盖：脱敏视频即为训练用产物（等长重渲染，无丢帧）
        _overlay_redactions(
            records,
            mp4_path,
            face_method=pcfg.face_method,
            text_method=pcfg.redaction_text_method,
            blur_ksize=pcfg.face_blur_ksize,
            blur_sigma=pcfg.face_blur_sigma,
        )
        manifest_path = mp4_path.parent / f"{stream_id}_redaction_manifest.parquet"
        write_manifest(records, manifest, manifest_path)

        vr["redacted"] = True
        vr["redaction_manifest_uri"] = f"data/{stream_id}_redaction_manifest.parquet"
        vr["redaction_face"] = policy.face_enabled
        vr["redaction_text"] = policy.text_enabled
        vr["redaction_stats"] = {
            "frames_processed": manifest.total_frames,
            "frames_with_faces": manifest.frames_with_faces,
            "frames_with_text": manifest.frames_with_text,
            "face_regions": manifest.total_face_regions,
            "text_regions": manifest.total_text_regions,
            "pii_categories_found": list(manifest.pii_categories_found),
            "llm_available": manifest.llm_available,
        }
        redacted_count += 1
    return redacted_count


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
    revision: str = "r0001",
    quality_issues: list[dict] | None = None,
    profile: str = "guida",
    source_assets: list[dict] | None = None,
    depth_npz_path: str | None = None,
    experience_dir: str | None = None,
    experience_version: str | None = None,
    privacy_enabled: bool = False,
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
        redacted_count = _redact_segment_videos(
            video_meta,
            video_results,
            Path(output_dir),
            profile,
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

    # ---- ⑦ 标定与 segment.json ----
    write_calibration(calibration, output_dir)
    segment = build_segment_json(
        dataset_path=dataset_path,
        span=span,
        video_results=video_results,
        imu_results=imu_results,
        calibration_id=calibration["calibration_id"],
        revision=revision,
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
        "--dataset", "-d",
        default=DATASET,
        help="数据集路径 (墨现: 目录; 遁甲: .mcap 文件)",
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

    # 项目根绝对路径（batch_prepare 可能从任意 cwd 启动）
    _load_dotenv(Path(__file__).resolve().parent / ".env")
    profile = args.profile

    # ---- 加载配置和候选方案 ----
    cfg = load_config(args.config)

    # 默认 candidates 路径按 profile 分子目录
    if args.candidates is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi", "epic": "epic"}
        subdir = profile_subdirs.get(profile, profile)
        candidates_path = Path("output") / subdir / "segment_candidates.json"
    else:
        candidates_path = Path(args.candidates)

    dataset_path = args.dataset

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi"}
        subdir = profile_subdirs.get(profile, profile)
        output_root = Path("output") / subdir / "prepared_segments"
    else:
        output_root = Path(args.output)

    # ZPDS dataset 结构：prepared_segments/<prep_revision>/<segment_id>/
    prep_revision = REVISION
    revision_root = output_root / prep_revision

    if not candidates_path.exists():
        print(f"错误: 候选文件不存在: {candidates_path}")
        print(f"请先运行: python -m zpds_prepare.main \"{dataset_path}\" --profile {profile}")
        return 1

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
                revision=REVISION,
                quality_issues=span_issues if span_issues else None,
                profile=profile,
                source_assets=source_assets,
                experience_dir=args.experience_dir,
                experience_version=args.experience_version,
                privacy_enabled=args.with_privacy,
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
    changes.append(
        "长度单位按 zpds/prepared/conventions.py（mm）；"
        "ZPDS 数据标准示例为 m，冲突待解决"
    )
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
