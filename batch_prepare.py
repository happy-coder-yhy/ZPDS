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
from segment.sample_map import (
    generate_sample_map,
    generate_sample_map_from_timestamps,
    write_sample_map,
)
from segment.segment_writer import build_segment_json, sha256_hex, write_segment_json
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

    return {
        "segment_id": segment_id,
        "status": validation["status"],
        "duration_s": duration_ns / 1_000_000_000,
        "rgb_frames": video_results[0]["output_frames"] if video_results else 0,
        "imu_samples": sum(ir["rows"] for ir in imu_results),
        "depth_frames": sum(dr["frames"] for dr in depth_results),
        "time_series_samples": sum(
            result["rows"]
            for result in time_series_results
        ),
        "annotation_rows": sum(
            ar.get("rows", 0) for ar in annotation_results
        ) if annotation_results else 0,
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
    args = parser.parse_args()

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
        seg_dir = output_root / seg_id

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

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
