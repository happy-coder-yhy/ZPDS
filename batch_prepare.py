"""
ZPDS 批量 Prepared Segment 生成。

读取 segment_candidates.json，对每个候选区间：
  ① 裁剪并转码 RGB 视频
  ② 生成 rgb_sample_map.parquet
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
import json
import sys
import time
from pathlib import Path

import yaml

from segment.video_transcoder import transcode_rgb
from segment.sample_map import (
    generate_sample_map,
    generate_sample_map_from_timestamps,
    write_sample_map,
)
from segment.imu_normalizer import normalize_imu, normalize_imu_df, write_imu
from segment.calibration import (
    extract_calibration,
    extract_calibration_from_mcap,
    write_calibration,
)
from segment.segment_writer import build_segment_json, write_segment_json
from segment.segment_writer import sha256_hex
from segment.validator import validate_segment, write_validation_report
from segment.span_determiner import load_index

# ============================================================
# 配置
# ============================================================
DATASET = "E:/datasets/egos/墨现"
CANDIDATES_PATH = "output/segment_candidates.json"
CONFIG_PATH = "config.yaml"
OUTPUT_ROOT = "prepared_segments"
REVISION = "r0001"


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def generate_segment(
    dataset_path: str,
    source_start_ns: int,
    source_end_ns: int,
    segment_id: str,
    output_dir: str,
    index_frames: list[dict],
    calibration: dict,
    cfg: dict,
    session_id: str = "guida_session_001",
    revision: str = "r0001",
    quality_issues: list[dict] | None = None,
    profile: str = "guida",
    source_video: str | None = None,
    imu_df: "pd.DataFrame | None" = None,
    timestamps_ns: list[int] | None = None,
    source_assets: list[dict] | None = None,
    video_results: list[dict] | None = None,  # Dunjia: 多个相机的转码结果
    depth_npz_path: str | None = None,          # Dunjia: 深度 npz 路径
) -> dict:
    """为单个候选区间生成完整 Prepared Segment。

    Args:
        ...
        video_results: Dunjia 模式: 由调用方预先生成的所有相机转码结果列表
        depth_npz_path: Dunjia 模式: 深度 .npz 文件路径
    """
    import pandas as pd

    target_fps = cfg["output"]["target_fps"]

    if source_video is None:
        source_video = str(Path(dataset_path) / "color_000000.mkv")

    duration_ns = source_end_ns - source_start_ns

    # 构建兼容 span 的 dict（segment_writer 需要）
    span = {
        "source_start_ns": source_start_ns,
        "source_end_ns": source_end_ns,
        "duration_s": duration_ns / 1_000_000_000,
        "total_frames_in_span": sum(
            1 for f in index_frames
            if source_start_ns <= f["timestamp_ns"] <= source_end_ns
        ),
        "reason": {"start": "from_candidate", "end": "from_candidate"},
        "trimmed_head_frames": 0,
        "trimmed_tail_frames": 0,
    }

    # ---- ① 裁剪转码 RGB 视频 ----
    if video_results is not None:
        # Dunjia: 多相机由调用方预先转码完成
        pass  # video_results 已包含所有相机的转码结果
    else:
        # Guida: 单个相机，在此处转码
        output_mp4 = str(Path(output_dir) / "data" / "ego_rgb.mp4")
        video_result = transcode_rgb(
            source_video=source_video,
            output_mp4=output_mp4,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
            index_frames=index_frames,
            target_fps=target_fps,
        )
        video_results = [video_result]

    # ---- ② 生成采样映射表 ----
    if profile == "dunjia" and timestamps_ns is not None:
        sample_map = generate_sample_map_from_timestamps(
            timestamps_ns=timestamps_ns,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
            target_fps=target_fps,
        )
        sample_map_rows = len(sample_map)
    else:
        sample_map = generate_sample_map(
            index_frames=index_frames,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
            target_fps=target_fps,
        )
        sample_map_rows = len(sample_map)
    write_sample_map(sample_map, output_dir)

    # ---- ③ 规范化 IMU ----
    if profile == "dunjia" and imu_df is not None:
        imu = normalize_imu_df(
            imu=imu_df,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
        )
    else:
        imu_path = str(Path(dataset_path) / "imu" / "imu_000000.csv")
        imu = normalize_imu(
            imu_path=imu_path,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
        )
    write_imu(imu, output_dir)

    # ---- ④ 写出 calibration ----
    write_calibration(calibration, output_dir)

    # ---- ⑤ 生成 segment.json ----
    segment = build_segment_json(
        dataset_path=dataset_path,
        span=span,
        video_results=video_results,
        sample_map_rows=sample_map_rows,
        imu_rows=len(imu),
        calibration_id=calibration["calibration_id"],
        revision=revision,
        segment_id=segment_id,
        session_id=session_id,
        quality_issues=quality_issues,
        source_assets=source_assets,
        profile=profile,
        depth_npz_path=depth_npz_path,
        calibrations=calibration.get("calibrations", None),
    )
    write_segment_json(segment, output_dir)

    # ---- ⑥ 写出后验证 ----
    validation = validate_segment(output_dir)
    write_validation_report(validation, output_dir)

    return {
        "segment_id": segment_id,
        "status": validation["status"],
        "duration_s": duration_ns / 1_000_000_000,
        "rgb_frames": video_results[0]["output_frames"],
        "imu_samples": len(imu),
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
        "--profile", "-p",
        default="guida",
        choices=["guida", "dunjia"],
        help="数据源 profile (默认: guida)",
    )
    args = parser.parse_args()

    profile = args.profile
    start_time = time.time()

    # ---- 加载配置和候选方案 ----
    cfg = load_config(args.config)

    # 默认 candidates 路径按 profile 分子目录
    if args.candidates is None:
        subdir = "moxian" if profile == "guida" else "dunjia"
        candidates_path = Path("output") / subdir / "segment_candidates.json"
    else:
        candidates_path = Path(args.candidates)

    dataset_path = args.dataset

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        subdir = "moxian" if profile == "guida" else "dunjia"
        output_root = Path("prepared_segments") / subdir
    else:
        output_root = Path(args.output)

    if not candidates_path.exists():
        print(f"错误: 候选文件不存在: {candidates_path}")
        print(f"请先运行: python -m zpds_prepare.main \"{dataset_path}\" --profile {profile}")
        return 1

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)

    candidates = candidates_doc.get("segments", [])
    source_session_id = candidates_doc.get(
        "source_session_id",
        "guida_session_001" if profile == "guida" else "dunjia_session_001",
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

        # 加载主相机 index_frames (camera0, 用于 QC 时间线)
        print("  读取 camera0 帧索引 ...")
        index_frames = dr.read_index_frames(dataset_path)
        timestamps_ns = dr.read_index_timestamps(dataset_path)
        print(f"  camera0: {len(index_frames)} 帧, "
              f"时间范围: {timestamps_ns[0]:,} → {timestamps_ns[-1]:,}")

        # 重构所有 RGB 相机视频
        print("  重构多相机 H264 视频 ...")
        video_results = []
        for cam_name in ["camera0", "camera1", "camera2"]:
            topic = dr.CAMERA_TOPICS[cam_name]
            mp4_path = dr.get_video_for_topic(dataset_path, topic)
            cam_frames = dr.read_index_frames(dataset_path, topic)
            try:
                calib = dr.read_calibration(dataset_path, dr.CALIB_TOPICS[cam_name])
                w, h = calib["width"], calib["height"]
            except (ValueError, KeyError):
                w, h = 0, 0
            print(f"    {cam_name}: {len(cam_frames)} 帧, {w}×{h}, {mp4_path}")

        # 重构主相机 source_video (保持向后兼容)
        source_video = dr.get_video_for_topic(dataset_path, dr.TOPIC_CAMERA0)

        # 加载 IMU
        print("  读取 MCAP IMU ...")
        imu_df = dr.read_imu(dataset_path)
        print(f"  IMU 样本: {len(imu_df)}")

        # 提取所有相机标定
        print("  提取多相机标定 ...")
        calibrations = {}
        for cam_name in ["camera0", "camera1", "camera2", "depth"]:
            try:
                calib_data = dr.read_calibration(dataset_path, dr.CALIB_TOPICS[cam_name])
                calibrations[cam_name] = calib_data
                print(f"    {cam_name}: {calib_data['width']}×{calib_data['height']}")
            except (ValueError, KeyError):
                pass
        calibration = extract_calibration_from_mcap(
            calibrations.get("camera0", {}), multi_cam=calibrations
        )
        print(f"  标定 ID: {calibration['calibration_id']}")

        # 处理深度
        depth_npz_path = None
        if dr.TOPIC_DEPTH in dr.CAMERA_TOPICS.values():
            print("  处理深度流 ...")
            depth_frames = dr.read_depth_frames(dataset_path, dr.TOPIC_DEPTH)
            if depth_frames:
                print(f"    depth: {len(depth_frames)} 帧, "
                      f"{depth_frames[0]['width']}×{depth_frames[0]['height']}, "
                      f"{depth_frames[0]['dtype']}")

        # 构建 source_assets
        mcap_path_obj = Path(dataset_path)
        source_assets = [
            {
                "source_asset_id": "raw_mcap",
                "uri": mcap_path_obj.name,
                "sha256": sha256_hex(dataset_path),
            },
        ]
    else:
        # Guida 默认模式
        print("  读取 index.jsonl ...")
        index_frames = load_index(dataset_path)
        timestamps = [f["timestamp_ns"] for f in index_frames]
        print(f"  总帧数: {len(index_frames)}, "
              f"时间范围: {timestamps[0]:,} → {timestamps[-1]:,}")

        print("  提取标定信息 ...")
        meta_path = str(Path(dataset_path) / "meta.json")
        calibration = extract_calibration(meta_path)
        print(f"  标定 ID: {calibration['calibration_id']}")

        source_video = None
        imu_df = None
        timestamps_ns = None
        source_assets = None

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
            # Dunjia: 每个相机单独转码
            if profile == "dunjia":
                seg_video_results = []
                for cam_name in ["camera0", "camera1", "camera2"]:
                    topic = dr.CAMERA_TOPICS[cam_name]
                    cam_video = dr.get_video_for_topic(dataset_path, topic)
                    cam_frames = dr.read_index_frames(dataset_path, topic)
                    stream_id = {
                        "camera0": "ego_rgb_center",
                        "camera1": "ego_rgb_left",
                        "camera2": "ego_rgb_right",
                    }[cam_name]
                    out_mp4 = str(Path(seg_dir) / "data" / f"{stream_id}.mp4")
                    vr = transcode_rgb(
                        source_video=cam_video,
                        output_mp4=out_mp4,
                        source_start_ns=source_start,
                        source_end_ns=source_end,
                        index_frames=cam_frames,
                        target_fps=cfg["output"]["target_fps"],
                    )
                    vr["stream_id"] = stream_id
                    vr["camera_name"] = cam_name
                    seg_video_results.append(vr)

                # 深度 — H.265 无损 MP4 视频
                seg_depth_path = None
                depth_vr = None
                if dr.TOPIC_DEPTH in dr.CAMERA_TOPICS.values():
                    seg_depth_path = str(Path(seg_dir) / "data" / "ego_depth.mp4")
                    depth_vr = dr.transcode_depth_video(
                        dataset_path, seg_depth_path,
                        source_start, source_end,
                        target_fps=cfg["output"]["target_fps"],
                    )
                    seg_depth_path = depth_vr["output_path"]

                result = generate_segment(
                    dataset_path=dataset_path,
                    source_start_ns=source_start,
                    source_end_ns=source_end,
                    segment_id=seg_id,
                    output_dir=str(seg_dir),
                    index_frames=index_frames,
                    calibration=calibration,
                    cfg=cfg,
                    session_id=source_session_id,
                    revision=REVISION,
                    quality_issues=span_issues if span_issues else None,
                    profile=profile,
                    source_video=source_video,
                    imu_df=imu_df,
                    timestamps_ns=timestamps_ns,
                    source_assets=source_assets,
                    video_results=seg_video_results,
                    depth_npz_path=seg_depth_path,
                )
            else:
                result = generate_segment(
                    dataset_path=dataset_path,
                source_start_ns=source_start,
                source_end_ns=source_end,
                segment_id=seg_id,
                output_dir=str(seg_dir),
                index_frames=index_frames,
                calibration=calibration,
                cfg=cfg,
                session_id=source_session_id,
                revision=REVISION,
                quality_issues=span_issues if span_issues else None,
                profile=profile,
                source_video=source_video,
                imu_df=imu_df,
                timestamps_ns=timestamps_ns,
                source_assets=source_assets,
            )
            elapsed = time.time() - t0
            result["elapsed_s"] = round(elapsed, 1)

            status_icon = "✓" if result["status"] == "pass" else "✗"
            print(f"    {status_icon} 状态: {result['status'].upper()}")
            print(f"    RGB 帧: {result['rgb_frames']}, "
                  f"IMU 样本: {result['imu_samples']}, "
                  f"耗时: {elapsed:.1f}s")

            if result["errors"]:
                for e in result["errors"]:
                    print(f"    ⚠ {e}")

        except Exception as exc:
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

    print(f"  总数:        {len(results)}")
    print(f"  ✓ 通过:      {pass_count}")
    if fail_count > 0:
        print(f"  ✗ 失败:      {fail_count}")
    print(f"  RGB 总帧:    {total_rgb}")
    print(f"  IMU 总样本:  {total_imu}")
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
        "total_elapsed_s": round(total_elapsed, 1),
        "segments": results,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  汇总文件:    {summary_path.resolve()}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
