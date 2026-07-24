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
from segment.imu_normalizer import normalize_imu_df, write_imu
from segment.calibration import (
    extract_calibration,
    extract_calibration_from_mcap,
    write_calibration,
)
from segment.segment_writer import build_segment_json, write_segment_json
from segment.segment_writer import sha256_hex
from segment.validator import validate_segment, write_validation_report

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

    # ---- ① 每个视频流: 转码 + sample_map ----
    video_results = []
    for stream_id, vs in session.video_streams.items():
        output_mp4 = str(Path(output_dir) / "data" / f"{stream_id}.mp4")
        vr = transcode_rgb(
            source_video=vs.video_path,
            output_mp4=output_mp4,
            source_start_ns=source_start_ns,
            source_end_ns=source_end_ns,
            index_frames=vs.index_frames,
            target_fps=target_fps,
        )
        vr["stream_id"] = stream_id

        # ② sample_map
        if profile in ("dunjia", "umi"):
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
        vr["sample_map_uri"] = f"maps/{stream_id}_sample_map.parquet"

        video_results.append(vr)

    # ---- ③ 每个 IMU 流: 规范化 + 写出 ----
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

    # ---- ④ 写出 calibration ----
    write_calibration(calibration, output_dir)

    # ---- ⑤ 生成 segment.json ----
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
        "rgb_frames": video_results[0]["output_frames"] if video_results else 0,
        "imu_samples": sum(ir["rows"] for ir in imu_results),
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
        choices=["guida", "dunjia", "umi"],
        help="数据源 profile (默认: guida)",
    )
    args = parser.parse_args()

    profile = args.profile
    start_time = time.time()

    # ---- 加载配置和候选方案 ----
    cfg = load_config(args.config)

    # 默认 candidates 路径按 profile 分子目录
    if args.candidates is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi"}
        subdir = profile_subdirs.get(profile, profile)
        candidates_path = Path("output") / subdir / "segment_candidates.json"
    else:
        candidates_path = Path(args.candidates)

    dataset_path = args.dataset

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi"}
        subdir = profile_subdirs.get(profile, profile)
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
        session = dr.read_session(dataset_path)
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps_ns = pv.timestamps_ns
        print(f"  camera0: {pv.frame_count} 帧, "
              f"时间范围: {timestamps_ns[0]:,} → {timestamps_ns[-1]:,}")
        print(f"  摄像头: {len(session.video_streams)} 个")

        # 显示所有视频流信息
        video_results = []
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
    elif profile == "umi":
        from zpds_prepare.readers import umi_reader as ur

        # 统一读取 Session
        print("  读取 UMI MCAP Session ...")
        session = ur.read_session(dataset_path)
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

        # UMI 第一版无深度
        depth_npz_path = None

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
        from zpds_prepare.readers import guida_reader as gr

        print("  读取 Session ...")
        session = gr.read_session(dataset_path)
        pv = session.primary_video
        index_frames = pv.index_frames
        timestamps = pv.timestamps_ns
        print(f"  总帧数: {len(index_frames)}, "
              f"时间范围: {timestamps[0]:,} → {timestamps[-1]:,}")

        print("  提取标定信息 ...")
        meta_path = str(Path(dataset_path) / "meta.json")
        calibration = extract_calibration(meta_path)
        print(f"  标定 ID: {calibration['calibration_id']}")

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
            # 深度 — Dunjia 专有（不在 session 中，需单独处理）
            seg_depth_path = None
            if profile == "dunjia" and dr.TOPIC_DEPTH in dr.CAMERA_TOPICS.values():
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
                session=session,
                calibration=calibration,
                cfg=cfg,
                session_id=source_session_id,
                revision=REVISION,
                quality_issues=span_issues if span_issues else None,
                profile=profile,
                source_assets=source_assets,
                depth_npz_path=seg_depth_path,
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
