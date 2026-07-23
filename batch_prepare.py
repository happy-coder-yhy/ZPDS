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
    python batch_prepare.py
    python batch_prepare.py --candidates output/segment_candidates.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

from segment.video_transcoder import transcode_rgb
from segment.sample_map import generate_sample_map, write_sample_map
from segment.imu_normalizer import normalize_imu, write_imu
from segment.calibration import extract_calibration, write_calibration
from segment.segment_writer import build_segment_json, write_segment_json
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
) -> dict:
    """为单个候选区间生成完整 Prepared Segment。

    Args:
        dataset_path: 原始数据集根目录
        source_start_ns: 源设备时间戳起始
        source_end_ns: 源设备时间戳结束
        segment_id: Segment ID (如 seg_000001)
        output_dir: 输出目录
        index_frames: index.jsonl 全部帧列表（共享，避免重复读取）
        calibration: 标定 dict（共享）
        cfg: 完整配置 dict
        session_id: 来源 Session ID
        revision: 修订版本号
        quality_issues: 落在此 Segment 内的 QualityIssue

    Returns:
        {"segment_id": str, "status": str, "duration_s": float, ...}
    """
    target_fps = cfg["output"]["target_fps"]
    source_mkv = str(Path(dataset_path) / "color_000000.mkv")
    imu_path = str(Path(dataset_path) / "imu" / "imu_000000.csv")

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
    output_mp4 = str(Path(output_dir) / "data" / "ego_rgb.mp4")
    video_result = transcode_rgb(
        source_mkv=source_mkv,
        output_mp4=output_mp4,
        source_start_ns=source_start_ns,
        source_end_ns=source_end_ns,
        index_frames=index_frames,
        target_fps=target_fps,
    )

    # ---- ② 生成采样映射表 ----
    sample_map = generate_sample_map(
        index_frames=index_frames,
        source_start_ns=source_start_ns,
        source_end_ns=source_end_ns,
        target_fps=target_fps,
    )
    write_sample_map(sample_map, output_dir)

    # ---- ③ 规范化 IMU ----
    imu = normalize_imu(
        imu_path=imu_path,
        source_start_ns=source_start_ns,
        source_end_ns=source_end_ns,
    )
    write_imu(imu, output_dir)

    # ---- ④ 写出 calibration（共享，只写一次由调用方处理） ----
    write_calibration(calibration, output_dir)

    # ---- ⑤ 生成 segment.json ----
    segment = build_segment_json(
        dataset_path=dataset_path,
        span=span,
        video_result=video_result,
        sample_map_rows=len(sample_map),
        imu_rows=len(imu),
        calibration_id=calibration["calibration_id"],
        revision=revision,
        segment_id=segment_id,
        session_id=session_id,
        quality_issues=quality_issues,
    )
    write_segment_json(segment, output_dir)

    # ---- ⑥ 写出后验证 ----
    validation = validate_segment(output_dir)
    write_validation_report(validation, output_dir)

    return {
        "segment_id": segment_id,
        "status": validation["status"],
        "duration_s": duration_ns / 1_000_000_000,
        "rgb_frames": video_result["output_frames"],
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
        default=CANDIDATES_PATH,
        help="segment_candidates.json 路径",
    )
    parser.add_argument(
        "--dataset", "-d",
        default=DATASET,
        help="数据集根目录",
    )
    parser.add_argument(
        "--output", "-o",
        default=OUTPUT_ROOT,
        help="输出根目录 (默认 prepared_segments/)",
    )
    parser.add_argument(
        "--config",
        default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    args = parser.parse_args()

    start_time = time.time()

    # ---- 加载配置和候选方案 ----
    cfg = load_config(args.config)
    candidates_path = Path(args.candidates)
    dataset_path = args.dataset
    output_root = Path(args.output)

    if not candidates_path.exists():
        print(f"错误: 候选文件不存在: {candidates_path}")
        print(f"请先运行: python -m zpds_prepare.main \"{dataset_path}\"")
        return 1

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)

    candidates = candidates_doc.get("segments", [])
    source_session_id = candidates_doc.get("source_session_id", "guida_session_001")

    if not candidates:
        print("没有候选 Segment，退出。")
        return 0

    print(f"数据源:       {dataset_path}")
    print(f"候选方案:     {candidates_path}")
    print(f"候选数量:     {len(candidates)}")
    print(f"Session ID:   {source_session_id}")

    # ---- 预加载共享资源 ----
    step_header("预加载共享资源")

    print("  读取 index.jsonl ...")
    index_frames = load_index(dataset_path)
    timestamps = [f["timestamp_ns"] for f in index_frames]
    print(f"  总帧数: {len(index_frames)}, "
          f"时间范围: {timestamps[0]:,} → {timestamps[-1]:,}")

    print("  提取标定信息 ...")
    meta_path = str(Path(dataset_path) / "meta.json")
    calibration = extract_calibration(meta_path)
    print(f"  标定 ID: {calibration['calibration_id']}")

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
                index_frames=index_frames,
                calibration=calibration,
                cfg=cfg,
                session_id=source_session_id,
                revision=REVISION,
                quality_issues=span_issues if span_issues else None,
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
