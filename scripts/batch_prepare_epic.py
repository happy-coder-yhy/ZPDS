"""
EPIC-KITCHENS-100 批量 Prepared Segment 生成器。

从 inventory.json 出发，对每条 status == "ready" 的记录：
  ① 写出单条 record JSON (output/epic/records/{video_id}.json)
  ② 运行技术质量检测 → segment_candidates.json
  ③ 对每个候选 Segment: 视频裁剪 + 标注标准化 + segment.json + Validator

用法:
    python scripts/batch_prepare_epic.py \\
        --inventory output/epic/inventory.json \\
        --videos-root E:/datasets/egos/EPIC-KITCHENS-100/videos \\
        --annotations-root E:/datasets/egos/EPIC-KITCHENS-100

    python scripts/batch_prepare_epic.py --inventory inventory.json --limit 10
    python scripts/batch_prepare_epic.py --inventory inventory.json --participant P01
    python scripts/batch_prepare_epic.py --inventory inventory.json --resume
    python scripts/batch_prepare_epic.py --inventory inventory.json --failed-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

# 确保项目根在 sys.path 上（无需手动设 PYTHONPATH）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import yaml  # noqa: E402 - project root must be available to direct script execution


# ---- 路径常量 ----

CONFIG_PATH = "config.yaml"
RECORDS_DIR = Path("output/epic/records")
PREPARED_ROOT = Path("output/epic/prepared_segments")
BATCH_SUMMARY_PATH = PREPARED_ROOT / "batch_summary.json"


# ---- 辅助: 写出单条 record JSON ----

def _write_record_json(record: dict) -> Path:
    """将单条 inventory record 写出为独立 JSON 文件。"""
    RECORDS_DIR.mkdir(parents=True, exist_ok=True)
    path = RECORDS_DIR / f"{record['video_id']}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return path


# ---- 辅助: 单视频质量检测 ----

def _run_quality_detection(
    record: dict,
    output_dir: Path,
    skip_pixel_qc: bool = True,
) -> dict:
    """对单条记录运行技术质量检测，返回 results dict。

    EPIC 是预清洗公开基准数据集，默认 skip_pixel_qc=True，
    跳过黑屏检测和坏帧检测（两个都是逐帧解码，30min 视频需 20+ 分钟）。
    保留帧数一致性和时间戳缺口检测（纯数学运算，秒级完成）。
    """
    from zpds_prepare.main import load_config
    from zpds_prepare.readers import epic_reader as rd

    cfg = load_config(CONFIG_PATH)

    video_path = record["video_uri"]
    if not video_path or not Path(video_path).exists():
        return {"stage": "quality_detection", "error": f"视频不存在: {video_path}"}

    # 构建 epic_config
    epic_config: dict = {}
    if record.get("hand_object_uri"):
        epic_config["hand_object_path"] = record["hand_object_uri"]
    if record.get("mask_uri"):
        epic_config["mask_path"] = record["mask_uri"]

    # 读取 Session（后续 segment 生成阶段复用，避免重复加载）
    session = rd.read_session(video_path, config=epic_config if epic_config else None)

    # 运行检测器 (复用 main.py 的检测逻辑)
    from zpds_prepare.detectors.black_frame import detect_black_frames
    from zpds_prepare.detectors.timestamp_gap import detect_timestamp_gaps
    from zpds_prepare.detectors.frame_count import detect_frame_count_mismatch
    from zpds_prepare.detectors.bad_frame import detect_bad_frames
    from zpds_prepare.decisions.segment_planner import plan_segments, get_issue_summary
    from zpds_prepare.writers.quality_writer import write_quality_issues
    from zpds_prepare.writers.candidate_writer import write_segment_candidates

    bd = cfg.get("video", {}).get("black_detection", {})
    black_threshold = bd.get("mean_intensity_threshold", 5.0)
    min_black_duration_s = bd.get("min_duration_s", 0.5)
    edge_tolerance_s = bd.get("edge_tolerance_s", 1.0)
    min_black_duration_ns = int(min_black_duration_s * 1_000_000_000)
    edge_tolerance_ns = int(edge_tolerance_s * 1_000_000_000)

    tv = cfg.get("timestamp", {}).get("video", {})
    video_gap_factor = tv.get("gap_factor", 2.0)
    video_split_gap_ns = int(tv.get("split_gap_s", 0.5) * 1_000_000_000)

    seg_cfg = cfg.get("segment", {})
    min_duration_ns = int(seg_cfg.get("min_duration_s", 1.0) * 1_000_000_000)
    max_duration_ns = int(seg_cfg.get("max_duration_s", 120.0) * 1_000_000_000)

    all_issues = []

    for stream_id, vs in session.video_streams.items():
        # 帧数一致性
        ann_max = None
        for ann_s in session.annotation_streams.values():
            if ann_s.records:
                max_idx = max(r["frame_index"] for r in ann_s.records)
                ann_max = max(ann_max, max_idx + 1) if ann_max is not None else max_idx + 1
        fc_issues = detect_frame_count_mismatch(
            stream_id=stream_id,
            timestamps_ns=vs.timestamps_ns,
            declared_count=vs.frame_count,
            timestamp_count=len(vs.timestamps_ns),
            annotation_max_frame=ann_max,
        )
        all_issues.extend(fc_issues)

        # 坏帧（像素级检测，大视频慢，EPIC 默认跳过）
        if not skip_pixel_qc:
            all_issues.extend(detect_bad_frames(
                video_path=vs.video_path,
                timestamps_ns=vs.timestamps_ns,
                stream_id=stream_id,
            ))

        # 黑屏（像素级检测，大视频慢，EPIC 默认跳过）
        if not skip_pixel_qc:
            all_issues.extend(detect_black_frames(
                video_path=vs.video_path,
                timestamps_ns=vs.timestamps_ns,
                mean_intensity_threshold=black_threshold,
                min_duration_ns=min_black_duration_ns,
                edge_tolerance_ns=edge_tolerance_ns,
            ))

        # 时间戳缺口
        expected_interval_ns = int(1_000_000_000 / vs.fps)
        all_issues.extend(detect_timestamp_gaps(
            timestamps_ns=vs.timestamps_ns,
            expected_interval_ns=expected_interval_ns,
            gap_factor=video_gap_factor,
            split_gap_ns=video_split_gap_ns,
            stream_id=stream_id,
        ))

    # 写出 quality_issues.json
    output_dir.mkdir(parents=True, exist_ok=True)
    write_quality_issues(
        output_path=output_dir / "quality_issues.json",
        issues=all_issues,
        source_session_id=session.session_id,
    )

    # 生成候选 Segment
    candidates = plan_segments(
        issues=all_issues,
        session_start_ns=session.session_start_ns,
        session_end_ns=session.session_end_ns,
        min_duration_ns=min_duration_ns,
        max_duration_ns=max_duration_ns,
    )

    write_segment_candidates(
        output_path=output_dir / "segment_candidates.json",
        candidates=candidates,
        source_session_id=session.session_id,
        source_start_ns=session.session_start_ns,
        source_end_ns=session.session_end_ns,
    )

    summary = get_issue_summary(all_issues)
    return {
        "stage": "quality_detection",
        "status": "ok",
        "session": session,
        "session_id": session.session_id,
        "issues_total": summary["total"],
        "candidate_count": len(candidates),
        "video_frames": session.primary_video.frame_count,
        "annotation_frames": sum(
            len(s.records) for s in session.annotation_streams.values()
        ),
    }


# ---- 辅助: 单视频 Prepared Segment 生成 ----

def _run_segment_generation(
    record: dict,
    candidates_path: Path,
    output_root: Path,
    session=None,
    epic_fields_root: str | None = None,
    experience_dir: str | None = None,
    experience_version: str | None = None,
) -> list[dict]:
    """对单条记录的所有候选 Segment 运行 batch_prepare。

    session 由 QC 阶段传入，避免重复 read_session（EPIC 视频大，加载慢）。
    """
    from batch_prepare import generate_segment

    from segment.epic_fields_calibration import (
        load_epic_fields_calibration,
        missing_epic_fields_calibration,
    )

    video_id = record["video_id"]
    try:
        if epic_fields_root is None:
            raise FileNotFoundError("未提供 --epic-fields-root")
        calibration = load_epic_fields_calibration(epic_fields_root, video_id)
    except FileNotFoundError:
        calibration = missing_epic_fields_calibration(video_id, epic_fields_root)
    coverage_status = calibration["coverage"]["status"]

    if not candidates_path.exists():
        return [{
            "status": "fail",
            "error": f"候选文件不存在: {candidates_path}",
            "epic_fields_coverage": coverage_status,
        }]

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)

    candidates = candidates_doc.get("segments", [])
    source_session_id = candidates_doc.get("source_session_id", "epic_session")

    # Session 由调用方传入（已由 QC 阶段加载）
    video_path = record["video_uri"]

    # source_assets（EPIC 视频 5GB+，跳过 SHA-256 避免读盘数分钟）
    from segment.segment_writer import sha256_hex
    source_assets = []
    if Path(video_path).exists():
        source_assets.append({
            "source_asset_id": "raw_color_0",
            "uri": Path(video_path).name,
            "sha256": "",  # EPIC: 跳过大文件哈希
        })
    for stream_id, ann_s in session.annotation_streams.items():
        pkl_path = ann_s.source_path
        if pkl_path.exists():
            source_assets.append({
                "source_asset_id": f"raw_{stream_id}_pkl",
                "uri": str(pkl_path),
                "media_type": "application/python-pickle",
                "sha256": sha256_hex(str(pkl_path)),
                "ground_truth_status": "model_generated",
            })

    cfg = yaml.safe_load(open(CONFIG_PATH, "r", encoding="utf-8"))

    results = []
    for idx, cand in enumerate(candidates):
        seg_id = f"seg_{idx + 1:06d}"
        seg_dir = str(output_root / seg_id)

        try:
            result = generate_segment(
                dataset_path=video_path,
                source_start_ns=cand["source_start_ns"],
                source_end_ns=cand["source_end_ns"],
                segment_id=seg_id,
                output_dir=seg_dir,
                session=session,
                calibration=calibration,
                cfg=cfg,
                session_id=source_session_id,
                revision="r0001",
                quality_issues=cand.get("issues_in_span"),
                profile="epic",
                source_assets=source_assets,
                experience_dir=experience_dir,
                experience_version=experience_version,
            )
            result["epic_fields_coverage"] = coverage_status
            results.append(result)
        except Exception:
            results.append({
                "segment_id": seg_id,
                "status": "fail",
                "error": traceback.format_exc(),
                "epic_fields_coverage": coverage_status,
            })

    return results


# ---- 主入口 ----

def main():
    parser = argparse.ArgumentParser(
        description="EPIC-KITCHENS-100 批量 Prepared Segment 生成"
    )
    parser.add_argument(
        "--inventory", "-i",
        required=True,
        help="inventory.json 路径 (由 scripts/inspect_epic_dataset.py 生成)",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="最多处理 N 条记录",
    )
    parser.add_argument(
        "--participant", default=None,
        help="仅处理指定参与者 (如 P01)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="跳过已有 batch_summary 中成功的记录",
    )
    parser.add_argument(
        "--failed-only", action="store_true",
        help="仅重试上一轮失败的记录",
    )
    parser.add_argument(
        "--skip-quality", action="store_true",
        help="跳过质量检测（使用已有的 segment_candidates.json）",
    )
    parser.add_argument(
        "--config", default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    parser.add_argument(
        "--epic-fields-root",
        default=None,
        help="EPIC-Fields JSON 根目录；未覆盖视频保持原 RGB 并记录状态",
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

    # ---- 加载 inventory ----
    inv_path = Path(args.inventory)
    if not inv_path.exists():
        print(f"错误: inventory 文件不存在: {inv_path}")
        return 1

    with open(inv_path, "r", encoding="utf-8") as f:
        inventory = json.load(f)

    all_records = inventory.get("records", [])
    print(f"Inventory: {len(all_records)} 条记录")

    # ---- 过滤 ----
    records = [r for r in all_records if r.get("status") == "ready"]
    print(f"  ready: {len(records)} 条")

    if args.participant:
        records = [r for r in records if r["participant_id"] == args.participant]
        print(f"  participant={args.participant}: {len(records)} 条")

    # --resume / --failed-only
    if args.resume and BATCH_SUMMARY_PATH.exists():
        with open(BATCH_SUMMARY_PATH, "r", encoding="utf-8") as f:
            prev = json.load(f)
        succeeded_ids = {
            s.get("video_id") for s in prev.get("segments", [])
            if s.get("status") == "ok" and s.get("video_id")
        }
        records = [r for r in records if r["video_id"] not in succeeded_ids]
        print(f"  resume: 跳过 {len(succeeded_ids)} 条已成功 → 剩余 {len(records)} 条")

    if args.failed_only and BATCH_SUMMARY_PATH.exists():
        with open(BATCH_SUMMARY_PATH, "r", encoding="utf-8") as f:
            prev = json.load(f)
        failed_ids = {
            s.get("video_id") for s in prev.get("segments", [])
            if s.get("status") == "fail" and s.get("video_id")
        }
        records = [r for r in records if r["video_id"] in failed_ids]
        print(f"  failed-only: {len(records)} 条待重试")

    if args.limit:
        records = records[:args.limit]
        print(f"  limit={args.limit}: 实际处理 {len(records)} 条")

    if not records:
        print("没有符合条件的记录，退出。")
        return 0

    # ---- 逐条处理 ----
    total_start = time.time()
    batch_results: list[dict] = []
    succeeded = 0
    failed = 0
    skipped = 0
    coverage_summary = {"covered": 0, "missing_calibration": 0}

    for i, rec in enumerate(records):
        video_id = rec["video_id"]
        print(f"\n{'=' * 60}")
        print(f"  [{i + 1}/{len(records)}] {video_id}")
        print(f"{'=' * 60}")

        t0 = time.time()
        result = {
            "video_id": video_id,
            "participant_id": rec["participant_id"],
            "stages": {},
        }

        try:
            # ① 写出 record JSON
            rec_path = _write_record_json(rec)
            print(f"  ① record JSON: {rec_path}")

            # ② 质量检测
            output_dir = Path("output/epic") / video_id
            if not args.skip_quality:
                print(f"  ② 质量检测 → {output_dir}")
                qd_result = _run_quality_detection(rec, output_dir)
                result["stages"]["quality_detection"] = qd_result
                session = qd_result.pop("session", None)
                if "error" in qd_result:
                    print(f"    ✗ 失败: {qd_result['error']}")
                    failed += 1
                    result["status"] = "fail"
                    result["error"] = qd_result["error"]
                    batch_results.append(result)
                    continue
                print(f"    ✓ {qd_result.get('issues_total', 0)} 个质量问题, "
                      f"{qd_result.get('candidate_count', 0)} 个候选")
            else:
                print("  ② 跳过质量检测")
                # 跳过 QC 时仍需加载 session
                from zpds_prepare.readers import epic_reader as er
                epic_config: dict = {}
                if rec.get("hand_object_uri"):
                    epic_config["hand_object_path"] = rec["hand_object_uri"]
                if rec.get("mask_uri"):
                    epic_config["mask_path"] = rec["mask_uri"]
                session = er.read_session(rec["video_uri"], config=epic_config if epic_config else None)

            # ③ 生成 Prepared Segment（复用 session）
            candidates_path = output_dir / "segment_candidates.json"
            seg_root = PREPARED_ROOT / video_id
            print(f"  ③ 生成 Prepared Segment → {seg_root}")
            seg_results = _run_segment_generation(
                rec,
                candidates_path,
                seg_root,
                session=session,
                epic_fields_root=args.epic_fields_root,
                experience_dir=args.experience_dir,
                experience_version=args.experience_version,
            )
            result["stages"]["segment_generation"] = seg_results
            coverage_status = next(
                (entry.get("epic_fields_coverage") for entry in seg_results),
                "missing_calibration",
            )
            result["epic_fields_coverage"] = coverage_status
            coverage_summary[coverage_status] = coverage_summary.get(coverage_status, 0) + 1

            seg_pass = sum(1 for s in seg_results if s.get("status") == "pass")
            seg_fail = sum(1 for s in seg_results if s.get("status") == "fail")
            print(f"    Segments: {len(seg_results)} 个 ({seg_pass} pass, {seg_fail} fail)")

            if seg_fail > 0:
                for s in seg_results:
                    if s.get("status") == "fail":
                        print(f"      ✗ {s.get('segment_id')}: {s.get('error', '?')[:120]}")

            result["status"] = "ok" if seg_fail == 0 else "partial"
            succeeded += 1

        except Exception:
            elapsed = time.time() - t0
            result["status"] = "fail"
            result["error"] = traceback.format_exc()
            result["elapsed_s"] = round(elapsed, 1)
            failed += 1
            print(f"  ✗ 未捕获异常: {result['error'][:200]}")

        result["elapsed_s"] = round(time.time() - t0, 1)
        batch_results.append(result)
        print(f"  耗时: {result['elapsed_s']:.1f}s")

    # ---- 批量汇总 ----
    total_elapsed = time.time() - total_start
    total_segments = sum(
        len(r.get("stages", {}).get("segment_generation", []))
        for r in batch_results
    )

    summary = {
        "schema_version": "batch_prepare_epic.v1",
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "inventory_path": str(inv_path.resolve()),
        "total_records": len(records),
        "succeeded": succeeded,
        "failed": failed,
        "skipped": skipped,
        "total_segments": total_segments,
        "total_elapsed_s": round(total_elapsed, 1),
        "epic_fields_coverage": coverage_summary,
        "results": batch_results,
    }

    BATCH_SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(BATCH_SUMMARY_PATH, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print("  批量完成")
    print(f"{'=' * 60}")
    print(f"  总数:       {len(records)}")
    print(f"  ✓ 成功:     {succeeded}")
    if failed > 0:
        print(f"  ✗ 失败:     {failed}")
    if skipped > 0:
        print(f"  ⊘ 跳过:     {skipped}")
    print(f"  Segments:   {total_segments}")
    print(f"  总耗时:     {total_elapsed:.1f}s")
    print(f"  汇总文件:   {BATCH_SUMMARY_PATH.resolve()}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
