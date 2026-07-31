"""
A2D 端到端 Pipeline — 从原始 Episode 到 Prepared Segment + 验证。

流程:
  1. Inventory 扫描
  2. HDF5 Schema 探测
  3. A2D Reader（读 Session）
  4. 通用视频 QC + 机器人时序 QC
  5. 候选 Segment
  6. Prepared Segment 写出
  7. Validator

用法:
    python scripts/pipeline_a2d.py E:/datasets/真机/A2D/
    python scripts/pipeline_a2d.py E:/datasets/真机/A2D/ --episode-id 8032
    python scripts/pipeline_a2d.py E:/datasets/真机/A2D/ --skip-inventory --skip-schema
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 确保项目根在 sys.path 上（无需手动设 PYTHONPATH）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ======================================================================
# Pipeline
# ======================================================================

def step_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 60}")


def run_pipeline(
    episode_path: str,
    episode_id: str | None = None,
    output_base: str | None = None,
    *,
    skip_inventory: bool = False,
    skip_schema: bool = False,
    skip_alignment: bool = False,
    skip_mcap: bool = True,   # MCAP 解析较重，默认跳过
    target_fps: float = 30.0,
    revision: str = "r0001",
    experience_dir: str | None = None,
    experience_version: str | None = None,
) -> int:
    """执行 A2D 完整 Pipeline。

    Returns:
        0 成功，1 失败。
    """
    episode_root = Path(episode_path).resolve()
    if not episode_root.is_dir():
        print(f"错误: Episode 目录不存在: {episode_root}", file=sys.stderr)
        return 1

    # ---- 推断 episode_id ----
    if episode_id is None:
        episode_id = _infer_episode_id(episode_root)
    print(f"Episode ID: {episode_id}")
    print(f"Episode 路径: {episode_root}")

    # ---- 输出目录 ----
    if output_base is None:
        output_dir = Path(f"output/a2d/{episode_id}")
    else:
        output_dir = Path(output_base) / episode_id
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {output_dir.resolve()}")

    pipeline_start = time.time()

    # ================================================================
    # Step 1: Inventory
    # ================================================================
    if not skip_inventory:
        step_header(1, "Inventory 扫描")
        from scripts.inspect_a2d_episode import build_inventory

        inventory = build_inventory(str(episode_root), episode_id)
        inv_path = output_dir / "inventory.json"
        with open(inv_path, "w", encoding="utf-8") as f:
            json.dump(inventory, f, indent=2, ensure_ascii=False)
        print(f"  → {inv_path}")

        s = inventory["summary"]
        print(f"  资产完整: {'✓' if s['all_assets_present'] else '✗'}")
        print(f"  MCAP 完整: {'✓' if s['all_mcap_present'] else '✗'}")
        print(f"  相机帧完整: {'✓' if s['all_camera_frames_complete'] else '✗'}")
        print(f"  总完成度: {'✓' if s['is_fully_complete'] else '✗ 部分不完整'}")

    # ================================================================
    # Step 2: HDF5 Schema
    # ================================================================
    if not skip_schema:
        step_header(2, "HDF5 Schema 探测")
        from scripts.inspect_a2d_hdf5 import probe as probe_hdf5

        aligned_path = episode_root / "aligned_joints.h5"
        raw_path = episode_root / "record" / "raw_joints.h5"
        joint_map_path = episode_root / "parameters" / "meshes" / "joint_map.json"

        if aligned_path.is_file() and raw_path.is_file():
            schema_report = probe_hdf5(
                aligned_path=str(aligned_path),
                raw_path=str(raw_path),
                joint_map_path=str(joint_map_path) if joint_map_path.is_file() else None,
            )
            schema_path = output_dir / "hdf5_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump(schema_report, f, indent=2, ensure_ascii=False, default=str)
            print(f"  → {schema_path}")

            aa = schema_report["aligned_joints"]["analysis"]
            print(f"  Timestamp: {aa['timestamp_unit']} / {aa['timestamp_clock']} "
                  f"({aa['timestamp_status']})")
            print(f"  DOF: state={aa['robot_state_dof']}, "
                  f"action={aa['robot_action_dof']}, "
                  f"gripper={aa['gripper_dof']}")
        else:
            print(f"  ⚠ aligned_joints.h5 或 raw_joints.h5 不存在，跳过 Schema 探测")

    # ================================================================
    # Step 2b: State Alignment Report（aligned vs raw）
    # ================================================================
    if not skip_alignment:
        step_header("2b", "State Alignment Report (aligned vs raw)")
        from segment.a2d_state_alignment import (
            generate_alignment_report,
            write_alignment_report,
        )

        aligned_path = episode_root / "aligned_joints.h5"
        raw_path = episode_root / "record" / "raw_joints.h5"

        if aligned_path.is_file() and raw_path.is_file():
            alignment_report = generate_alignment_report(
                aligned_path=str(aligned_path),
                raw_path=str(raw_path),
            )
            ar_path = write_alignment_report(alignment_report, str(output_dir))
            print(f"  → {ar_path}")
            print(f"  结论: {alignment_report.get('conclusion', 'unknown')}")
            if "message_counts" in alignment_report:
                mc = alignment_report["message_counts"]
                ratios = mc.get("resample_ratios", {})
                for group, ratio in ratios.items():
                    print(f"  {group}: raw={mc['raw'].get(f'{group}_positions','?')}, "
                          f"aligned={mc['aligned'].get(f'{group}_positions','?')}, "
                          f"ratio={ratio}:1")
        else:
            print(f"  ⚠ aligned_joints.h5 或 raw_joints.h5 不存在，跳过")

    # ================================================================
    # Step 3: A2D Reader（三路 RGB 时间恢复 + State/Action 读取）
    # ================================================================
    step_header(3, "A2D Reader — 读取 Session")
    from zpds_prepare.readers.a2d_reader import read_session

    session = read_session(episode_root)
    print(f"  Session ID:    {session.session_id}")
    print(f"  视频流:        {list(session.video_streams.keys())}")
    print(f"  时序流:        {list(session.time_series_streams.keys())}")
    print(f"  相机帧数:      {session.meta.get('camera_frame_count', '?')}")
    print(f"  Aligned 样本:  {session.meta.get('aligned_samples', '?')}")
    # 确认三路相机时间推断方法
    for cam_id, vs in session.video_streams.items():
        methods = set(
            f.get("timestamp_method", "?") for f in vs.index_frames
        )
        print(f"    {cam_id}: {vs.frame_count} 帧, "
              f"时间推断方法: {methods}")

    # ================================================================
    # Step 4: 通用视频 QC + 机器人时序 QC → 候选 Segment
    # ================================================================
    step_header(4, "质量检测 + 候选 Segment")
    from zpds_prepare.segmentation.a2d_segmenter import run_and_write

    config = {
        "min_duration_s": 1.0,
        "max_duration_s": 300.0,
    }

    sc_path = run_and_write(
        session=session,
        output_dir=output_dir,
        config=config,
        alignment_dir=None,  # 第一轮无 alignment 目录
    )
    print(f"  → {sc_path}")

    # 读取结果摘要
    with open(sc_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)
    candidates = candidates_doc.get("segments", [])
    print(f"  候选 Segment: {len(candidates)} 个")
    for c in candidates:
        dur = (c["source_end_ns"] - c["source_start_ns"]) / 1e9
        print(f"    {c['candidate_id']}: {dur:.3f}s "
              f"({len(c.get('issues_in_span', []))} issues)")

    if not candidates:
        print("\n  没有有效候选 Segment，Pipeline 结束。")
        return 0

    # ================================================================
    # Step 5: Prepared Segment 写出
    # ================================================================
    step_header(5, "Prepared Segment 生成")
    from scripts.prepare_a2d_segment import prepare_segment

    output_base_path = output_dir / "prepared_segments"
    all_segments = []

    for i, candidate in enumerate(candidates, start=1):
        seg = prepare_segment(
            candidate=candidate,
            session=session,
            output_base=output_base_path,
            segment_index=i,
            revision=revision,
            experience_dir=experience_dir,
            experience_version=experience_version,
        )
        all_segments.append(seg)

        qual = seg.get("quality", {})
        print(f"  {seg['segment_id']}: "
              f"{[s['stream_id'] for s in seg['streams']]} | "
              f"quality={qual.get('status', 'pass')} "
              f"({len(qual.get('issues', []))} issues)")

    # ================================================================
    # Step 6: Validator
    # ================================================================
    step_header(6, "Validator")
    from segment.a2d_validator import validate_segment, write_validation_report

    all_pass = 0
    all_warn = 0
    all_fail = 0

    for seg_dir in sorted(output_base_path.iterdir()):
        if not seg_dir.is_dir() or not (seg_dir / "segment.json").is_file():
            continue

        report = validate_segment(seg_dir)
        write_validation_report(report, seg_dir)

        status = report["status"]
        if status == "pass":
            all_pass += 1
        elif status == "pass_with_warning":
            all_warn += 1
        else:
            all_fail += 1

        print(f"  {seg_dir.name}: {status}")
        for check_id, result in report["checks"].items():
            icon = {"pass": "✓", "warning": "⚠", "fail": "✗"}.get(result, "?")
            print(f"    {icon} {check_id}: {result}")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = time.time() - pipeline_start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline 完成")
    print(f"  总耗时:          {elapsed:.1f}s")
    print(f"  候选 Segment:    {len(candidates)}")
    print(f"  Prepared:        {len(all_segments)}")
    print(f"  验证 Pass:       {all_pass}")
    print(f"  验证 Pass+Warn:  {all_warn}")
    print(f"  验证 Fail:       {all_fail}")
    print(f"  输出目录:        {output_dir.resolve()}")
    print(f"{'=' * 60}")

    return 0 if all_fail == 0 else 1


def _infer_episode_id(root: Path) -> str:
    """从 review JSON 文件名或目录名推断 episode_id。"""
    review_dir = root / "review"
    if review_dir.is_dir():
        for f in review_dir.glob("review_*.json"):
            stem = f.stem
            if stem.startswith("review_"):
                return stem[len("review_"):]
    return root.name


# ======================================================================
# CLI
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A2D 端到端 Pipeline — Inventory → Segment → Validate"
    )
    parser.add_argument(
        "episode",
        help="A2D Episode 根目录路径",
    )
    parser.add_argument(
        "--episode-id",
        default=None,
        help="Episode ID（默认从 review 文件名推断）",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出根目录（默认: output/a2d/{episode_id}/）",
    )
    parser.add_argument(
        "--skip-inventory",
        action="store_true",
        help="跳过 Inventory 扫描",
    )
    parser.add_argument(
        "--skip-schema",
        action="store_true",
        help="跳过 HDF5 Schema 探测",
    )
    parser.add_argument(
        "--skip-alignment",
        action="store_true",
        help="跳过 State Alignment Report",
    )
    parser.add_argument(
        "--mcap",
        action="store_true",
        help="同时运行 ROS2 MCAP Schema 探测（较慢）",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="目标 CFR 帧率（默认: 30.0）",
    )
    parser.add_argument(
        "--revision", "-r",
        default="r0001",
        help="record_revision（默认: r0001）",
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

    return run_pipeline(
        episode_path=args.episode,
        episode_id=args.episode_id,
        output_base=args.output,
        skip_inventory=args.skip_inventory,
        skip_schema=args.skip_schema,
        skip_alignment=args.skip_alignment,
        skip_mcap=not args.mcap,
        target_fps=args.target_fps,
        revision=args.revision,
        experience_dir=args.experience_dir,
        experience_version=args.experience_version,
    )


if __name__ == "__main__":
    sys.exit(main())
