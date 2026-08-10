"""
A2D Prepared Segment 生成 — CLI 入口。

从 segment_candidates.json 读取候选分段，对每个候选 Segment：
  ① 图像序列 → CFR H.264 MP4（三路相机）
  ② 生成 {stream_id}_sample_map.parquet
  ③ 规范化机器人时序 → Parquet（4 流）
  ④ 提取 calibration.json（仅内参）
  ⑤ 构建 segment.json
  ⑥ 验证

用法:
    python scripts/prepare_a2d_segment.py output/a2d/8032/segment_candidates.json
    python scripts/prepare_a2d_segment.py -c candidate_000001 output/a2d/8032/
    python scripts/prepare_a2d_segment.py --output-dir prepared_segments/a2d_8032 output/a2d/8032/
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# Allow direct ``python scripts/prepare_a2d_segment.py`` execution.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from segment.a2d_calibration import extract_a2d_calibration, write_calibration
from segment.a2d_depth_copy import (
    copy_depth_sequence,
    generate_depth_sample_map,
    probe_depth_properties,
    write_depth_sample_map,
)
from segment.a2d_review_annotations import (
    build_annotation_stream_entry,
    convert_review_actions,
    write_review_annotations,
)
from segment.a2d_video_transcoder import (
    generate_image_sample_map,
    transcode_image_sequence,
    write_image_sample_map,
)
from segment.image_undistorter import plan_undistortion
from segment.segment_writer import build_segment_json, write_segment_json
from segment.timeseries_normalizer import (
    normalize_time_series,
    write_time_series,
)
from zpds_prepare.readers.a2d_reader import read_session

# 输出流配置
TARGET_FPS = 30.0
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

CAMERA_STREAMS = ["head_rgb", "hand_left_rgb", "hand_right_rgb"]
DEPTH_STREAMS = ["head_depth", "hand_left_depth", "hand_right_depth"]
TIME_SERIES_STREAMS = [
    "robot_state", "robot_action",
    "gripper_state", "gripper_action",
]
OPTIONAL_TS_STREAMS = ["gripper_state", "gripper_action"]


# ======================================================================
# 辅助
# ======================================================================

def step_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 60}")


def _find_episode_root(candidates_path: Path) -> Path | None:
    """从 segment_candidates.json 所在目录推测 Episode 根路径。

    检查 output/a2d/{ep_id}/ 下有无 inventory.json。
    """
    # candidates_path = output/a2d/{ep_id}/segment_candidates.json
    parent = candidates_path.parent  # output/a2d/{ep_id}
    inv_path = parent / "inventory.json"
    if inv_path.is_file():
        with open(inv_path, "r", encoding="utf-8") as f:
            inv = json.load(f)
        ep_path = inv.get("episode_path", "")
        if ep_path and Path(ep_path).is_dir():
            return Path(ep_path)
    return None


# ======================================================================
# 核心流程
# ======================================================================

def prepare_segment(
    candidate: dict,
    session,
    output_base: Path,
    segment_index: int,
    revision: str = "r0001",
    experience_dir: str | Path | None = None,
    experience_version: str | None = None,
    with_privacy: bool = False,
) -> dict:
    """为一个候选 Segment 生成完整 Prepared Segment。

    Args:
        candidate: segment_candidates.json 中 segments[] 的一项。
        session: 已读取的 Session 对象。
        output_base: Prepared Segment 根目录。
        segment_index: 从 1 开始的序号。
        revision: record_revision。
        experience_dir: 可选的 Experience 输出目录；写入已声明的既有标注。
        experience_version: Experience 版本；默认使用 Experience 目录名。
        with_privacy: 对转码产物执行隐私脱敏（A2D profile 人脸不适用、
            文本适用），脱敏版即训练用产物。

    Returns:
        segment dict（segment.json 内容）。
    """
    seg_id = f"seg_{segment_index:06d}"
    seg_dir = output_base / seg_id

    source_start = candidate["source_start_ns"]
    source_end = candidate["source_end_ns"]
    duration_ns = source_end - source_start

    print(f"\n  --- {seg_id} ---")
    print(f"  时间范围: {source_start} → {source_end}")
    print(f"  时长: {duration_ns / 1e9:.3f}s")

    video_results = []
    ts_results = []

    # 先建立每路相机的去畸变映射。映射在整段内复用，且只应用到
    # Prepared 派生产物；原始 JPEG 始终保持不变。
    calib = extract_a2d_calibration(session.meta)
    undistortion_plans = {}
    undistortion_coverage = calib.setdefault("undistortion", {"streams": {}})
    for stream_id in CAMERA_STREAMS:
        vs = session.video_streams.get(stream_id)
        if vs is None:
            continue
        plan = plan_undistortion(calib, stream_id, width=vs.width, height=vs.height)
        undistortion_plans[stream_id] = plan
        undistortion_coverage["streams"][stream_id] = {
            "status": plan.status,
            "detail": plan.detail,
            "operation": {
                "applied": "undistort",
                "identity": "identity",
            }.get(plan.status, "preserve_original"),
            "calibration_source": calib.get("source", {}).get(
                "reference_url",
                calib.get("source", {}).get("uri", ""),
            ),
        }

    # ---- 8.1a: 图像序列转 MP4 ----
    for stream_id in CAMERA_STREAMS:
        vs = session.video_streams.get(stream_id)
        if vs is None:
            print(f"    ⚠ {stream_id}: 流不存在，跳过")
            continue

        output_mp4 = str(seg_dir / "data" / f"{stream_id}.mp4")

        print(f"    转码 {stream_id}...", end=" ", flush=True)
        vr = transcode_image_sequence(
            index_frames=vs.index_frames,
            output_mp4=output_mp4,
            source_start_ns=source_start,
            source_end_ns=source_end,
            target_fps=TARGET_FPS,
            width=vs.width,
            height=vs.height,
            frame_transform=undistortion_plans[stream_id].frame_transform,
        )
        undistortion = undistortion_plans[stream_id]
        geometry_status = "，已去畸变" if undistortion.status == "applied" else ""
        print(f"{vr['output_frames']} 帧{geometry_status}")

        # ---- 8.1b: 生成 sample_map ----
        sample_map = generate_image_sample_map(
            index_frames=vs.index_frames,
            source_start_ns=source_start,
            source_end_ns=source_end,
            target_fps=TARGET_FPS,
        )
        sm_path = write_image_sample_map(sample_map, str(seg_dir), stream_id)
        print(f"    sample_map → {sm_path}")

        video_results.append({
            "stream_id": stream_id,
            "width": vr["width"],
            "height": vr["height"],
            "output_fps": vr["output_fps"],
            "output_frames": vr["output_frames"],
            "sample_map_uri": f"maps/{stream_id}_sample_map.parquet",
            "role": "observation",
            "frame_id": f"{stream_id}_optical",
            "undistorted": undistortion.status == "applied",
            "undistortion": undistortion_coverage["streams"][stream_id],
        })

    # ---- 8.1a2: 隐私脱敏（可选） ----
    # 对转码产物原地脱敏（A2D profile 人脸不适用、文本适用），脱敏版即
    # 训练用产物。redaction 字段写入 video_results，由 build_segment_json
    # 透传进 segment.json 的 streams[].redaction 条目。
    if with_privacy:
        from zpds.privacy.segment_redaction import redact_segment_videos

        video_meta = [
            {"output_mp4": str(seg_dir / "data" / f"{vr['stream_id']}.mp4")}
            for vr in video_results
        ]
        redacted_count = redact_segment_videos(
            video_meta, video_results, seg_dir, "a2d",
        )
        print(f"  隐私脱敏: {redacted_count} 个视频流")

    # ---- 8.1c: 深度图像序列（拷贝 PNG，不转码） ----
    depth_results = []
    for stream_id in DEPTH_STREAMS:
        ds = session.video_streams.get(stream_id)
        if ds is None:
            print(f"    ⚠ {stream_id}: 深度流不存在，跳过")
            continue

        depth_out_dir = str(seg_dir / "data" / "depth" / stream_id)

        try:
            dr = copy_depth_sequence(
                index_frames=ds.index_frames,
                output_dir=depth_out_dir,
                source_start_ns=source_start,
                source_end_ns=source_end,
            )
        except ValueError as e:
            print(f"    ⚠ {stream_id}: 无深度帧 ({e})，跳过")
            continue

        print(f"    深度 {stream_id}: {dr['copied_frames']}/{dr['total_in_span']} 帧 "
              f"({dr['width']}×{dr['height']}, {dr['dtype']})")

        # 深度 sample_map
        depth_sm = generate_depth_sample_map(
            index_frames=ds.index_frames,
            source_start_ns=source_start,
            source_end_ns=source_end,
        )
        dsm_path = write_depth_sample_map(depth_sm, str(seg_dir), stream_id)
        print(f"    depth_sample_map → {dsm_path}")

        # 深度属性探测
        depth_props = probe_depth_properties(
            ds.index_frames, source_start, source_end,
        )
        print(f"    零值比例: {depth_props.get('zero_ratio', '?')}, "
              f"max: {depth_props.get('max_value', '?')}")

        depth_results.append({
            "stream_id": stream_id,
            "width": dr["width"],
            "height": dr["height"],
            "dtype": dr["dtype"],
            "copied_frames": dr["copied_frames"],
            "sample_map_uri": f"maps/{stream_id}_sample_map.parquet",
            "depth_props": depth_props,
        })

    # ---- 8.2: 机器人时序 ----
    for stream_id in TIME_SERIES_STREAMS:
        ts = session.time_series_streams.get(stream_id)
        if ts is None:
            if stream_id in OPTIONAL_TS_STREAMS:
                print(f"    ⚠ {stream_id}: 可选流不存在，跳过")
                continue
            else:
                print(f"    ✗ {stream_id}: 必需流不存在！")
                continue

        try:
            df = normalize_time_series(ts, source_start, source_end)
        except ValueError as e:
            if stream_id in OPTIONAL_TS_STREAMS:
                print(f"    ⚠ {stream_id}: 范围内无数据，跳过 ({e})")
                continue
            raise

        out_path = write_time_series(df, str(seg_dir), stream_id)
        print(f"    {stream_id} → {out_path} ({len(df)} 行)")

        ts_results.append({
            "stream_id": stream_id,
            "uri": f"data/{stream_id}.parquet",
            "rows": len(df),
        })

    # ---- 8.3: calibration.json ----
    calib_path = write_calibration(calib, str(seg_dir))
    print(f"    calibration → {calib_path}")
    print(f"    相机: {[c['stream_id'] for c in calib['cameras']]}")
    print(f"    外参状态: {calib['extrinsics_status']}")

    # ---- 9: metadata/joint_names.json ----
    robot_state = session.time_series_streams.get("robot_state")
    if robot_state is not None and "joint_names" in robot_state.metadata:
        jn_path = _write_joint_names(
            seg_dir, robot_state.metadata["joint_names"]
        )
        print(f"    joint_names → {jn_path}")

    # ---- 10: review 动作标注 ----
    review_annotations = None
    episode_root = Path(session.source_path)
    episode_id = session.meta.get("episode_id", "")
    review_path = episode_root / "review" / f"review_{episode_id}.json"
    if review_path.is_file():
        # 从 session 获取 aligned timestamps
        aligned_ts = robot_state.timestamps_ns if robot_state else []
        head_rgb_sm_path = str(seg_dir / "maps" / "head_rgb_sample_map.parquet")

        review_df = convert_review_actions(
            review_path=str(review_path),
            aligned_timestamps_ns=aligned_ts,
            segment_start_ns=source_start,
            segment_end_ns=source_end,
            rgb_sample_map_path=head_rgb_sm_path,
        )
        if len(review_df) > 0:
            ra_path = write_review_annotations(review_df, str(seg_dir))
            print(f"    review_actions → {ra_path} ({len(review_df)} actions)")
            review_annotations = build_annotation_stream_entry()
        else:
            print("    review_actions: 无动作与此 Segment 重叠")
    else:
        print(f"    review_actions: review JSON 不存在 ({review_path})")

    # ---- 构建 segment.json ----
    span = {
        "source_start_ns": source_start,
        "source_end_ns": source_end,
    }

    # 收集该候选段的 quality issues
    quality_issues = candidate.get("issues_in_span", [])

    segment = build_segment_json(
        dataset_path=str(session.source_path),
        span=span,
        video_results=video_results,
        imu_results=[],
        calibration_id=calib["calibration_id"],
        revision=revision,
        segment_id=seg_id,
        session_id=session.session_id,
        quality_issues=quality_issues,
        source_assets=_build_source_assets(session),
        profile="a2d",
    )

    # 追加 time_series streams 到 segment
    _append_ts_streams(segment, ts_results, duration_ns, session)

    # 追加 depth streams 到 segment
    _append_depth_streams(segment, depth_results, duration_ns)

    # 追加 review annotation stream 到 segment
    if review_annotations is not None:
        segment["streams"].append(review_annotations)

    seg_path = write_segment_json(segment, str(seg_dir))
    print(f"    segment.json → {seg_path}")

    if experience_dir is not None:
        from zpds.annotation.importer import import_segment_annotations

        manifest = import_segment_annotations(
            seg_dir,
            experience_dir,
            experience_version=experience_version,
        )
        if manifest is not None:
            print(f"    existing annotations → {manifest}")

    return segment


def _build_source_assets(session) -> list[dict]:
    """构建 source_assets 列表。"""
    assets = []
    source_path = Path(session.source_path)

    # meta_info.json
    meta_path = source_path / "meta_info.json"
    if meta_path.is_file():
        assets.append({
            "source_asset_id": "raw_meta_info",
            "uri": "meta_info.json",
            "sha256": _sha256(meta_path),
        })

    # aligned_joints.h5
    h5_path = source_path / "aligned_joints.h5"
    if h5_path.is_file():
        assets.append({
            "source_asset_id": "raw_aligned_joints",
            "uri": "aligned_joints.h5",
            "sha256": _sha256(h5_path),
        })

    return assets


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


# 需要 joint_names_uri 的时序字段组（关节维度组）
_JOINT_FIELD_GROUPS = {
    "positions", "velocities", "efforts", "temperatures",
    "accelerations", "decelerations", "torque_rates",
}


def _group_fields_by_source(fields: list[dict]) -> list[dict]:
    """将扁平字段列表按 source_hdf5_path 分组为结构化字段描述。

    例如 72 个 robot_state 字段 →
      [{name: "positions",   shape: [18], unit: "rad", ...},
       {name: "velocities",  shape: [18], unit: "rad/s", ...},
       {name: "efforts",     shape: [18], unit: "N·m", ...},
       {name: "temperatures", shape: [18], unit: "°C", ...}]
    """
    from collections import OrderedDict

    grouped = OrderedDict()
    for f in fields:
        source = f.get("source_hdf5_path", "")
        group_name = source.rsplit("/", 1)[-1] if source else f["name"]
        if group_name not in grouped:
            grouped[group_name] = []
        grouped[group_name].append(f)

    result = []
    for group_name, group_fields in grouped.items():
        first = group_fields[0]
        entry = {
            "name": group_name,
            "shape": [len(group_fields)],
            "dtype": "float32",
            "unit": first.get("unit", ""),
            "unit_status": "needs_verification",
        }
        if group_name in _JOINT_FIELD_GROUPS:
            entry["joint_names_uri"] = "metadata/joint_names.json"
        result.append(entry)

    return result


def _write_joint_names(
    seg_dir: Path,
    joint_names: list[str],
    filename: str = "joint_names.json",
) -> str:
    """写出关节名列表到 metadata/ 目录。

    Returns:
        输出文件路径。
    """
    meta_dir = seg_dir / "metadata"
    meta_dir.mkdir(parents=True, exist_ok=True)
    out_path = meta_dir / filename
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(joint_names, f, indent=2, ensure_ascii=False)
    return str(out_path)


def _append_ts_streams(
    segment: dict,
    ts_results: list[dict],
    duration_ns: int,
    session,
) -> None:
    """将 TimeSeries 流添加到 segment.json 的 streams 列表中。

    使用分组字段描述（positions[18]、velocities[18]...）、
    unit_status: needs_verification、joint_names_uri。
    """
    for tsr in ts_results:
        stream_id = tsr["stream_id"]
        ts_stream = session.time_series_streams.get(stream_id)
        if ts_stream is None:
            continue

        # 按 HDF5 source path 分组 → 结构化 fields
        fields_desc = _group_fields_by_source(ts_stream.fields)

        entry = {
            "stream_id": stream_id,
            "role": ts_stream.role,
            "modality": ts_stream.modality,
            "uri": tsr["uri"],
            "format": "parquet",
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
                "timestamp_column": "timestamp_ns",
            },
            "fields": fields_desc,
            "origin": {
                "kind": "source_derived",
                "source_asset_id": "aligned_joints_h5",
                "operation": "time_crop_and_schema_normalize",
            },
        }

        # gripper 流标记为 optional
        if stream_id in OPTIONAL_TS_STREAMS:
            entry["availability"] = "optional"

        segment["streams"].append(entry)


def _append_depth_streams(
    segment: dict,
    depth_results: list[dict],
    duration_ns: int,
) -> None:
    """将深度流添加到 segment.json 的 streams 列表中。"""
    for dr in depth_results:
        depth_props = dr.get("depth_props", {})
        segment["streams"].append({
            "stream_id": dr["stream_id"],
            "role": "observation",
            "modality": "depth",
            "uri": f"data/depth/{dr['stream_id']}/",
            "format": "png_sequence",
            "encoding": "png16",
            "shape": [dr["height"], dr["width"]],
            "dtype": dr["dtype"],
            "unit": "unknown",
            "unit_status": "needs_verification",
            "frame_id": f"{dr['stream_id']}_optical",
            "time": {
                "clock_id": "segment",
                "sampling": "irregular",
            },
            "origin": {
                "kind": "deterministic_transform",
                "source_asset_id": "raw_camera_frames",
                "operation": "trim_copy",
                "sample_map_uri": dr.get("sample_map_uri", ""),
            },
            "quality": {
                "zero_ratio": depth_props.get("zero_ratio"),
                "max_value": depth_props.get("max_value"),
                "resolution_consistent": (
                    isinstance(depth_props.get("width"), int)
                    and isinstance(depth_props.get("height"), int)
                ),
            },
        })


# ======================================================================
# 主入口
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="A2D Prepared Segment 生成 — 图像转码 + 时序规范化 + 标定提取"
    )
    parser.add_argument(
        "source",
        help="segment_candidates.json 路径，或 output 目录（如 output/a2d/8032/）",
    )
    parser.add_argument(
        "--candidate", "-c",
        default=None,
        help="只处理指定 candidate_id（如 candidate_000001）",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=None,
        help="Prepared Segment 输出根目录 (默认: {source_dir}/prepared_segments/)",
    )
    parser.add_argument(
        "--episode",
        default=None,
        help="Episode 根目录路径 (默认从 candidates 同目录的 inventory.json 推断)",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=30.0,
        help="目标 CFR 帧率 (默认: 30.0)",
    )
    parser.add_argument(
        "--revision", "-r",
        default="r0001",
        help="record_revision (默认: r0001)",
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
    parser.add_argument(
        "--with-privacy",
        action="store_true",
        help="对转码后的视频执行隐私脱敏（人脸模糊 + 文本遮挡），"
             "训练集只出脱敏版",
    )
    args = parser.parse_args()

    source_path = Path(args.source)

    # ---- 解析输入 ----
    if source_path.is_file() and source_path.suffix == ".json":
        candidates_path = source_path
    elif source_path.is_dir():
        candidates_path = source_path / "segment_candidates.json"
    else:
        print(f"错误: 找不到 segment_candidates.json: {source_path}", file=sys.stderr)
        return 1

    if not candidates_path.is_file():
        print(f"错误: 文件不存在: {candidates_path}", file=sys.stderr)
        return 1

    with open(candidates_path, "r", encoding="utf-8") as f:
        candidates_doc = json.load(f)

    all_candidates = candidates_doc.get("segments", [])
    if not all_candidates:
        print("没有候选 Segment，退出。")
        return 0

    # 过滤指定 candidate
    if args.candidate:
        all_candidates = [
            c for c in all_candidates
            if c["candidate_id"] == args.candidate
        ]
        if not all_candidates:
            print(f"错误: 找不到 candidate: {args.candidate}", file=sys.stderr)
            return 1

    # ---- 解析 Episode 路径 ----
    episode_root = None
    if args.episode:
        episode_root = Path(args.episode)
    else:
        episode_root = _find_episode_root(candidates_path)

    if episode_root is None:
        print("错误: 无法推断 Episode 根路径，请用 --episode 指定", file=sys.stderr)
        return 1

    # ---- 输出目录 ----
    if args.output_dir:
        output_base = Path(args.output_dir)
    else:
        output_base = candidates_path.parent / "prepared_segments"

    # ---- 读取 Session ----
    print(f"读取 A2D Episode: {episode_root}")
    session = read_session(episode_root)
    print(f"  Session ID: {session.session_id}")
    print(f"  视频流: {list(session.video_streams.keys())}")
    print(f"  时序流: {list(session.time_series_streams.keys())}")

    # 全局 target_fps
    global TARGET_FPS
    TARGET_FPS = args.target_fps

    # ---- 逐个处理候选 Segment ----
    start_time = time.time()

    print(f"\n候选 Segment: {len(all_candidates)} 个")
    print(f"输出目录: {output_base.resolve()}")

    for i, candidate in enumerate(all_candidates, start=1):
        seg = prepare_segment(
            candidate=candidate,
            session=session,
            output_base=output_base,
            segment_index=i,
            revision=args.revision,
            experience_dir=args.experience_dir,
            experience_version=args.experience_version,
            with_privacy=args.with_privacy,
        )

        # 打印 segment 摘要
        print(f"\n  Segment ID: {seg['segment_id']}")
        print(f"  Streams:    {[s['stream_id'] for s in seg['streams']]}")
        qual = seg.get("quality", {})
        print(f"  Quality:    {qual.get('status', 'pass')} "
              f"({len(qual.get('issues', []))} issues)")

    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print(f"  完成 — {len(all_candidates)} 个 Segment，耗时 {elapsed:.1f}s")
    print(f"  输出: {output_base.resolve()}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
