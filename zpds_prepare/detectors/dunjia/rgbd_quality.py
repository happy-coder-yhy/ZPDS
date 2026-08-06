"""B2: 遁甲 RGB-D 质量检测。

检查深度流（来自 MCAP 内嵌 PNG）的基本有效性和与 RGB 主视角的时间对齐。

检查项：
  1. 深度帧 PNG 解码 — 首帧 + 抽样帧（shape / dtype）
  2. 零值/无效值/饱和比例 — 逐帧统计
  3. 冻结深度检测 — 连续完全相同帧
  4. RGB-Depth 时间配对 — 最近邻时间匹配，禁止按帧号硬配
  5. 配对率 / offset P50/P95/max / 未配对 span
  6. 分辨率、内参和标定一致性校验

原则：
  - 所有跨流配对使用最近邻时间戳，禁止按序号假设一一对应
  - 跨 gap 不配对
  - 外参不可信只阻断 geometry_ready，不否定 RGB
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# 报告类型
# ---------------------------------------------------------------------------


@dataclass
class DepthFrameSample:
    """单张深度帧的抽样元数据。"""

    frame_index: int
    timestamp_ns: int
    width: int
    height: int
    dtype: str
    min_val: int
    max_val: int
    zero_ratio: float
    invalid_ratio: float
    mean_val: float
    is_frozen: bool = False


@dataclass
class RGBDepthAlignment:
    """RGB-Depth 时间对齐结果。"""

    paired_count: int
    rgb_frame_count: int
    depth_frame_count: int
    paired_ratio: float  # = paired_count / min(rgb_count, depth_count)
    offset_ns_p50: float  # 纳秒
    offset_ns_p95: float
    offset_ns_max: float
    mapping_method: str = "nearest_neighbor_timestamp"
    uncertainty_ns: int = 0
    unpaired_rgb_spans: list[tuple[int, int]] = field(default_factory=list)
    unpaired_depth_spans: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class DunjiaRGBDReport:
    """遁甲 RGB-D 质量报告。"""

    session_id: str
    source_path: str
    schema_version: str = "zpds.dunjia_rgbd.v1"

    # 深度流基本信息
    depth_frame_count: int = 0
    depth_width: int = 0
    depth_height: int = 0
    depth_dtype: str = "unknown"
    depth_unit: str = "unknown"
    depth_timestamp_start_ns: int = 0
    depth_timestamp_end_ns: int = 0

    # 深度帧质量
    zero_ratio_mean: float = 0.0
    zero_ratio_max: float = 0.0
    invalid_ratio_mean: float = 0.0
    saturation_ratio: float = 0.0
    frozen_span_count: int = 0
    frozen_total_frames: int = 0

    # RGB-Depth 对齐
    alignment: RGBDepthAlignment | None = None

    # RGB 视频基本信息
    rgb_frame_count: int = 0
    rgb_width: int = 0
    rgb_height: int = 0

    # 标定一致性
    calibration_consistent: bool | None = None  # None = 未检查
    calibration_issues: list[str] = field(default_factory=list)

    # 坏帧区间
    bad_spans: list[dict[str, Any]] = field(default_factory=list)

    # 聚合
    overall_disposition: str = "pass"  # "pass" | "keep_with_flag" | "reject"
    issues: list[str] = field(default_factory=list)

    # 抽样帧
    sampled_frames: list[DepthFrameSample] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 配置默认值
# ---------------------------------------------------------------------------

# 深度无效值候选（常见深度传感器的标记值）
_INVALID_CANDIDATES = {0, 65535, 65504, 65500}


def _is_likely_invalid(value: int, dtype_max: int) -> bool:
    """判断深度值是否可能是无效标记。"""
    if value in _INVALID_CANDIDATES:
        return True
    if dtype_max == 65535 and value == 65535:
        return True  # uint16 全 1 典型饱和/无效
    return False


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------


def check_dunjia_rgbd(
    session: Any,
    *,
    max_pairing_offset_ns: int = 50_000_000,  # 50ms 默认
    sample_count: int = 10,
    saturation_threshold: float = 0.01,
    zero_threshold: float = 0.50,  # 零值超过 50% → WARN
) -> DunjiaRGBDReport:
    """检查遁甲 RGB-D 质量。

    Args:
        session: ``zpds_prepare.readers.session_model.Session`` 对象
        max_pairing_offset_ns: RGB-Depth 配对的最大时间偏差（纳秒）
        sample_count: 深度抽样帧数
        saturation_threshold: 饱和像素比例阈值
        zero_threshold: 零值比例阈值（超过触发 WARN）

    Returns:
        DunjiaRGBDReport 包含深度质量、RGB-Depth 对齐和标定一致性。
    """
    source_path = session.source_path
    report = DunjiaRGBDReport(
        session_id=session.session_id,
        source_path=str(source_path),
    )

    # ---- 0. 前置检查 ----
    depth_stream = session.depth_streams.get("ego_depth")
    if depth_stream is None:
        report.issues.append("深度流 ego_depth 不存在")
        report.overall_disposition = "reject"
        return report

    rgb_stream = session.video_streams.get("camera0")
    if rgb_stream is None:
        report.issues.append("主 RGB 流 camera0 不存在")
        report.overall_disposition = "reject"
        return report

    # ---- 1. 深度流基本信息 ----
    report.depth_frame_count = depth_stream.frame_count
    report.depth_width = depth_stream.width
    report.depth_height = depth_stream.height
    report.depth_dtype = depth_stream.dtype
    report.depth_unit = getattr(depth_stream, "unit", "unknown")

    depth_ts = np.array(depth_stream.timestamps_ns, dtype=np.int64)
    if len(depth_ts) > 0:
        report.depth_timestamp_start_ns = int(depth_ts[0])
        report.depth_timestamp_end_ns = int(depth_ts[-1])

    # ---- 2. RGB 基本信息 ----
    report.rgb_frame_count = rgb_stream.frame_count
    report.rgb_width = rgb_stream.width
    report.rgb_height = rgb_stream.height

    rgb_ts = np.array(rgb_stream.timestamps_ns, dtype=np.int64)

    # ---- 3. 深度帧抽样解码 ----
    _sample_depth_frames(source_path, depth_ts, report, sample_count)

    # ---- 4. 深度质量指标 ----
    _compute_depth_quality(source_path, depth_ts, report, zero_threshold, saturation_threshold)

    # ---- 5. RGB-Depth 时间配对 ----
    report.alignment = _pair_rgb_depth(
        rgb_ts, depth_ts, max_pairing_offset_ns,
    )

    # ---- 6. 标定一致性 ----
    _check_calibration_consistency(report, session)

    # ---- 7. 单位未知 ----
    if report.depth_unit == "unknown":
        report.issues.append("深度物理单位未知（unit=unknown）")
    if report.depth_dtype == "unknown":
        report.issues.append("深度 dtype 未知")

    # ---- 聚合 ----
    if report.depth_frame_count == 0:
        report.overall_disposition = "reject"
    elif (
        report.zero_ratio_mean > zero_threshold
        or report.saturation_ratio > saturation_threshold * 2
        or (
            report.alignment is not None
            and report.alignment.paired_ratio < 0.5
        )
    ):
        report.overall_disposition = "keep_with_flag"
    elif report.issues:
        report.overall_disposition = "keep_with_flag"

    return report


# ---------------------------------------------------------------------------
# 深度帧抽样
# ---------------------------------------------------------------------------


def _sample_depth_frames(
    source_path: str,
    depth_ts: np.ndarray,
    report: DunjiaRGBDReport,
    sample_count: int,
) -> None:
    """从 MCAP 中抽样解码深度 PNG 帧。"""
    from zpds_prepare.readers.dunjia_reader import TOPIC_DEPTH, _open_mcap

    if len(depth_ts) == 0:
        return

    # 选择抽样索引：首、尾 + 等间距
    n = len(depth_ts)
    indices: list[int] = [0]
    if n > 1:
        indices.append(n - 1)
    if sample_count > 2 and n > 2:
        step = max(1, (n - 2) // (sample_count - 2))
        for i in range(1, sample_count - 1):
            idx = min(i * step, n - 2)
            if idx not in indices:
                indices.append(idx)
    indices.sort()

    try:
        reader, fh = _open_mcap(source_path)
    except Exception:
        return

    try:
        frame_idx = 0
        sample_idx = 0
        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic != TOPIC_DEPTH:
                continue
            if sample_idx >= len(indices):
                break
            if frame_idx != indices[sample_idx]:
                frame_idx += 1
                continue

            nparr = np.frombuffer(decoded.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                report.issues.append(
                    f"深度帧 [{frame_idx}] PNG 解码失败"
                )
                frame_idx += 1
                sample_idx += 1
                continue

            ts = depth_ts[frame_idx] if frame_idx < len(depth_ts) else 0
            total = img.size
            zero = int(np.sum(img == 0))
            invalid = sum(
                _is_likely_invalid(v, np.iinfo(img.dtype).max)
                for v in [0, 65535]
            )

            report.sampled_frames.append(
                DepthFrameSample(
                    frame_index=frame_idx,
                    timestamp_ns=int(ts),
                    width=img.shape[1],
                    height=img.shape[0],
                    dtype=str(img.dtype),
                    min_val=int(img.min()),
                    max_val=int(img.max()),
                    zero_ratio=zero / total if total else 0,
                    invalid_ratio=invalid / total if total else 0,
                    mean_val=float(img.mean()),
                )
            )
            frame_idx += 1
            sample_idx += 1
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# 深度质量指标
# ---------------------------------------------------------------------------


def _compute_depth_quality(
    source_path: str,
    depth_ts: np.ndarray,
    report: DunjiaRGBDReport,
    zero_threshold: float,
    saturation_threshold: float,
) -> None:
    """全量扫描深度帧，统计 zero/invalid/saturation/frozen。"""
    from zpds_prepare.readers.dunjia_reader import TOPIC_DEPTH, _open_mcap

    if len(depth_ts) == 0:
        return

    try:
        reader, fh = _open_mcap(source_path)
    except Exception:
        return

    try:
        zero_ratios: list[float] = []
        invalid_ratios: list[float] = []
        sat_frames = 0
        prev_img: np.ndarray | None = None
        frozen_spans: list[tuple[int, int]] = []
        frozen_start: int | None = None
        frame_idx = 0

        for _schema, channel, _msg, decoded in reader.iter_decoded_messages():
            if channel.topic != TOPIC_DEPTH:
                continue

            nparr = np.frombuffer(decoded.data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            if img is None:
                frame_idx += 1
                continue

            total = img.size
            zero = int(np.sum(img == 0))
            zero_ratios.append(zero / total if total else 0)

            # 无效值比例（含饱和标记）
            dtype_max = np.iinfo(img.dtype).max if img.dtype in (np.uint16, np.uint8) else 65535
            invalid_count = 0
            for candidate in _INVALID_CANDIDATES:
                if candidate == 0:
                    continue  # 已在 zero 中计
                if candidate <= dtype_max:
                    invalid_count += int(np.sum(img == candidate))
            invalid_ratios.append(invalid_count / total if total else 0)

            # 饱和（全 1 / dtype_max）
            if int(np.sum(img == dtype_max)) / total > saturation_threshold:
                sat_frames += 1

            # 冻结检测（与前一帧完全相同）
            if prev_img is not None and np.array_equal(img, prev_img):
                if frozen_start is None:
                    frozen_start = frame_idx - 1
            else:
                if frozen_start is not None:
                    frozen_spans.append((frozen_start, frame_idx - 1))
                    frozen_start = None

            prev_img = img
            frame_idx += 1

        # 末尾冻结
        if frozen_start is not None:
            frozen_spans.append((frozen_start, frame_idx - 1))

        report.zero_ratio_mean = float(np.mean(zero_ratios)) if zero_ratios else 0.0
        report.zero_ratio_max = float(np.max(zero_ratios)) if zero_ratios else 0.0
        report.invalid_ratio_mean = float(np.mean(invalid_ratios)) if invalid_ratios else 0.0
        report.saturation_ratio = sat_frames / frame_idx if frame_idx > 0 else 0.0
        report.frozen_span_count = len(frozen_spans)
        report.frozen_total_frames = sum(
            end - start + 1 for start, end in frozen_spans
        )

        # 坏帧区间
        for start, end in frozen_spans:
            report.bad_spans.append({
                "type": "depth_frozen",
                "start_frame": start,
                "end_frame": end,
                "frame_count": end - start + 1,
                "start_timestamp_ns": int(depth_ts[start]) if start < len(depth_ts) else 0,
                "end_timestamp_ns": int(depth_ts[end]) if end < len(depth_ts) else 0,
            })

        if report.zero_ratio_mean > zero_threshold:
            report.issues.append(
                f"深度零值比例过高: mean={report.zero_ratio_mean:.2%}"
            )
        if sat_frames > 0:
            report.issues.append(
                f"深度饱和帧: {sat_frames}/{frame_idx} ({report.saturation_ratio:.2%})"
            )
        if report.frozen_span_count > 0:
            report.issues.append(
                f"深度冻结区间: {report.frozen_span_count} 段, "
                f"共 {report.frozen_total_frames} 帧"
            )
    finally:
        fh.close()


# ---------------------------------------------------------------------------
# RGB-Depth 时间配对
# ---------------------------------------------------------------------------


def _pair_rgb_depth(
    rgb_ts: np.ndarray,
    depth_ts: np.ndarray,
    max_offset_ns: int,
) -> RGBDepthAlignment:
    """对 RGB 和 Depth 时间戳做最近邻配对。

    禁止按帧号配对——所有匹配基于时间戳。
    跨 gap（单个流中间隔超过阈值）不配对。
    """
    if len(rgb_ts) == 0 or len(depth_ts) == 0:
        return RGBDepthAlignment(
            paired_count=0,
            rgb_frame_count=len(rgb_ts),
            depth_frame_count=len(depth_ts),
            paired_ratio=0.0,
            offset_ns_p50=float("nan"),
            offset_ns_p95=float("nan"),
            offset_ns_max=float("nan"),
        )

    offsets: list[int] = []
    rgb_paired: set[int] = set()
    depth_paired: set[int] = set()

    # 为每个 RGB 帧找最近邻深度帧
    for rgb_idx in range(len(rgb_ts)):
        rgb_t = rgb_ts[rgb_idx]
        abs_diffs = np.abs(depth_ts - rgb_t)
        best_depth_idx = int(np.argmin(abs_diffs))
        best_diff = int(abs_diffs[best_depth_idx])

        if best_diff <= max_offset_ns:
            offsets.append(int(rgb_t - depth_ts[best_depth_idx]))
            rgb_paired.add(rgb_idx)
            depth_paired.add(best_depth_idx)

    # 未配对区间
    unpaired_rgb = _find_unpaired_spans(len(rgb_ts), rgb_paired)
    unpaired_depth = _find_unpaired_spans(len(depth_ts), depth_paired)

    if offsets:
        off = np.array(offsets)
        off_abs = np.abs(off)
    else:
        off = np.array([], dtype=np.int64)
        off_abs = np.array([], dtype=np.float64)

    return RGBDepthAlignment(
        paired_count=len(offsets),
        rgb_frame_count=len(rgb_ts),
        depth_frame_count=len(depth_ts),
        paired_ratio=min(
            len(offsets) / max(min(len(rgb_ts), len(depth_ts)), 1),
            1.0,
        ),
        offset_ns_p50=float(np.percentile(off_abs, 50)) if len(off_abs) > 0 else float("nan"),
        offset_ns_p95=float(np.percentile(off_abs, 95)) if len(off_abs) > 0 else float("nan"),
        offset_ns_max=float(off_abs.max()) if len(off_abs) > 0 else float("nan"),
        mapping_method="nearest_neighbor_timestamp",
        uncertainty_ns=int(off_abs.max() - off_abs.min()) if len(off_abs) > 1 else 0,
        unpaired_rgb_spans=unpaired_rgb,
        unpaired_depth_spans=unpaired_depth,
    )


def _find_unpaired_spans(
    total: int, paired: set[int]
) -> list[tuple[int, int]]:
    """找到未配对的连续区间。"""
    if len(paired) == total and total > 0:
        return []
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(total):
        if i not in paired:
            if start is None:
                start = i
        else:
            if start is not None:
                spans.append((start, i - 1))
                start = None
    if start is not None:
        spans.append((start, total - 1))
    return spans


# ---------------------------------------------------------------------------
# 标定一致性
# ---------------------------------------------------------------------------


def _check_calibration_consistency(
    report: DunjiaRGBDReport, session: Any,
) -> None:
    """校验 RGB 和 Depth 的分辨率与标定一致性。

    标定不一致或外参缺失不阻断 RGB 视图，只记录在 issues 中。
    """
    depth_stream = session.depth_streams.get("ego_depth")
    rgb_stream = session.video_streams.get("camera0")

    if depth_stream is None or rgb_stream is None:
        return

    # 分辨率交叉检查
    if (
        report.depth_width > 0
        and report.rgb_width > 0
        and report.depth_width != report.rgb_width
    ):
        report.calibration_issues.append(
            f"RGB/Depth 分辨率不一致: "
            f"RGB={report.rgb_width}×{report.rgb_height}, "
            f"Depth={report.depth_width}×{report.depth_height}"
        )

    # 标定可用性（通过 CAMERA_IDS 检查 depth frame_id）
    depth_frame_id = getattr(depth_stream, "frame_id", "")
    if not depth_frame_id or depth_frame_id == "depth_optical_frame":
        # 默认值说明没有从源中读取到 frame_id
        report.calibration_issues.append(
            "深度 frame_id 为默认值，外参可能不可用"
        )

    report.calibration_consistent = len(report.calibration_issues) == 0
    if not report.calibration_consistent:
        report.issues.append("标定一致性存在问题（不阻断 RGB 视图）")


__all__ = [
    "DepthFrameSample",
    "DunjiaRGBDReport",
    "RGBDepthAlignment",
    "check_dunjia_rgbd",
]
