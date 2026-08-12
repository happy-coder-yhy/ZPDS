"""Stage 3: 黑屏 / 过曝 / 模糊 / 冻结检测。

包含 D13（过曝检测）和 D14（模糊检测）的完整实现。

D13 过曝检测：
  - 灰度均值
  - P95 / P99 像素强度
  - 高亮饱和像素比例
  - 连续过曝帧数和持续时间

D14 模糊检测：
  - Laplacian 方差作为基础指标
  - 按相机 / 分辨率 / 数据源配置阈值
  - 帧级指标保存
  - 可复核的证据帧 URI
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.qc.cascade import register_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认阈值（当配置缺失时使用；生产环境应从 configs/qc_thresholds/ 加载）
# ---------------------------------------------------------------------------

DEFAULT_OVEREXPOSURE_MEAN_THRESHOLD = 220      # 灰度均值阈值 (0-255)
DEFAULT_OVEREXPOSURE_RATIO_THRESHOLD = 0.95    # 过曝像素比例阈值
DEFAULT_OVEREXPOSURE_SATURATED_RATIO = 0.30    # 饱和像素 (>=250) 比例阈值
DEFAULT_OVEREXPOSURE_P95_THRESHOLD = 245       # P95 强度阈值
DEFAULT_OVEREXPOSURE_P99_THRESHOLD = 250       # P99 强度阈值
DEFAULT_OVEREXPOSURE_CONSECUTIVE_MIN = 3       # 连续过曝帧数最少触发值

DEFAULT_BLUR_LAPLACIAN_THRESHOLD = 100.0       # Laplacian 方差阈值（低于此值视为模糊）
DEFAULT_BLUR_CONSECUTIVE_MIN = 3               # 连续模糊帧数最少触发值
# 模糊段处置阈值（秒）：仅在 blur_split_enabled=True 时生效——
# 达到该时长的 span 才 split（作为切分缺口）；否则一律 keep_with_flag 打标。
# 默认不切分：模糊段切开会破坏动作完整性（动作跨模糊段时两侧都不完整），
# 模仿学习数据清洗的主流是细粒度标记而非切段（见 SCIZOR/HaptalAI）。
DEFAULT_BLUR_QUARANTINE_DURATION_S = 1.0
# 是否允许长模糊段切分（默认关闭，避免破坏完整动作）
DEFAULT_BLUR_SPLIT_ENABLED = False

# 已知分辨率下的自适应模糊阈值（经验值）
BLUR_THRESHOLD_BY_RESOLUTION: dict[tuple[int, int], float] = {
    (1600, 1300): 150.0,
    (1280, 720): 80.0,
    (1920, 1080): 200.0,
    (640, 480): 30.0,
    (352, 288): 15.0,
}


# ---------------------------------------------------------------------------
# 公共工具
# ---------------------------------------------------------------------------


def _find_continuous_spans(
    flags: np.ndarray, min_consecutive: int = 1
) -> list[tuple[int, int]]:
    """在布尔数组中找出连续 True 的区间 [start, end)（半开区间）。"""
    if not len(flags):
        return []
    padded = np.pad(flags.astype(np.int8), (1, 1))
    starts = np.where((padded[1:] == 1) & (padded[:-1] == 0))[0]
    ends = np.where((padded[1:] == 0) & (padded[:-1] == 1))[0]
    # 对齐长度
    spans = []
    for s, e in zip(starts, ends):
        if e - s >= min_consecutive:
            spans.append((int(s), int(e)))
    return spans


def _compute_frame_timestamp_ns(
    frame_idx: int, fps: float, start_ns: int = 0
) -> int:
    """根据帧序号和 fps 推算纳秒时间戳（无精确时间戳时的 fallback）。"""
    return start_ns + int((frame_idx / fps) * 1_000_000_000)


# ---------------------------------------------------------------------------
# D13 过曝检测
# ---------------------------------------------------------------------------


def _overexposure_stats(
    frame: np.ndarray,
    mean_threshold: float,
    overexposure_ratio: float,
    saturated_ratio: float,
    p95_threshold: float,
    p99_threshold: float,
) -> tuple[bool, dict]:
    """单帧过曝判定与指标（VideoCapture / 共享帧源两分支共用）。"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_mean = float(np.mean(gray))
    p95 = float(np.percentile(gray, 95))
    p99 = float(np.percentile(gray, 99))
    saturated_px_ratio = float(np.sum(gray >= 250) / gray.size)
    overexposed_px_ratio = float(np.sum(gray >= mean_threshold) / gray.size)

    is_overexposed = (
        (gray_mean >= mean_threshold and overexposed_px_ratio >= overexposure_ratio)
        or saturated_px_ratio >= saturated_ratio
        or p95 >= p95_threshold
        or p99 >= p99_threshold
    )
    metrics = {
        "frame_idx": 0,  # 由调用方回填
        "gray_mean": round(gray_mean, 2),
        "p95": round(p95, 2),
        "p99": round(p99, 2),
        "saturated_ratio": round(saturated_px_ratio, 4),
        "overexposed_px_ratio": round(overexposed_px_ratio, 4),
    }
    return is_overexposed, metrics


def detect_overexposure(
    video_path: str,
    *,
    mean_threshold: float = DEFAULT_OVEREXPOSURE_MEAN_THRESHOLD,
    overexposure_ratio: float = DEFAULT_OVEREXPOSURE_RATIO_THRESHOLD,
    saturated_ratio: float = DEFAULT_OVEREXPOSURE_SATURATED_RATIO,
    p95_threshold: float = DEFAULT_OVEREXPOSURE_P95_THRESHOLD,
    p99_threshold: float = DEFAULT_OVEREXPOSURE_P99_THRESHOLD,
    consecutive_min: int = DEFAULT_OVEREXPOSURE_CONSECUTIVE_MIN,
    fps: float = 30.0,
    max_frames: int | None = None,
    start_ns: int = 0,
    sample_interval: int = 1,
    evidence_dir: str | None = None,
    frames: Sequence[np.ndarray] | None = None,
) -> list[Decision]:
    """检测视频中的过曝帧。

    Parameters
    ----------
    video_path : str
        视频文件路径。
    mean_threshold : float
        灰度均值超过此值视为过曝 (0-255)。
    overexposure_ratio : float
        超过 mean_threshold 的像素比例超过此值视为过曝。
    saturated_ratio : float
        饱和像素 (>=250) 比例阈值。
    p95_threshold : float
        P95 强度阈值。
    p99_threshold : float
        P99 强度阈值。
    consecutive_min : int
        连续过曝帧最少触发帧数。
    fps : float
        帧率（用于推算时间戳）。
    max_frames : Optional[int]
        最多检测帧数。
    start_ns : int
        起始纳秒时间戳。
    sample_interval : int
        采样间隔（每隔 N 帧检测）。
    evidence_dir : Optional[str]
        证据帧保存目录。
    frames : Optional[Sequence[np.ndarray]]
        共享帧源（BGR，下标 = 帧号）。提供时跳过内部 VideoCapture
        解码，直接从帧源随机访问——注意 sample_interval 跳过的帧
        在帧源模式下不再读取（计算与 IO 双省），flags 仍按帧号填充，
        输出与 VideoCapture 模式等价。

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    frame_metrics: list[dict] = []

    if frames is not None:
        total_frames = len(frames)
        if total_frames <= 0:
            return []
        overexposed_flags = np.zeros(total_frames, dtype=bool)

        for frame_idx in range(total_frames):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % sample_interval != 0:
                continue

            frame = frames[frame_idx]
            is_overexposed, metrics = _overexposure_stats(
                frame, mean_threshold, overexposure_ratio,
                saturated_ratio, p95_threshold, p99_threshold,
            )
            metrics["frame_idx"] = frame_idx
            overexposed_flags[frame_idx] = is_overexposed
            frame_metrics.append(metrics)

            # 保存证据帧
            if is_overexposed and evidence_dir:
                _save_evidence_frame(
                    frame, frame_idx, evidence_dir, prefix="overexposed"
                )

        # 对齐 cap 分支的 frame_idx 语义：已扫描到的帧号上限
        frame_idx = (
            min(total_frames, max_frames) if max_frames is not None else total_frames
        )
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [
                Decision(
                    stage=3,
                    reason=ReasonCode.OVEREXPOSED,
                    severity=Severity.ERROR,
                    message=f"Cannot open video for overexposure check: {video_path}",
                )
            ]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        if video_fps > 0:
            fps = video_fps

        overexposed_flags = np.zeros(total_frames, dtype=bool)

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            is_overexposed, metrics = _overexposure_stats(
                frame, mean_threshold, overexposure_ratio,
                saturated_ratio, p95_threshold, p99_threshold,
            )
            metrics["frame_idx"] = frame_idx
            overexposed_flags[frame_idx] = is_overexposed
            frame_metrics.append(metrics)

            # 保存证据帧
            if is_overexposed and evidence_dir:
                _save_evidence_frame(
                    frame, frame_idx, evidence_dir, prefix="overexposed"
                )

            frame_idx += 1

        cap.release()

    if not frame_metrics:
        return []

    # 聚合连续过曝区间
    spans = _find_continuous_spans(overexposed_flags[:frame_idx], consecutive_min)
    total_overexposed = int(np.sum(overexposed_flags[:frame_idx]))
    overall_ratio = total_overexposed / frame_idx if frame_idx > 0 else 0.0

    for s, e in spans:
        duration_frames = e - s
        duration_s = duration_frames / fps if fps > 0 else 0.0
        t0_ns = _compute_frame_timestamp_ns(s, fps, start_ns)
        t1_ns = _compute_frame_timestamp_ns(e, fps, start_ns)

        severity = Severity.WARN

        decisions.append(
            Decision(
                stage=3,
                reason=ReasonCode.OVEREXPOSED,
                severity=severity,
                message=(
                    f"Overexposed span: frames [{s}, {e}) "
                    f"({duration_frames} frames, {duration_s:.2f}s)"
                ),
                frame_idx=s,
                timestamp_ns=t0_ns,
                detail={
                    "start_frame": s,
                    "end_frame": e,
                    "duration_frames": duration_frames,
                    "duration_s": round(duration_s, 3),
                    "start_ns": t0_ns,
                    "end_ns": t1_ns,
                    "overall_overexposed_ratio": round(overall_ratio, 4),
                    "total_overexposed_frames": total_overexposed,
                    "mean_threshold": mean_threshold,
                    "overexposure_ratio_threshold": overexposure_ratio,
                    "saturated_ratio_threshold": saturated_ratio,
                    "p95_threshold": p95_threshold,
                    "p99_threshold": p99_threshold,
                    # 决策建议：未经金标校准 -> quarantine
                    "recommended_action": (
                        "quarantine"
                    ),
                },
            )
        )

    # 如果没有连续区间但整体比例偏高，给出汇总级 WARN
    if not spans and overall_ratio > 0.05:
        decisions.append(
            Decision(
                stage=3,
                reason=ReasonCode.OVEREXPOSED,
                severity=Severity.INFO,
                message=(
                    f"Isolated overexposed frames: {total_overexposed}/{frame_idx} "
                    f"({overall_ratio:.2%}), no consecutive span"
                ),
                detail={
                    "total_overexposed_frames": total_overexposed,
                    "overall_ratio": round(overall_ratio, 4),
                    "frame_metrics_sample": frame_metrics[:10],
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# D14 模糊检测
# ---------------------------------------------------------------------------


def detect_blur(
    video_path: str,
    *,
    laplacian_threshold: float | None = None,
    resolution_overrides: dict[tuple[int, int], float] | None = None,
    consecutive_min: int = DEFAULT_BLUR_CONSECUTIVE_MIN,
    fps: float = 30.0,
    max_frames: int | None = None,
    start_ns: int = 0,
    sample_interval: int = 1,
    evidence_dir: str | None = None,
    profile: str = "",
    quarantine_duration_s: float = DEFAULT_BLUR_QUARANTINE_DURATION_S,
    blur_split_enabled: bool = DEFAULT_BLUR_SPLIT_ENABLED,
    frames: Sequence[np.ndarray] | None = None,
) -> list[Decision]:
    """检测视频中的模糊帧。

    使用 Laplacian 方差作为基础指标。
    阈值按分辨率自适应，并可通过 profile 配置覆盖。

    处置策略（默认打标不切分）：
    - 默认（blur_split_enabled=False）：所有模糊 span 产生
      KEEP_WITH_FLAG —— 标记保留，不切分。模糊段切开可能破坏
      动作完整性（动作跨模糊段时两侧都不完整）。
    - blur_split_enabled=True 且 span 时长 >= quarantine_duration_s：
      产生 SPLIT（作为切分缺口，两侧各成候选段），需要用户
      显式确认动作边界后才启用。

    Parameters
    ----------
    video_path : str
        视频文件路径。
    laplacian_threshold : Optional[float]
        全局 Laplacian 方差阈值。None 时按分辨率自动选择。
    resolution_overrides : Optional[dict]
        按分辨率覆盖的阈值映射 {(w,h): threshold}。
    consecutive_min : int
        连续模糊帧最少触发帧数。
    fps : float
        帧率。
    max_frames : Optional[int]
        最多检测帧数。
    start_ns : int
        起始纳秒时间戳。
    sample_interval : int
        采样间隔。
    evidence_dir : Optional[str]
        证据帧保存目录。
    profile : str
        Profile 名称（用于日志）。
    quarantine_duration_s : float
        模糊段切分时长阈值（秒），仅 blur_split_enabled=True 时生效。
    blur_split_enabled : bool
        是否允许长模糊段切分（默认 False：一律打标保留）。
    frames : Optional[Sequence[np.ndarray]]
        共享帧源（BGR，下标 = 帧号）。提供时跳过内部 VideoCapture
        解码，直接从帧源随机访问；sample_interval 跳过的帧在帧源
        模式下不再读取，flags 仍按帧号填充，输出与 VideoCapture 模式等价。

    Returns
    -------
    list[Decision]
    """
    decisions: list[Decision] = []
    if frames is not None:
        total_frames = len(frames)
        if total_frames <= 0:
            return []
        video_fps = fps
        video_width = int(frames[0].shape[1])
        video_height = int(frames[0].shape[0])
    else:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return [
                Decision(
                    stage=3,
                    reason=ReasonCode.BLUR_DETECTED,
                    severity=Severity.ERROR,
                    message=f"Cannot open video for blur check: {video_path}",
                )
            ]

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if video_fps > 0:
        fps = video_fps

    # 确定阈值
    merged_overrides = dict(BLUR_THRESHOLD_BY_RESOLUTION)
    if resolution_overrides:
        merged_overrides.update(resolution_overrides)

    if laplacian_threshold is not None:
        threshold = laplacian_threshold
    else:
        threshold = merged_overrides.get((video_width, video_height))
        if threshold is None:
            # fallback: 面积比例缩放
            area = video_width * video_height
            threshold = area / (1280 * 720) * DEFAULT_BLUR_LAPLACIAN_THRESHOLD
            logger.debug(
                "Blur threshold auto-scaled to %.1f for %dx%d",
                threshold,
                video_width,
                video_height,
            )

    blur_flags = np.zeros(total_frames, dtype=bool)
    frame_variances: list[dict] = []

    if frames is not None:
        for frame_idx in range(total_frames):
            if max_frames is not None and frame_idx >= max_frames:
                break
            if frame_idx % sample_interval != 0:
                continue

            frame = frames[frame_idx]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            is_blur = lap_var < threshold

            blur_flags[frame_idx] = is_blur
            frame_variances.append(
                {
                    "frame_idx": frame_idx,
                    "laplacian_variance": round(lap_var, 2),
                    "threshold": round(threshold, 2),
                }
            )

            if is_blur and evidence_dir:
                _save_evidence_frame(frame, frame_idx, evidence_dir, prefix="blur")

        # 对齐 cap 分支的 frame_idx 语义：已扫描到的帧号上限
        frame_idx = (
            min(total_frames, max_frames) if max_frames is not None else total_frames
        )
    else:
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if max_frames is not None and frame_idx >= max_frames:
                break

            if frame_idx % sample_interval != 0:
                frame_idx += 1
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            is_blur = lap_var < threshold

            blur_flags[frame_idx] = is_blur
            frame_variances.append(
                {
                    "frame_idx": frame_idx,
                    "laplacian_variance": round(lap_var, 2),
                    "threshold": round(threshold, 2),
                }
            )

            if is_blur and evidence_dir:
                _save_evidence_frame(frame, frame_idx, evidence_dir, prefix="blur")

            frame_idx += 1

        cap.release()

    if frame_idx == 0:
        return []

    # 聚合连续模糊区间
    spans = _find_continuous_spans(blur_flags[:frame_idx], consecutive_min)
    total_blur = int(np.sum(blur_flags[:frame_idx]))
    overall_ratio = total_blur / frame_idx if frame_idx > 0 else 0.0

    for s, e in spans:
        duration_frames = e - s
        duration_s = duration_frames / fps if fps > 0 else 0.0
        t0_ns = _compute_frame_timestamp_ns(s, fps, start_ns)
        t1_ns = _compute_frame_timestamp_ns(e, fps, start_ns)

        severity = Severity.WARN

        # 处置：默认打标保留（不切分，保护动作完整性）；
        # 仅显式启用 blur_split_enabled 且 span 达到时长阈值时才切分。
        if blur_split_enabled and duration_s >= quarantine_duration_s:
            disposition = Disposition.SPLIT
            recommended = "split"
        else:
            disposition = Disposition.KEEP_WITH_FLAG
            recommended = "keep_with_flag"

        decisions.append(
            Decision(
                stage=3,
                reason=ReasonCode.BLUR_DETECTED,
                severity=severity,
                message=(
                    f"Blur span: frames [{s}, {e}) "
                    f"({duration_frames} frames, {duration_s:.2f}s), "
                    f"threshold={threshold:.1f}"
                ),
                frame_idx=s,
                timestamp_ns=t0_ns,
                end_frame_idx=e,
                end_timestamp_ns=t1_ns,
                disposition=disposition,
                detail={
                    "start_frame": s,
                    "end_frame": e,
                    "duration_frames": duration_frames,
                    "duration_s": round(duration_s, 3),
                    "start_ns": t0_ns,
                    "end_ns": t1_ns,
                    "laplacian_threshold": round(threshold, 2),
                    "video_resolution": f"{video_width}x{video_height}",
                    "overall_blur_ratio": round(overall_ratio, 4),
                    "total_blur_frames": total_blur,
                    "recommended_action": recommended,
                },
            )
        )

    # 汇总 info
    if not spans and overall_ratio > 0.05:
        decisions.append(
            Decision(
                stage=3,
                reason=ReasonCode.BLUR_DETECTED,
                severity=Severity.INFO,
                message=(
                    f"Isolated blur frames: {total_blur}/{frame_idx} "
                    f"({overall_ratio:.2%}), no consecutive span"
                ),
                detail={
                    "total_blur_frames": total_blur,
                    "overall_ratio": round(overall_ratio, 4),
                    "laplacian_threshold": round(threshold, 2),
                    "video_resolution": f"{video_width}x{video_height}",
                    "frame_variances_sample": frame_variances[:10],
                },
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 证据帧保存
# ---------------------------------------------------------------------------


def _save_evidence_frame(
    frame: np.ndarray,
    frame_idx: int,
    evidence_dir: str,
    prefix: str = "evidence",
) -> str | None:
    """保存证据帧到指定目录，返回保存路径。"""
    try:
        out_dir = Path(evidence_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fname = f"{prefix}_frame_{frame_idx:06d}.jpg"
        out_path = out_dir / fname
        cv2.imwrite(str(out_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return str(out_path)
    except Exception:
        logger.warning("Failed to save evidence frame %d", frame_idx, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# Stage 3 统一入口（供 QCCascade 注册调用）
# ---------------------------------------------------------------------------


def check(
    video_path: str,
    *,
    stage_config: dict | None = None,
    evidence_dir: str | None = None,
    fps: float = 30.0,
    start_ns: int = 0,
    profile: str = "",
    frames: Sequence[np.ndarray] | None = None,
) -> list[Decision]:
    """Stage 3 统一检查入口：过曝 + 模糊 + 黑屏 + 冻结。

    当前已实现 D13（过曝）和 D14（模糊）。
    黑屏（D03）和冻结画面在 zpds_prepare/detectors/ 中已有实现，
    可在后续版本中迁移或桥接到本模块。

    frames: 共享帧源（可选）。提供时两个检测器都不再自行解码。
    """
    cfg = stage_config or {}
    decisions: list[Decision] = []

    # --- D13 过曝 ---
    overexp_cfg = cfg.get("overexposure", {})
    if overexp_cfg.get("enabled", True):
        decisions.extend(
            detect_overexposure(
                video_path,
                mean_threshold=overexp_cfg.get(
                    "mean_threshold", DEFAULT_OVEREXPOSURE_MEAN_THRESHOLD
                ),
                overexposure_ratio=overexp_cfg.get(
                    "overexposure_ratio", DEFAULT_OVEREXPOSURE_RATIO_THRESHOLD
                ),
                saturated_ratio=overexp_cfg.get(
                    "saturated_ratio", DEFAULT_OVEREXPOSURE_SATURATED_RATIO
                ),
                p95_threshold=overexp_cfg.get(
                    "p95_threshold", DEFAULT_OVEREXPOSURE_P95_THRESHOLD
                ),
                p99_threshold=overexp_cfg.get(
                    "p99_threshold", DEFAULT_OVEREXPOSURE_P99_THRESHOLD
                ),
                consecutive_min=overexp_cfg.get(
                    "consecutive_min", DEFAULT_OVEREXPOSURE_CONSECUTIVE_MIN
                ),
                fps=fps,
                start_ns=start_ns,
                sample_interval=overexp_cfg.get("sample_interval", 1),
                evidence_dir=evidence_dir,
                frames=frames,
            )
        )

    # --- D14 模糊 ---
    blur_cfg = cfg.get("blur", {})
    if blur_cfg.get("enabled", True):
        decisions.extend(
            detect_blur(
                video_path,
                laplacian_threshold=blur_cfg.get("laplacian_threshold"),
                consecutive_min=blur_cfg.get(
                    "consecutive_min", DEFAULT_BLUR_CONSECUTIVE_MIN
                ),
                fps=fps,
                start_ns=start_ns,
                sample_interval=blur_cfg.get("sample_interval", 1),
                evidence_dir=evidence_dir,
                profile=profile,
                quarantine_duration_s=blur_cfg.get(
                    "quarantine_duration_s", DEFAULT_BLUR_QUARANTINE_DURATION_S
                ),
                blur_split_enabled=blur_cfg.get(
                    "blur_split_enabled", DEFAULT_BLUR_SPLIT_ENABLED
                ),
                frames=frames,
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口（context dict → 委托给具体函数）
# ---------------------------------------------------------------------------


@register_stage(3)
def _check_stage3(context: dict) -> list[Decision]:
    """Stage 3 QCCascade 入口：从 context dict 提取参数并调用 check()。"""
    video_path = context.get("video_path", "")
    # 无视频文件时跳过视觉检测（如 A2D JPEG 序列），避免 FileNotFoundError 噪音
    if not video_path or not Path(video_path).exists():
        return []
    stage_config = context.get("stage_config", {})
    evidence_dir = context.get("evidence_dir")
    fps = context.get("fps", 30.0)
    start_ns = context.get("start_ns", 0)
    profile = context.get("profile", "")
    return check(
        video_path=video_path,
        stage_config=stage_config,
        evidence_dir=evidence_dir,
        fps=fps,
        start_ns=start_ns,
        profile=profile,
        frames=context.get("frames"),
    )
