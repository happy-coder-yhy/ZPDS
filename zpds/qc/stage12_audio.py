"""Stage 12: 音频质量检测。

对 Prepared Segment / Session 的音频流做质量检查：
  - AUDIO_UNREADABLE: WAV 无法读取或为空
  - AUDIO_SILENCE:    静音比例过高（整段几乎无声）
  - AUDIO_GAP:        音频包时间戳缺口（丢包/采集中断）
  - AUDIO_DURATION_MISMATCH: 音频时长与期望（包数×20ms）偏差过大

检测输入支持两种模式：
  - Prepared Segment: context["segment_dir"] 指向 segment 目录
  - Session 流: context["audio_streams"] 为 list[dict]（含 packets/uri 等）
"""

from __future__ import annotations

import logging
import wave
from pathlib import Path

import numpy as np
import pandas as pd

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.qc.cascade import register_stage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 默认阈值
# ---------------------------------------------------------------------------

DEFAULT_SILENCE_RATIO_THRESHOLD = 0.90    # 静音帧占比超过此值 → 静音告警
DEFAULT_SILENCE_ABS_THRESHOLD = 100       # 采样绝对值低于此值视为静音
DEFAULT_GAP_FACTOR = 2.5                  # 包间隔超过期望 × N 倍 → gap
DEFAULT_GAP_MIN_S = 0.2                   # 缺口最小秒数（避免噪声触发）
DEFAULT_DURATION_TOLERANCE_S = 0.5        # 时长偏差容差（秒）
EXPECTED_PACKET_INTERVAL_S = 0.02         # Opus 20ms/包


# ---------------------------------------------------------------------------
# 检测函数
# ---------------------------------------------------------------------------

def detect_audio_unreadable(
    wav_path: str | Path,
    *,
    stream_id: str = "ego_audio",
) -> list[Decision]:
    """检测音频文件是否可读。"""
    path = Path(wav_path)
    if not path.exists():
        return [Decision(
            stage=12,
            reason=ReasonCode.AUDIO_UNREADABLE,
            severity=Severity.ERROR,
            message=f"[{stream_id}] 音频文件缺失: {path}",
            disposition=Disposition.QUARANTINE,
            detail={"stream_id": stream_id, "uri": str(path)},
        )]
    try:
        with wave.open(str(path), "rb") as w:
            n = w.getnframes()
            rate = w.getframerate()
            if n == 0 or rate <= 0:
                raise ValueError("空 WAV 或采样率无效")
        return []
    except (wave.Error, EOFError, OSError, ValueError) as exc:
        return [Decision(
            stage=12,
            reason=ReasonCode.AUDIO_UNREADABLE,
            severity=Severity.ERROR,
            message=f"[{stream_id}] 音频文件不可读: {exc}",
            disposition=Disposition.QUARANTINE,
            detail={"stream_id": stream_id, "uri": str(path), "error": str(exc)},
        )]


def detect_audio_silence(
    samples: np.ndarray,
    *,
    abs_threshold: int = DEFAULT_SILENCE_ABS_THRESHOLD,
    silence_ratio_threshold: float = DEFAULT_SILENCE_RATIO_THRESHOLD,
    stream_id: str = "ego_audio",
) -> list[Decision]:
    """检测整段音频是否几乎全静音。

    Args:
        samples: PCM16 采样数组（int16）。
        abs_threshold: 采样绝对值低于此值视为静音。
        silence_ratio_threshold: 静音占比阈值（0.9 = 90% 静音才告警）。
    """
    if samples.size == 0:
        return []
    arr = np.abs(samples.astype(np.int32))
    silence_ratio = float(np.mean(arr < abs_threshold))
    if silence_ratio >= silence_ratio_threshold:
        return [Decision(
            stage=12,
            reason=ReasonCode.AUDIO_SILENCE,
            severity=Severity.WARN,
            message=(
                f"[{stream_id}] 音频静音比例 {silence_ratio:.1%} "
                f"超过阈值 {silence_ratio_threshold:.1%}"
            ),
            disposition=Disposition.KEEP_WITH_FLAG,
            detail={
                "stream_id": stream_id,
                "silence_ratio": round(silence_ratio, 4),
                "abs_threshold": abs_threshold,
            },
        )]
    return []


def detect_audio_gap(
    timestamps_ns: list[int],
    *,
    expected_interval_s: float = EXPECTED_PACKET_INTERVAL_S,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    gap_min_s: float = DEFAULT_GAP_MIN_S,
    stream_id: str = "ego_audio",
) -> list[Decision]:
    """检测音频包时间戳缺口（丢包/采集中断）。

    Args:
        timestamps_ns: 音频包时间戳（纳秒，已排序）。
        expected_interval_s: 期望包间隔（Opus 20ms = 0.02）。
        gap_factor: 间隔超过期望 × N 倍 → gap。
        gap_min_s: 缺口最小秒数。
    """
    decisions: list[Decision] = []
    ts = np.array(timestamps_ns, dtype=np.int64)
    if len(ts) < 2:
        return decisions

    diffs = np.diff(ts)
    expected_ns = int(expected_interval_s * 1e9)
    threshold_ns = int(expected_ns * gap_factor)

    for i, gap_ns in enumerate(diffs):
        if gap_ns > threshold_ns:
            gap_s = gap_ns / 1e9
            if gap_s >= gap_min_s:
                decisions.append(Decision(
                    stage=12,
                    reason=ReasonCode.AUDIO_GAP,
                    severity=Severity.WARN,
                    message=(
                        f"[{stream_id}] 音频缺口 {gap_s:.2f}s "
                        f"(包 {i}→{i + 1}, 期望 {expected_interval_s}s)"
                    ),
                    disposition=Disposition.KEEP_WITH_FLAG,
                    timestamp_ns=int(ts[i]),
                    end_timestamp_ns=int(ts[i + 1]),
                    detail={
                        "stream_id": stream_id,
                        "gap_s": round(gap_s, 3),
                        "packet_a": int(i),
                        "packet_b": int(i + 1),
                        "expected_interval_s": expected_interval_s,
                    },
                ))
    return decisions


def detect_audio_duration_mismatch(
    duration_s: float,
    packets: int,
    *,
    expected_packet_s: float = EXPECTED_PACKET_INTERVAL_S,
    tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    stream_id: str = "ego_audio",
) -> list[Decision]:
    """检测音频时长与包数×包长是否一致。"""
    expected_s = packets * expected_packet_s
    if abs(duration_s - expected_s) > tolerance_s:
        return [Decision(
            stage=12,
            reason=ReasonCode.AUDIO_DURATION_MISMATCH,
            severity=Severity.WARN,
            message=(
                f"[{stream_id}] 音频时长 {duration_s:.2f}s "
                f"≠ 期望 {expected_s:.2f}s (packets={packets})"
            ),
            disposition=Disposition.KEEP_WITH_FLAG,
            detail={
                "stream_id": stream_id,
                "duration_s": round(duration_s, 3),
                "expected_s": round(expected_s, 3),
                "packets": packets,
            },
        )]
    return []


# ---------------------------------------------------------------------------
# 聚合入口
# ---------------------------------------------------------------------------

def check(
    *,
    wav_path: str | Path | None = None,
    timestamps_ns: list[int] | None = None,
    samples: np.ndarray | None = None,
    duration_s: float | None = None,
    packets: int | None = None,
    silence_abs_threshold: int = DEFAULT_SILENCE_ABS_THRESHOLD,
    silence_ratio_threshold: float = DEFAULT_SILENCE_RATIO_THRESHOLD,
    gap_factor: float = DEFAULT_GAP_FACTOR,
    gap_min_s: float = DEFAULT_GAP_MIN_S,
    duration_tolerance_s: float = DEFAULT_DURATION_TOLERANCE_S,
    stream_id: str = "ego_audio",
) -> list[Decision]:
    """音频质量检测聚合入口。

    参数均可选——能提供什么就检什么。
    """
    decisions: list[Decision] = []

    # 1. 可读性
    if wav_path is not None:
        decisions.extend(detect_audio_unreadable(wav_path, stream_id=stream_id))
        # 若文件可读且未提供 samples，从 WAV 读
        if samples is None:
            path = Path(wav_path)
            if path.exists():
                try:
                    with wave.open(str(path), "rb") as w:
                        frames = w.readframes(w.getnframes())
                        samples = np.frombuffer(frames, dtype=np.int16)
                        if duration_s is None:
                            duration_s = w.getnframes() / w.getframerate()
                except (wave.Error, EOFError, OSError):
                    samples = None

    # 2. 静音
    if samples is not None:
        decisions.extend(detect_audio_silence(
            samples,
            abs_threshold=silence_abs_threshold,
            silence_ratio_threshold=silence_ratio_threshold,
            stream_id=stream_id,
        ))

    # 3. 缺口
    if timestamps_ns is not None:
        decisions.extend(detect_audio_gap(
            timestamps_ns,
            gap_factor=gap_factor,
            gap_min_s=gap_min_s,
            stream_id=stream_id,
        ))

    # 4. 时长一致性
    if duration_s is not None and packets is not None:
        decisions.extend(detect_audio_duration_mismatch(
            duration_s,
            packets,
            tolerance_s=duration_tolerance_s,
            stream_id=stream_id,
        ))

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------

@register_stage(12)
def _check_stage12(context: dict) -> list[Decision]:
    """Stage 12 QCCascade 入口。

    支持三种输入模式：
    - ``context["segment_dir"]``: 直接校验 Prepared Segment 的音频流
    - ``context["audio_streams"]``: list[dict]，含 stream_id / packets / wav_uri
    - ``context["audio_wav_path"]`` + ``audio_timestamps_ns``: 单流 flat keys
    """
    decisions: list[Decision] = []

    stage_config = context.get("stage_config", {}) or {}
    silence_ratio = stage_config.get("silence_ratio_threshold", DEFAULT_SILENCE_RATIO_THRESHOLD)
    gap_factor = stage_config.get("gap_factor", DEFAULT_GAP_FACTOR)
    duration_tol = stage_config.get("duration_tolerance_s", DEFAULT_DURATION_TOLERANCE_S)

    # 模式 1: Prepared Segment 目录
    seg_dir = context.get("segment_dir")
    if seg_dir:
        seg_dir = Path(seg_dir)
        seg_json = seg_dir / "segment.json"
        if seg_json.exists():
            import json
            with open(seg_json, encoding="utf-8") as f:
                segment = json.load(f)
            for stream in segment.get("streams", []):
                if stream.get("modality") != "audio":
                    continue
                sid = stream["stream_id"]
                wav_path = seg_dir / stream.get("uri", "")
                ts = None
                sm_uri = stream.get("origin", {}).get("sample_map_uri")
                if sm_uri:
                    sm_path = seg_dir / sm_uri
                    if sm_path.exists():
                        sm = pd.read_parquet(str(sm_path))
                        ts = sm["timestamp_ns"].tolist()
                decisions.extend(check(
                    wav_path=wav_path,
                    timestamps_ns=ts,
                    duration_s=stream.get("duration_s"),
                    packets=stream.get("packets"),
                    silence_ratio_threshold=silence_ratio,
                    gap_factor=gap_factor,
                    duration_tolerance_s=duration_tol,
                    stream_id=sid,
                ))
        return decisions

    # 模式 2: audio_streams 列表
    audio_streams = context.get("audio_streams")
    if audio_streams:
        for stream in audio_streams:
            sid = stream.get("stream_id", "ego_audio")
            decisions.extend(check(
                wav_path=stream.get("wav_uri"),
                timestamps_ns=stream.get("timestamps_ns"),
                duration_s=stream.get("duration_s"),
                packets=stream.get("packets"),
                silence_ratio_threshold=silence_ratio,
                gap_factor=gap_factor,
                duration_tolerance_s=duration_tol,
                stream_id=sid,
            ))
        return decisions

    # 模式 3: flat keys
    return check(
        wav_path=context.get("audio_wav_path"),
        timestamps_ns=context.get("audio_timestamps_ns"),
        duration_s=context.get("audio_duration_s"),
        packets=context.get("audio_packets"),
        silence_ratio_threshold=silence_ratio,
        gap_factor=gap_factor,
        duration_tolerance_s=duration_tol,
        stream_id=context.get("stream_id", "ego_audio"),
    )
