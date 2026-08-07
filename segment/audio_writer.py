"""遁甲音频流标准化写出 — Prepared Segment 的 ego_audio 流。

把 MCAP 中的 foxglove.CompressedAudio（Opus 包）标准化为：
    data/ego_audio.wav                    PCM16 WAV（统一采样率/声道）
    maps/ego_audio_sample_map.parquet     包 ↔ 时间戳映射（对齐其他流）

依赖 segment.audio_decoder（Ogg Opus 封装 + ffmpeg 解码），零新增依赖。
"""

from __future__ import annotations

import json
import wave
from pathlib import Path
from typing import Any

import pandas as pd

from segment.audio_decoder import OPUS_FRAME_SAMPLES, OPUS_SAMPLE_RATE, decode_opus_to_wav


def build_audio_sample_map(
    timestamps_ns: list[int],
    log_times_ns: list[int] | None = None,
    packet_sizes: list[int] | None = None,
) -> pd.DataFrame:
    """构建音频包 sample map。

    Args:
        timestamps_ns: 每个音频包的消息内 timestamp（有序）。
        log_times_ns: 每个音频包的 MCAP log_time（可选，保留原始记录时刻）。
        packet_sizes: 每个音频包的压缩后字节数（可选）。

    Returns:
        DataFrame，列：
        - packet_index        音频包序号
        - timestamp_ns        消息内时间戳
        - log_time_ns         MCAP log_time（无则与 timestamp_ns 相同）
        - packet_size        压缩包大小（字节）
        - duration_ns        本包时长（20ms @ 48kHz = 960 samples）
        - sample_offset      累计 PCM 采样偏移（用于 WAV 内定位）
    """
    frame_samples = OPUS_FRAME_SAMPLES  # 960 = 20ms @ 48kHz
    rows = []
    sample_offset = 0
    for i, ts in enumerate(timestamps_ns):
        rows.append({
            "packet_index": i,
            "timestamp_ns": int(ts),
            "log_time_ns": int(log_times_ns[i]) if log_times_ns else int(ts),
            "packet_size": int(packet_sizes[i]) if packet_sizes else 0,
            "duration_ns": int(frame_samples * 1_000_000_000 / OPUS_SAMPLE_RATE),
            "sample_offset": sample_offset,
        })
        sample_offset += frame_samples
    return pd.DataFrame(rows)


def write_audio_stream(
    packets: list[dict[str, Any]],
    output_dir: str | Path,
    source_start_ns: int,
    source_end_ns: int,
    sample_rate: int = 16000,
    channels: int = 1,
    stream_id: str = "ego_audio",
    ffmpeg_path: str = "ffmpeg",
    keep_ogg: bool = False,
) -> dict:
    """把音频包标准化为 Prepared Segment 的音频流。

    Args:
        packets: AudioStream.packets（[{timestamp_ns, data, format, log_time_ns}]）。
        output_dir: segment 输出根目录（data/、maps/ 子目录）。
        source_start_ns: span 起始时间戳（音频裁剪对齐）。
        source_end_ns: span 结束时间戳。
        sample_rate: 输出采样率（默认 16000）。
        channels: 输出声道数（默认 1）。
        stream_id: 流标识（默认 ego_audio）。
        ffmpeg_path: ffmpeg 可执行文件。
        keep_ogg: 保留中间 .ogg（调试）。

    Returns:
        result dict：
        - stream_id, uri, format, sample_rate, channels, packets, duration_s
        - sample_map_uri, source_asset_id, operation
    """
    out_dir = Path(output_dir)
    data_dir = out_dir / "data"
    maps_dir = out_dir / "maps"
    data_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    # 裁剪到 span 内的包
    span_pkts = [
        p for p in packets
        if source_start_ns <= p["timestamp_ns"] <= source_end_ns
    ]
    if not span_pkts:
        raise ValueError(f"span 内没有音频包: {source_start_ns}~{source_end_ns}")

    # 按时间戳排序
    span_pkts.sort(key=lambda p: p["timestamp_ns"])

    wav_path = data_dir / f"{stream_id}.wav"
    decode_opus_to_wav(
        [p["data"] for p in span_pkts],
        output_wav_path=wav_path,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_path=ffmpeg_path,
        keep_ogg=keep_ogg,
    )

    # sample map
    sm = build_audio_sample_map(
        timestamps_ns=[p["timestamp_ns"] for p in span_pkts],
        log_times_ns=[p.get("log_time_ns", p["timestamp_ns"]) for p in span_pkts],
        packet_sizes=[len(p["data"]) for p in span_pkts],
    )
    sample_map_uri = f"maps/{stream_id}_sample_map.parquet"
    sm.to_parquet(str(out_dir / sample_map_uri), index=False)

    duration_s = len(span_pkts) * OPUS_FRAME_SAMPLES / OPUS_SAMPLE_RATE

    return {
        "stream_id": stream_id,
        "uri": f"data/{stream_id}.wav",
        "sample_map_uri": sample_map_uri,
        "format": "wav",
        "source_format": span_pkts[0].get("format", "opus"),
        "sample_rate": sample_rate,
        "channels": channels,
        "packets": len(span_pkts),
        "duration_s": round(duration_s, 3),
        "source_asset_id": "raw_mcap_audio",
        "operation": "decode_opus_to_wav_and_sample_map",
        "source_topic": "/robot0/sensor/audio",
    }
