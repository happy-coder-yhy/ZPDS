"""遁甲音频解码 — raw Opus 包 → WAV。

MCAP 中 foxglove.CompressedAudio 的 data 字段是 raw Opus 包
（无 Ogg 封装，约 20ms/包 @ 48kHz）。ffmpeg 无法直接 demux raw Opus 流，
因此本模块实现 RFC 7845 的最小 Ogg Opus 封装器：

    1. build_ogg_opus(): 把 raw Opus 包列表封装为合法 .ogg 文件字节流
    2. decode_opus_to_wav(): 封装后交给项目已有 ffmpeg 解码为 WAV

零新增 Python 依赖，只依赖系统 ffmpeg（项目已安装 7.1）。
"""

from __future__ import annotations

import os
import struct
import subprocess
from pathlib import Path

# Opus 内部固定 48kHz；每个 Opus 包 = 20ms → 960 采样
OPUS_SAMPLE_RATE = 48000
OPUS_FRAME_SAMPLES = 960  # 20ms @ 48kHz


# ---------------------------------------------------------------------------
# Ogg page 封装（RFC 3533 / RFC 7845）
# ---------------------------------------------------------------------------

def _ogg_crc(data: bytes) -> int:
    """Ogg page CRC-32（RFC 3533）：多项式 0x04C11DB7，非反射，init=0，无 final XOR。

    与 zlib.crc32（反射式）不同，必须单独实现。
    """
    crc = 0
    for byte in data:
        crc ^= byte << 24
        for _ in range(8):
            if crc & 0x80000000:
                crc = ((crc << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
            else:
                crc = (crc << 1) & 0xFFFFFFFF
    return crc


def _ogg_page(
    packets: list[bytes],
    serial: int,
    seq: int,
    granule: int,
    header_type: int,
) -> bytes:
    """构造一个 Ogg page。

    Args:
        packets: 本页要承载的完整包（不跨页切分）。
        serial: Ogg bitstream serial。
        seq: page 序号（从 0 起）。
        granule: 本页最后一个完整包累计的 PCM 采样数。
        header_type: 0x00=普通, 0x02=BOS(首页), 0x04=EOS(末页)。
    """
    # lacing values：每个包切成 ≤255 字节的段
    lacing: list[int] = []
    for pkt in packets:
        if not pkt:
            lacing.append(0)
            continue
        n_seg = (len(pkt) + 254) // 255
        for i in range(n_seg - 1):
            lacing.append(255)
        last = len(pkt) - 255 * (n_seg - 1)
        lacing.append(last % 255 if last % 255 or n_seg == 1 else 0)
    if len(lacing) > 255:
        raise ValueError("单页包数过多（>255 lacing），请减小每页包数")

    header = bytearray()
    header += b"OggS"
    header += b"\x00"  # version 0
    header += bytes([header_type])
    header += struct.pack("<q", granule)  # granule position (signed 64)
    header += struct.pack("<I", serial)
    header += struct.pack("<I", seq)
    # CRC 占位（先补 0），稍后计算
    header += struct.pack("<I", 0)
    header += bytes([len(lacing)])
    header += bytes(lacing)
    # 计算 Ogg CRC-32（覆盖 header + payload）
    body = b"".join(packets)
    crc = _ogg_crc(bytes(header) + body)
    struct.pack_into("<I", header, 22, crc)
    return bytes(header) + body


def build_ogg_opus(
    packets: list[bytes],
    channels: int = 1,
    input_sample_rate: int = OPUS_SAMPLE_RATE,
    preskip: int = 0,
    vendor: str = "ZPDS",
    serial: int = 0x5A5A5A5A,
) -> bytes:
    """把 raw Opus 包列表封装为完整 Ogg Opus 文件字节流。

    Args:
        packets: 按时间顺序排列的 raw Opus 包。
        channels: Opus 声道数（1=mono, 2=stereo）。
        input_sample_rate: 元数据字段（Opus 内部固定 48kHz 解码）。
        preskip: 解码后跳过的前导采样数（未知时传 0，影响 <几 ms）。
        vendor: OpusTags vendor 字符串。
        serial: Ogg bitstream serial（任意固定值）。

    Returns:
        可直接写盘的 .ogg 字节流。
    """
    if not packets:
        raise ValueError("空 Opus 包列表，无法封装")

    # OpusHead（RFC 7845 §5.1）— 19 字节
    head = bytearray()
    head += b"OpusHead"
    head += b"\x01"                     # version
    head += bytes([channels])
    head += struct.pack("<H", preskip)
    head += struct.pack("<I", input_sample_rate)
    head += struct.pack("<h", 0)        # output gain (Q8), 0
    head += b"\x00"                     # mapping family 0 (单一流, 无耦合)

    # OpusTags（RFC 7845 §5.2）
    vendor_b = vendor.encode("utf-8")
    tags = bytearray()
    tags += b"OpusTags"
    tags += struct.pack("<I", len(vendor_b))
    tags += vendor_b
    tags += struct.pack("<I", 0)        # user comment count = 0

    # 分页（与 ffmpeg oggparseopus 期望一致）：
    #   页 0 = OpusHead 单独一页 (BOS)
    #   页 1 = OpusTags 单独一页 (granule 0)
    #   其后 = 音频包页，最后一页 EOS，granule = total_samples + preskip
    pages = bytearray()
    pages += _ogg_page(
        [bytes(head)], serial=serial, seq=0,
        granule=0, header_type=0x02,
    )
    pages += _ogg_page(
        [bytes(tags)], serial=serial, seq=1,
        granule=0, header_type=0x00,
    )

    seq = 2
    total_samples = 0
    chunk_size = 32
    for i in range(0, len(packets), chunk_size):
        chunk = packets[i:i + chunk_size]
        total_samples += OPUS_FRAME_SAMPLES * len(chunk)
        is_last = i + chunk_size >= len(packets)
        pages += _ogg_page(
            chunk, serial=serial, seq=seq,
            granule=total_samples + preskip if is_last else total_samples,
            header_type=0x04 if is_last else 0x00,
        )
        seq += 1

    return bytes(pages)


# ---------------------------------------------------------------------------
# ffmpeg 解码
# ---------------------------------------------------------------------------

def decode_opus_to_wav(
    packets: list[bytes],
    output_wav_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_path: str = "ffmpeg",
    keep_ogg: bool = False,
) -> dict:
    """把 raw Opus 包列表解码为 WAV 文件。

    Args:
        packets: raw Opus 包（按时间顺序）。
        output_wav_path: 输出 .wav 路径。
        sample_rate: 输出采样率（ffmpeg 重采样，如 16000）。
        channels: 输出声道数（1=mono）。
        ffmpeg_path: ffmpeg 可执行文件路径或命令名。
        keep_ogg: 保留中间 .ogg 文件（调试用）。

    Returns:
        {"wav_path": str, "ogg_bytes": int, "packets": n, "duration_s": float}
    """
    ogg_bytes = build_ogg_opus(packets, channels=channels)
    out = Path(output_wav_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    ogg_tmp = out.with_suffix(".ogg.tmp")
    ogg_tmp.write_bytes(ogg_bytes)
    try:
        cmd = [
            ffmpeg_path, "-y", "-hide_banner", "-loglevel", "error",
            "-i", str(ogg_tmp),
            "-ar", str(sample_rate),
            "-ac", str(channels),
            "-c:a", "pcm_s16le",
            str(out),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                              encoding="utf-8", errors="replace")
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg 解码失败 (rc={proc.returncode}): {proc.stderr.strip()}"
            )
    finally:
        if not keep_ogg:
            try:
                ogg_tmp.unlink()
            except OSError:
                pass

    duration_s = len(packets) * OPUS_FRAME_SAMPLES / OPUS_SAMPLE_RATE
    return {
        "wav_path": str(out),
        "ogg_bytes": len(ogg_bytes),
        "packets": len(packets),
        "duration_s": round(duration_s, 3),
        "sample_rate": sample_rate,
        "channels": channels,
    }
