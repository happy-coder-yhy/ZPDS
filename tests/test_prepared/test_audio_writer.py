"""音频流标准化写出 — 单元测试。

覆盖:
  - AudioStream 数据模型（session_model）
  - dunjia_reader.read_audio() / has_audio_topic()
  - audio_writer.write_audio_stream()（WAV + sample_map + 元数据）
  - audio_decoder.build_ogg_opus()（Ogg 封装合法性）
"""

from __future__ import annotations

import struct
import subprocess
import sys
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from segment.audio_decoder import (  # noqa: E402
    OPUS_FRAME_SAMPLES,
    OPUS_SAMPLE_RATE,
    _ogg_crc,
    build_ogg_opus,
    decode_opus_to_wav,
)
from segment.audio_writer import (  # noqa: E402
    build_audio_sample_map,
    write_audio_stream,
)
from zpds_prepare.readers.dunjia_reader import (  # noqa: E402
    AudioPacket,
    TOPIC_AUDIO,
    has_audio_topic,
    read_audio,
)
from zpds_prepare.readers.session_model import AudioStream  # noqa: E402


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _fake_opus_packet(size: int = 40, toc: int = 0x48) -> bytes:
    """构造一个伪 Opus 包：TOC 字节 + 随机负载。

    TOC=0x48 → config=9 (20ms), stereo=0 (mono), 1帧。
    """
    rng = np.random.default_rng(42)
    payload = rng.integers(0, 256, size=size - 1, dtype=np.uint8)
    return bytes([toc]) + payload.tobytes()


@pytest.fixture
def fake_packets() -> list[bytes]:
    """10 个伪 Opus 包（20ms each = 0.2s 总时长）。"""
    return [_fake_opus_packet() for _ in range(10)]


@pytest.fixture
def audio_packet_objs() -> list[AudioPacket]:
    """5 个 AudioPacket 对象，时间戳 20ms 递增。"""
    base = 1_700_000_000_000_000_000  # 任意基准
    return [
        AudioPacket(
            timestamp_ns=base + i * 20_000_000,
            data=_fake_opus_packet(),
            format="opus",
            log_time_ns=base + i * 20_000_000,
        )
        for i in range(5)
    ]


# ---------------------------------------------------------------------------
# 1. AudioStream 数据模型
# ---------------------------------------------------------------------------

class TestAudioStream:
    def test_basic(self):
        stream = AudioStream(
            stream_id="ego_audio",
            packets=[{"timestamp_ns": 0}, {"timestamp_ns": 20_000_000}],
            sample_rate_hz=48000,
            channels=1,
            format="opus",
        )
        assert stream.num_packets == 2
        assert stream.duration_ns == 20_000_000
        assert stream.sample_rate_hz == 48000

    def test_empty(self):
        stream = AudioStream(stream_id="ego_audio")
        assert stream.num_packets == 0
        assert stream.duration_ns == 0


# ---------------------------------------------------------------------------
# 2. audio_decoder: Ogg Opus 封装
# ---------------------------------------------------------------------------

class TestBuildOggOpus:
    def test_ogg_magic_and_crc(self, fake_packets):
        ogg = build_ogg_opus(fake_packets, channels=1)
        assert ogg[:4] == b"OggS"
        # 逐页验证 CRC
        pos = 0
        page = 0
        while pos < len(ogg):
            assert ogg[pos:pos + 4] == b"OggS"
            seg = ogg[pos + 26]
            lacing = list(ogg[pos + 27:pos + 27 + seg])
            body_len = sum(lacing)
            hdr = bytearray(ogg[pos:pos + 27 + seg])
            stored = struct.unpack("<I", hdr[22:26])[0]
            hdr[22:26] = b"\x00\x00\x00\x00"
            body = ogg[pos + 27 + seg:pos + 27 + seg + body_len]
            assert _ogg_crc(bytes(hdr) + body) == stored, f"page {page} CRC"
            pos += 27 + seg + body_len
            page += 1

    def test_head_tail_pages(self, fake_packets):
        """OpusHead 单独一页、OpusTags 单独一页、EOS 尾页。"""
        ogg = build_ogg_opus(fake_packets, channels=1)
        # 第一页：seg=1（OpusHead 19B），header=27+1=28
        assert ogg[26] == 1
        assert ogg[28:36] == b"OpusHead"
        # 第二页：seg=1（OpusTags），计算 offset
        p2 = 28 + 19
        assert ogg[p2:p2 + 4] == b"OggS"
        assert ogg[p2 + 27 + 1:p2 + 27 + 1 + 8] == b"OpusTags"

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            build_ogg_opus([], channels=1)

    def test_granule_final(self, fake_packets):
        """EOS 页 granule = 总采样数 + preskip。"""
        ogg = build_ogg_opus(fake_packets, channels=1, preskip=312)
        # 找最后一页
        pos = 0
        last_granule = None
        while pos < len(ogg):
            seg = ogg[pos + 26]
            granule = struct.unpack("<q", ogg[pos + 6:pos + 14])[0]
            last_granule = granule
            pos += 27 + seg + sum(ogg[pos + 27:pos + 27 + seg])
        expected = OPUS_FRAME_SAMPLES * len(fake_packets) + 312
        assert last_granule == expected


class TestDecodeOpusToWav:
    def test_decodes_wav(self, fake_packets, tmp_path):
        out = tmp_path / "out.wav"
        result = decode_opus_to_wav(fake_packets, out, sample_rate=16000)
        assert out.exists()
        w = wave.open(str(out), "rb")
        assert w.getframerate() == 16000
        assert w.getnchannels() == 1
        assert w.getnframes() == 3200  # 10包 × 20ms × 16kHz
        assert abs(result["duration_s"] - 0.2) < 0.01
        w.close()


# ---------------------------------------------------------------------------
# 3. audio_writer
# ---------------------------------------------------------------------------

class TestBuildAudioSampleMap:
    def test_columns_and_offsets(self):
        ts = [1000, 1020, 1040]
        df = build_audio_sample_map(ts)
        assert list(df.columns) == [
            "packet_index", "timestamp_ns", "log_time_ns",
            "packet_size", "duration_ns", "sample_offset",
        ]
        assert len(df) == 3
        assert df["sample_offset"].tolist() == [0, 960, 1920]
        assert df["duration_ns"].tolist() == [20_000_000] * 3

    def test_log_time_override(self):
        ts = [1000, 1020]
        log = [999, 1019]
        df = build_audio_sample_map(ts, log_times_ns=log)
        assert df["log_time_ns"].tolist() == [999, 1019]

    def test_packet_sizes(self):
        ts = [1000, 1020]
        df = build_audio_sample_map(ts, packet_sizes=[28, 46])
        assert df["packet_size"].tolist() == [28, 46]


class TestWriteAudioStream:
    def test_write_full(self, audio_packet_objs, tmp_path):
        packets = [{
            "timestamp_ns": p.timestamp_ns,
            "data": p.data,
            "format": p.format,
            "log_time_ns": p.log_time_ns,
        } for p in audio_packet_objs]
        start = audio_packet_objs[0].timestamp_ns
        end = audio_packet_objs[-1].timestamp_ns + 1

        result = write_audio_stream(
            packets=packets,
            output_dir=tmp_path,
            source_start_ns=start,
            source_end_ns=end,
            sample_rate=16000,
            channels=1,
        )
        assert result["stream_id"] == "ego_audio"
        assert result["uri"] == "data/ego_audio.wav"
        assert result["packets"] == 5
        assert result["sample_rate"] == 16000
        assert result["channels"] == 1

        # WAV 存在
        wav_path = tmp_path / "data" / "ego_audio.wav"
        assert wav_path.exists()
        # sample_map 存在
        sm_path = tmp_path / "maps" / "ego_audio_sample_map.parquet"
        assert sm_path.exists()
        df = pd.read_parquet(sm_path)
        assert len(df) == 5

    def test_span_filter(self, audio_packet_objs, tmp_path):
        """只写出 span 内的包。"""
        packets = [{
            "timestamp_ns": p.timestamp_ns,
            "data": p.data,
            "format": p.format,
            "log_time_ns": p.log_time_ns,
        } for p in audio_packet_objs]
        # 只取中间 3 个包
        start = audio_packet_objs[1].timestamp_ns
        end = audio_packet_objs[3].timestamp_ns + 1
        result = write_audio_stream(
            packets=packets,
            output_dir=tmp_path,
            source_start_ns=start,
            source_end_ns=end,
        )
        assert result["packets"] == 3
        df = pd.read_parquet(tmp_path / "maps" / "ego_audio_sample_map.parquet")
        assert len(df) == 3

    def test_empty_span_raises(self, audio_packet_objs, tmp_path):
        packets = [{
            "timestamp_ns": p.timestamp_ns,
            "data": p.data,
            "format": p.format,
            "log_time_ns": p.log_time_ns,
        } for p in audio_packet_objs]
        # span 完全在包时间戳之前（包从 1_700_000_000_000_000_000 开始）
        with pytest.raises(ValueError):
            write_audio_stream(
                packets=packets,
                output_dir=tmp_path,
                source_start_ns=1_600_000_000_000_000_000,
                source_end_ns=1_650_000_000_000_000_000,
            )


# ---------------------------------------------------------------------------
# 4. dunjia_reader: read_audio / has_audio_topic
# ---------------------------------------------------------------------------

class TestDunjiaReaderAudio:
    def test_topic_constant(self):
        assert TOPIC_AUDIO == "/robot0/sensor/audio"

    def test_has_audio_topic_missing_file(self):
        assert has_audio_topic("nonexistent.mcap") is False

    def test_read_audio_real_mcap(self):
        """真实遁甲 mcap（8月4日，含音频）——集成验证。"""
        mcap = Path(r"D:/datasets/egos/遁甲/20260804_103237_00.mcap")
        if not mcap.exists():
            pytest.skip("真实 mcap 不存在")
        assert has_audio_topic(str(mcap)) is True
        packets = read_audio(str(mcap))
        assert len(packets) > 1000
        assert packets[0].format == "opus"
        # 时间戳有序
        ts = [p.timestamp_ns for p in packets]
        assert ts == sorted(ts)
        # 间隔约 20ms
        gaps = np.diff(np.array(ts, dtype=np.int64))
        assert np.median(gaps) / 1e6 == pytest.approx(20.0, abs=2.0)

    def test_read_audio_no_audio_mcap(self):
        """6月18日遁甲 mcap（无音频）→ 空列表。"""
        mcap = Path(r"D:/datasets/egos/遁甲/20260618_084650_00.mcap")
        if not mcap.exists():
            pytest.skip("旧 mcap 不存在")
        assert has_audio_topic(str(mcap)) is False
        assert read_audio(str(mcap)) == []
