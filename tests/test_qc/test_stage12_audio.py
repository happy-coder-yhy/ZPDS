"""Stage 12 音频质量检测 — 单元测试。

覆盖:
  - detect_audio_unreadable: 文件缺失 / 损坏
  - detect_audio_silence: 全静音告警 / 正常音频无告警
  - detect_audio_gap: 正常间隔无告警 / 缺口告警
  - detect_audio_duration_mismatch: 时长一致 / 不一致
  - _check_stage12: QCCascade 注册 + segment_dir 模式
"""

from __future__ import annotations

import json
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from zpds.core.decisions import ReasonCode, Severity  # noqa: E402
from zpds.qc.cascade import QCCascade, CascadeConfig, get_stage_checker  # noqa: E402
from zpds.qc.stage12_audio import (  # noqa: E402
    detect_audio_duration_mismatch,
    detect_audio_gap,
    detect_audio_silence,
    detect_audio_unreadable,
)


def _write_wav(path: Path, samples: np.ndarray, rate: int = 16000) -> None:
    """写一个 PCM16 WAV 文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(samples.astype(np.int16).tobytes())


class TestDetectAudioUnreadable:
    def test_missing_file(self):
        d = detect_audio_unreadable("nonexistent.wav")
        assert len(d) == 1
        assert d[0].reason == ReasonCode.AUDIO_UNREADABLE
        assert d[0].severity == Severity.ERROR

    def test_readable_file(self, tmp_path):
        wav = tmp_path / "a.wav"
        _write_wav(wav, np.zeros(16000, dtype=np.int16))
        d = detect_audio_unreadable(wav)
        assert d == []

    def test_corrupt_file(self, tmp_path):
        wav = tmp_path / "bad.wav"
        wav.write_bytes(b"not a wav file at all")
        d = detect_audio_unreadable(wav)
        assert len(d) == 1
        assert d[0].reason == ReasonCode.AUDIO_UNREADABLE


class TestDetectAudioSilence:
    def test_all_zero(self):
        d = detect_audio_silence(np.zeros(16000, dtype=np.int16))
        assert len(d) == 1
        assert d[0].reason == ReasonCode.AUDIO_SILENCE
        assert d[0].severity == Severity.WARN

    def test_normal_audio_no_alert(self):
        rng = np.random.default_rng(42)
        samples = rng.integers(-20000, 20000, size=16000, dtype=np.int16)
        d = detect_audio_silence(samples)
        assert d == []

    def test_threshold_respect(self):
        """99% 静音但阈值 95% → 告警。"""
        samples = np.zeros(16000, dtype=np.int16)
        samples[100:200] = 30000  # 0.6% 非静音
        d = detect_audio_silence(samples, silence_ratio_threshold=0.95)
        assert len(d) == 1


class TestDetectAudioGap:
    def test_normal_interval(self):
        ts = [1_000_000_000 + i * 20_000_000 for i in range(100)]
        d = detect_audio_gap(ts)
        assert d == []

    def test_gap_detected(self):
        ts = [1_000_000_000 + i * 20_000_000 for i in range(50)]
        ts += [1_000_000_000 + 50 * 20_000_000 + 1_000_000_000 + i * 20_000_000
               for i in range(50)]
        d = detect_audio_gap(ts)
        assert len(d) == 1
        assert d[0].reason == ReasonCode.AUDIO_GAP
        assert d[0].timestamp_ns is not None
        assert d[0].end_timestamp_ns is not None

    def test_short_gap_ignored(self):
        """小缺口（< gap_min_s）不告警。"""
        ts = [1_000_000_000 + i * 20_000_000 for i in range(50)]
        ts += [1_000_000_000 + 50 * 20_000_000 + 50_000_000 + i * 20_000_000
               for i in range(50)]  # 50ms gap
        d = detect_audio_gap(ts, gap_min_s=0.2)
        assert d == []


class TestDetectAudioDurationMismatch:
    def test_consistent(self):
        d = detect_audio_duration_mismatch(1.0, 50)  # 50包×20ms=1.0s
        assert d == []

    def test_mismatch(self):
        d = detect_audio_duration_mismatch(2.0, 50)  # 声明 2s 实际应 1s
        assert len(d) == 1
        assert d[0].reason == ReasonCode.AUDIO_DURATION_MISMATCH


class TestStage12Registration:
    def test_registered(self):
        assert get_stage_checker(12) is not None

    def test_cascade_runs_segment_dir(self, tmp_path):
        """用含音频流的 segment.json 跑完整级联。"""
        # 构造 segment 目录（含静音音频 → 应触发 AUDIO_SILENCE）
        seg_dir = tmp_path / "seg_000001"
        wav = seg_dir / "data" / "ego_audio.wav"
        _write_wav(wav, np.zeros(16000, dtype=np.int16))  # 全静音
        segment = {
            "streams": [{
                "stream_id": "ego_audio",
                "modality": "audio",
                "uri": "data/ego_audio.wav",
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "packets": 50,
                "duration_s": 1.0,
                "origin": {"sample_map_uri": "maps/ego_audio_sample_map.parquet"},
            }]
        }
        (seg_dir / "segment.json").write_text(json.dumps(segment), encoding="utf-8")

        cfg = CascadeConfig(enabled_stages=[12])
        cascade = QCCascade(config=cfg)
        report = cascade.run({
            "session_id": "test",
            "segment_dir": str(seg_dir),
        })
        assert len(report.decisions) >= 1
        reasons = {d.reason for d in report.decisions}
        assert ReasonCode.AUDIO_SILENCE in reasons

    def test_cascade_pass_normal_audio(self, tmp_path):
        """正常音频 → 无决策。"""
        seg_dir = tmp_path / "seg_pass"
        wav = seg_dir / "data" / "ego_audio.wav"
        rng = np.random.default_rng(7)
        samples = rng.integers(-15000, 15000, size=16000, dtype=np.int16)
        _write_wav(wav, samples)
        segment = {
            "streams": [{
                "stream_id": "ego_audio",
                "modality": "audio",
                "uri": "data/ego_audio.wav",
                "format": "wav",
                "sample_rate": 16000,
                "channels": 1,
                "packets": 50,
                "duration_s": 1.0,
                "origin": {"sample_map_uri": "maps/ego_audio_sample_map.parquet"},
            }]
        }
        (seg_dir / "segment.json").write_text(json.dumps(segment), encoding="utf-8")

        cfg = CascadeConfig(enabled_stages=[12])
        cascade = QCCascade(config=cfg)
        report = cascade.run({
            "session_id": "test",
            "segment_dir": str(seg_dir),
        })
        assert report.decisions == []
        assert report.overall_pass is True

    def test_cascade_audio_streams_gap_detected(self):
        """清洗阶段 ctx（audio_streams 无 WAV）→ 缺口/时长决策。"""
        from zpds.qc.stage12_audio import _check_stage12

        # 50 个正常包 + 2s 缺口 + 50 个包
        ts = [1_000_000_000 + i * 20_000_000 for i in range(50)]
        ts += [1_000_000_000 + 50 * 20_000_000 + 2_000_000_000 + i * 20_000_000
               for i in range(50)]
        ctx = {
            "audio_streams": [{
                "stream_id": "ego_audio",
                "timestamps_ns": ts,
                "duration_s": 4.0,
                "packets": 100,
            }],
            "stage_config": {},
        }
        decisions = _check_stage12(ctx)
        reasons = {d.reason for d in decisions}
        assert ReasonCode.AUDIO_GAP in reasons
        assert ReasonCode.AUDIO_DURATION_MISMATCH in reasons
        # 无 WAV → 不产出 unreadable
        assert ReasonCode.AUDIO_UNREADABLE not in reasons
