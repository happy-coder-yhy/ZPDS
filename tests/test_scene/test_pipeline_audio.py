"""run_pipeline 音频接入 — 单元测试。

覆盖:
  - build_parser 支持 --audio-source 参数
  - _prepare_audio: 无音频 mcap 返回 None / 缺失文件报错
  - _prepare_audio: 有音频 mcap 提取 WAV + 返回 context（真实遁甲数据）
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from scripts.run_pipeline import _prepare_audio, build_parser  # noqa: E402


class TestBuildParser:
    def test_audio_source_arg(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--source", "v.mp4", "--profile", "dunjia_ego",
             "--audio-source", "audio.mcap"]
        )
        assert args.audio_source == "audio.mcap"

    def test_audio_source_optional(self):
        parser = build_parser()
        args = parser.parse_args(
            ["--source", "v.mp4", "--profile", "dunjia_ego"]
        )
        assert args.audio_source is None


class TestPrepareAudio:
    def test_missing_mcap_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            _prepare_audio("nonexistent.mcap", tmp_path)

    def test_no_audio_topic_returns_none(self, tmp_path):
        """6月18日遁甲 mcap（无音频）→ None。"""
        mcap = Path(r"D:/datasets/egos/遁甲/20260618_084650_00.mcap")
        if not mcap.exists():
            pytest.skip("旧 mcap 不存在")
        result = _prepare_audio(mcap, tmp_path)
        assert result is None

    def test_real_audio_extraction(self, tmp_path):
        """8月4日遁甲 mcap（含音频）→ WAV + context。"""
        mcap = Path(r"D:/datasets/egos/遁甲/20260804_103237_00.mcap")
        if not mcap.exists():
            pytest.skip("真实 mcap 不存在")
        result = _prepare_audio(mcap, tmp_path)
        assert result is not None
        assert result["stream_id"] == "ego_audio"
        assert result["packets"] > 1000
        assert result["duration_s"] > 50
        assert result["source_topic"] == "/robot0/sensor/audio"

        # WAV 已写出
        wav = Path(result["wav_uri"])
        assert wav.exists()
        assert wav.stat().st_size > 100_000  # > 100KB 真实音频

        # timestamps 有序
        ts = result["timestamps_ns"]
        assert ts == sorted(ts)

    def test_real_audio_custom_output_dir(self, tmp_path):
        """输出目录应包含 audio/ego_audio.wav。"""
        mcap = Path(r"D:/datasets/egos/遁甲/20260804_103237_00.mcap")
        if not mcap.exists():
            pytest.skip("真实 mcap 不存在")
        result = _prepare_audio(mcap, tmp_path)
        assert result is not None
        expected = tmp_path / "audio" / "ego_audio.wav"
        assert Path(result["wav_uri"]).resolve() == expected.resolve()
        assert expected.exists()
