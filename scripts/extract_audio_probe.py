"""Phase 1 验证：遁甲 MCAP 音频提取 → WAV。

用法:
    python scripts/extract_audio_probe.py D:/datasets/egos/遁甲/20260804_103237_00.mcap \
        -o output/audio_probe.wav [--sample-rate 16000] [--ffmpeg ffmpeg]

依赖项目已有 ffmpeg；零新增 Python 包。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segment.audio_decoder import decode_opus_to_wav  # noqa: E402
from zpds_prepare.readers import dunjia_reader  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="遁甲 MCAP 音频提取 → WAV（Phase 1 验证）")
    p.add_argument("input", help="遁甲 .mcap 路径")
    p.add_argument("-o", "--output", default="output/audio_probe.wav", help="输出 WAV 路径")
    p.add_argument("--sample-rate", type=int, default=16000, help="输出采样率 (默认 16000)")
    p.add_argument("--channels", type=int, default=1, help="输出声道数 (默认 1)")
    p.add_argument("--ffmpeg", default="ffmpeg", help="ffmpeg 可执行文件")
    p.add_argument("--keep-ogg", action="store_true", help="保留中间 .ogg（调试）")
    p.add_argument("--max-packets", type=int, default=None, help="只处理前 N 个包（调试）")
    return p


def main() -> int:
    args = build_parser().parse_args()
    mcap = Path(args.input)
    if not mcap.is_file():
        print(f"[错误] MCAP 不存在: {mcap}")
        return 1

    if not dunjia_reader.has_audio_topic(str(mcap)):
        print(f"[信息] MCAP 无音频 topic: {mcap.name}")
        return 2

    print(f"[1/3] 读取音频 topic: {dunjia_reader.TOPIC_AUDIO}")
    t0 = time.time()
    packets = dunjia_reader.read_audio(str(mcap))
    if args.max_packets:
        packets = packets[: args.max_packets]
    if not packets:
        print("[错误] 音频 topic 存在但无消息")
        return 3

    ts = [p.timestamp_ns for p in packets]
    span_s = (max(ts) - min(ts)) / 1e9
    fmt = packets[0].format
    print(f"      包数={len(packets)} 格式={fmt} 跨度={span_s:.2f}s "
          f"间隔={ (ts[1]-ts[0])/1e6:.1f}ms")
    non_opus = {p.format for p in packets} - {"opus"}
    if non_opus:
        print(f"[警告] 存在非 opus 格式: {non_opus}")

    print(f"[2/3] 封装 Ogg Opus + ffmpeg 解码 → {args.output}")
    result = decode_opus_to_wav(
        [p.data for p in packets],
        output_wav_path=args.output,
        sample_rate=args.sample_rate,
        channels=args.channels,
        ffmpeg_path=args.ffmpeg,
        keep_ogg=args.keep_ogg,
    )
    print(f"[3/3] 完成: wav={result['wav_path']} "
          f"packets={result['packets']} duration={result['duration_s']}s "
          f"({result['sample_rate']}Hz/{result['channels']}ch) 耗时={time.time()-t0:.1f}s")

    wav = Path(result["wav_path"])
    if wav.is_file():
        print(f"      WAV 大小: {wav.stat().st_size / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
