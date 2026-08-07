"""端到端流水线入口：场景分割 + VLM 复核 + QCCascade 统一质量报告。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

if __package__ in {None, ""}:
    # 支持直接 `python scripts/run_pipeline.py` 执行。
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_scene_detection import _load_dotenv, read_video
from zpds.qc.cascade import QCCascade
from zpds.qc.stage10_scene import _check_stage10  # noqa: F401 - 注册 Stage 10
from zpds.scene.config import SceneConfig
from zpds.scene.pipeline import run_scene_pipeline
from zpds.scene.preview import write_scene_previews
from zpds.scene.validator import sha256_file, validate_scene_outputs
from zpds.scene.vlm_review import (
    OpenAICompatibleVLMReviewer,
    load_scene_labels,
)
from zpds.scene.writer import write_scene_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ZPDS 端到端流水线（scene 分割 + VLM 复核 + QC 报告 + 音频检查）"
    )
    parser.add_argument("--source", required=True, help="输入视频路径")
    parser.add_argument(
        "--audio-source",
        help="可选：遁甲 MCAP 路径，从中提取音频并执行 Stage 12 音频 QC",
    )
    parser.add_argument(
        "--profile",
        required=True,
        help="采集源 profile 名（对应 configs/qc_thresholds/<profile>.yaml）",
    )
    parser.add_argument(
        "--output",
        help="输出目录；默认取 scene 配置 output_dir",
    )
    parser.add_argument(
        "--scene-config",
        default="configs/scene/default.yaml",
        help="Scene YAML 配置路径",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="最多读取的帧数，仅用于冒烟测试",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        type=int,
        help="仅运行指定的 QC stage（默认 0~12）",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭进度输出",
    )
    return parser


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_audio(
    mcap_path: str | Path,
    output_dir: Path,
    *,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_path: str = "ffmpeg",
) -> dict | None:
    """从遁甲 MCAP 提取音频并解码为 WAV。

    Returns:
        audio_streams context（供 Stage 12 使用），无音频时返回 None。
    """
    from segment.audio_decoder import decode_opus_to_wav
    from zpds_prepare.readers.dunjia_reader import (
        TOPIC_AUDIO,
        has_audio_topic,
        read_audio,
    )

    mcap = Path(mcap_path)
    if not mcap.is_file():
        raise FileNotFoundError(f"音频 MCAP 不存在: {mcap}")
    if not has_audio_topic(str(mcap)):
        print(f"[audio] MCAP 无音频 topic: {mcap.name}", file=sys.stderr)
        return None

    print(f"[audio] 从 {mcap.name} 提取音频...", file=sys.stderr)
    packets = read_audio(str(mcap))
    if not packets:
        print("[audio] 音频 topic 存在但无消息", file=sys.stderr)
        return None

    # 解码 WAV 到输出目录
    audio_dir = output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    wav_path = audio_dir / "ego_audio.wav"
    decode_opus_to_wav(
        [p.data for p in packets],
        output_wav_path=wav_path,
        sample_rate=sample_rate,
        channels=channels,
        ffmpeg_path=ffmpeg_path,
    )

    # 构造 Stage 12 的 audio_streams context
    timestamps = [p.timestamp_ns for p in packets]
    duration_s = (
        (timestamps[-1] - timestamps[0]) / 1e9 if len(timestamps) > 1 else 0.0
    )
    audio_stream = {
        "stream_id": "ego_audio",
        "wav_uri": str(wav_path),
        "timestamps_ns": timestamps,
        "duration_s": round(duration_s, 3),
        "packets": len(packets),
        "source_topic": TOPIC_AUDIO,
        "source_format": packets[0].format,
    }
    print(
        f"[audio] 完成: {wav_path} packets={len(packets)} "
        f"duration={duration_s:.2f}s sample_rate={sample_rate}",
        file=sys.stderr,
    )
    return audio_stream


def run(args: argparse.Namespace) -> int:
    """执行 scene 分割 + VLM 复核，并生成统一 QC 报告。"""

    quiet = bool(getattr(args, "quiet", False))
    _load_dotenv()
    profile = args.profile
    profile_path = (
        Path("configs/qc_thresholds") / f"{profile}.yaml"
    ).expanduser()
    config = (
        SceneConfig.load_with_profile(args.scene_config, profile_path)
        if profile_path.is_file()
        else SceneConfig.load(args.scene_config)
    )
    video = read_video(args.source, max_frames=args.max_frames)

    reviewer = None
    labels = None
    if config.vlm.enabled:
        if not config.vlm.labels_path.strip():
            raise RuntimeError("scene.vlm.labels_path 未配置，无法运行 VLM 复核")
        labels = load_scene_labels(config.vlm.labels_path)
        reviewer = OpenAICompatibleVLMReviewer(
            config.vlm,
            labels=labels,
            config_hash=config.config_hash,
        )

    raw_sha256_before = sha256_file(video.path)
    started = time.perf_counter()
    run_result = run_scene_pipeline(
        video.frames,
        fps=video.fps,
        config=config,
        vlm_reviewer=reviewer,
        labels=labels,
        start_ns=0,
    )
    if not quiet:
        print(
            f"scene+VLM 完成，总耗时 {time.perf_counter() - started:.1f} 秒",
            file=sys.stderr,
        )

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else config.output_dir
    )
    # ---- 音频提取（可选：遁甲 MCAP）----
    audio_stream = None
    audio_source = getattr(args, "audio_source", None)
    if audio_source:
        audio_stream = _prepare_audio(audio_source, output_dir)
    written = write_scene_run(
        output_dir,
        input_path=video.path,
        config_hash=config.config_hash,
        profile=config.profile,
        fps=video.fps,
        frame_count=len(video.frames),
        start_ns=0,
        end_ns=run_result.end_ns,
        scenes=run_result.scenes,
        vlm_results=run_result.vlm_results,
        review_queue=run_result.review_queue,
        skipped=run_result.skipped,
        skip_reason=run_result.skip_reason,
    )
    previews = []
    if not run_result.skipped and run_result.scenes:
        previews = write_scene_previews(
            written.output_dir,
            video.frames,
            run_result.scenes,
            fps=video.fps,
            start_ns=0,
        )
    validation = validate_scene_outputs(
        written.output_dir,
        raw_path=video.path,
        raw_sha256_before=raw_sha256_before,
        expected_scene_count=(
            len(run_result.scenes) if not run_result.skipped else None
        ),
        expect_artifacts=not run_result.skipped,
    )

    cascade = QCCascade.from_profile(profile)
    if args.stages:
        cascade.config.enabled_stages = list(args.stages)
    cascade_context: dict = {
        "session_id": written.output_dir.name or profile,
        "segment_id": "",
        "scene_pipeline_run": run_result,
        "scene_config": config,
    }
    if audio_stream is not None:
        cascade_context["audio_streams"] = [audio_stream]
    report = cascade.run(
        context=cascade_context,
    )
    disposition_counts: dict[str, int] = {}
    for decision in report.decisions:
        disposition = (
            decision.disposition.value
            if decision.disposition is not None
            else "none"
        )
        disposition_counts[disposition] = (
            disposition_counts.get(disposition, 0) + 1
        )
    report_file = written.output_dir / "cascade_report.json"
    _write_json_atomic(
        report_file,
        {
            "profile": profile,
            "overall_pass": report.overall_pass,
            "decision_count": len(report.decisions),
            "disposition_counts": disposition_counts,
            "decisions": [asdict(decision) for decision in report.decisions],
            "metrics": [asdict(metric) for metric in report.metrics],
        },
    )
    if not quiet:
        print(
            f"QC 报告: {report_file} overall_pass={report.overall_pass} "
            f"decisions={len(report.decisions)}",
            file=sys.stderr,
        )
    document: dict[str, object] = {
        "input": str(video.path),
        "profile": profile,
        "config_hash": config.config_hash,
        "skipped": run_result.skipped,
        "skip_reason": run_result.skip_reason,
        "scene_count": len(run_result.scenes),
        "vlm_reviewed": len(run_result.vlm_results),
        "review_queue_scene_ids": [
            result.scene_id for result in run_result.review_queue
        ],
        "audio": (
            {
                "source": audio_stream["source_topic"],
                "packets": audio_stream["packets"],
                "duration_s": audio_stream["duration_s"],
                "wav_uri": audio_stream["wav_uri"],
            }
            if audio_stream is not None
            else None
        ),
        "output_dir": str(written.output_dir),
        "previews": [str(path) for path in previews],
        "cascade_report": str(report_file),
        "validation_ok": validation.ok,
        "validation_issues": list(validation.issues),
    }
    print(json.dumps(document, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"pipeline failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
