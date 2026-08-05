"""场景自动分割 Stage 1/2/all 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import ClassVar, Protocol

import cv2
import numpy as np

from zpds.scene.backend_router import SceneBackendRouter
from zpds.scene.backends import (
    BrightnessTransitionDetector,
    DinoV2SmallEmbedder,
    HistogramTransitionDetector,
    OpticalFlowTransitionDetector,
    SSIMTransitionDetector,
)
from zpds.scene.config import SceneConfig
from zpds.scene.fusion import SceneBoundaryFusion, StageATransitionFusion
from zpds.scene.schemas import BoundaryScore, SceneProposal, TransitionProposal


class StageBBackend(Protocol):
    def embed(self, frames_rgb: Sequence[np.ndarray]) -> np.ndarray: ...

    def detect(
        self,
        frames_bgr: Sequence[np.ndarray],
        *,
        fps: float,
        start_timestamp_ns: int = 0,
        candidate_frame_indices: Sequence[int] | None = None,
    ) -> list[BoundaryScore]: ...


ProgressCallback = Callable[[str, int, int], None]


class ConsoleProgress:
    """向 stderr 输出可刷新的进度，不污染 stdout JSON。"""

    _LABELS: ClassVar[dict[str, str]] = {
        "histogram": "直方图",
        "ssim": "SSIM",
        "optical_flow": "光流",
        "brightness": "亮度",
        "fusion": "Stage A 融合",
        "dino": "DINOv2-Small",
        "scene_fusion": "场景定稿",
    }

    def __call__(self, phase: str, completed: int, total: int) -> None:
        safe_total = max(1, total)
        percent = min(100.0, 100.0 * completed / safe_total)
        label = self._LABELS.get(phase, phase)
        ending = "\n" if completed >= total else ""
        print(
            f"\r[{label}] {completed}/{total} ({percent:5.1f}%)",
            file=sys.stderr,
            end=ending,
            flush=True,
        )


@dataclass(frozen=True)
class VideoFrames:
    path: Path
    frames: tuple[np.ndarray, ...]
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class SceneDetectionRun:
    stage: str
    skipped: bool
    skip_reason: str | None
    frame_count: int
    fps: float
    start_ns: int
    end_ns: int
    config_hash: str
    profile: str | None
    transitions: tuple[TransitionProposal, ...] = ()
    semantic_boundaries: tuple[BoundaryScore, ...] = ()
    scenes: tuple[SceneProposal, ...] = ()

    def to_document(self, *, input_path: str | None = None) -> dict[str, object]:
        document: dict[str, object] = asdict(self)
        if input_path is not None:
            document["input"] = input_path
        return document


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="运行场景转场检测与 DINOv2-Small 语义边界检测",
    )
    parser.add_argument("--input", required=True, help="输入视频路径")
    parser.add_argument(
        "--config",
        default="configs/scene/default.yaml",
        help="Scene YAML 配置路径",
    )
    parser.add_argument(
        "--profile",
        help="可选 QC Profile YAML，用其 scene 小节覆盖默认配置",
    )
    parser.add_argument(
        "--stage",
        choices=["1", "2", "all"],
        default="all",
        help="1=低成本转场，2=DINO 语义变化，all=最终 scene",
    )
    parser.add_argument(
        "--start-ns",
        type=int,
        default=0,
        help="输入视频在 Segment 内的起始纳秒时间戳",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        help="最多读取的帧数，仅用于冒烟测试",
    )
    parser.add_argument(
        "--candidate-frame",
        type=int,
        action="append",
        help="Stage 2 候选帧，可重复指定；all 默认使用 Stage 1 候选边界",
    )
    parser.add_argument(
        "--with-vlm",
        action="store_true",
        help="请求人员 B 的 VLM 复核流水线；仅可与 --stage all 一起使用",
    )
    parser.add_argument(
        "--output-json",
        help="可选调试 JSON；正式 parquet 由 scene writer 负责",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="关闭进度与耗时输出",
    )
    return parser


def read_video(path: str | Path, *, max_frames: int | None = None) -> VideoFrames:
    video_path = Path(path).expanduser().resolve()
    if not video_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames 必须大于 0")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"无法打开输入视频: {video_path}")
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not np.isfinite(fps) or fps <= 0:
            raise ValueError(f"视频 FPS 非法: {fps}")
        frames: list[np.ndarray] = []
        while max_frames is None or len(frames) < max_frames:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise RuntimeError(f"视频包含无法解码的空帧: {len(frames)}")
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"输入视频没有可解码帧: {video_path}")
    return VideoFrames(video_path, tuple(frames), fps, width, height)


def _transition_detectors(
    config: SceneConfig,
    progress: ProgressCallback | None = None,
):
    router = SceneBackendRouter.from_config(config)
    factories = {
        "histogram": lambda: HistogramTransitionDetector(config.stage_a.histogram),
        "ssim": lambda: SSIMTransitionDetector(config.stage_a.ssim),
        "optical_flow": lambda: OpticalFlowTransitionDetector(
            config.stage_a.optical_flow,
            progress_callback=(
                (lambda completed, total: progress("optical_flow", completed, total))
                if progress is not None
                else None
            ),
        ),
        "brightness": lambda: BrightnessTransitionDetector(config.stage_a.brightness),
    }
    return tuple(factories[name]() for name in router.policy.stage_a_backends)


def _run_stage_a(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    start_ns: int,
    config: SceneConfig,
    progress: ProgressCallback | None = None,
) -> list[TransitionProposal]:
    detectors = _transition_detectors(config, progress)
    frame_scores = {}
    proposals: list[TransitionProposal] = []
    total_pairs = max(0, len(frames) - 1)
    for detector in detectors:
        if progress is not None:
            progress(detector.source, 0, total_pairs)
        scores = detector.score_frames(frames, fps=fps)
        frame_scores[detector.source] = scores
        if progress is not None:
            progress(detector.source, total_pairs, total_pairs)
        proposals.extend(
            detector.detect(
                frames,
                fps=fps,
                start_timestamp_ns=start_ns,
                frame_scores=scores,
            )
        )
    if progress is not None:
        progress("fusion", 0, 1)
    fused = StageATransitionFusion(config.stage_a).fuse(
        proposals,
        frame_scores,
        fps=fps,
        start_timestamp_ns=start_ns,
    )
    if progress is not None:
        progress("fusion", 1, 1)
    return fused


def _center_embedding_provider(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    start_ns: int,
    embedder: StageBBackend,
):
    cache: dict[int, np.ndarray] = {}

    def provide(timestamp: int) -> np.ndarray:
        relative_ns = timestamp - start_ns
        index = round(relative_ns * fps / 1_000_000_000)
        index = max(0, min(len(frames) - 1, index))
        if index not in cache:
            frame_rgb = cv2.cvtColor(frames[index], cv2.COLOR_BGR2RGB)
            cache[index] = embedder.embed([frame_rgb])[0]
        return cache[index]

    return provide


def run_scene_detection(
    frames: Sequence[np.ndarray],
    *,
    fps: float,
    config: SceneConfig,
    stage: str,
    start_ns: int = 0,
    stage_b_backend: StageBBackend | None = None,
    candidate_frame_indices: Sequence[int] | None = None,
    progress: ProgressCallback | None = None,
) -> SceneDetectionRun:
    if stage not in {"1", "2", "all"}:
        raise ValueError("stage 必须是 1、2 或 all")
    if isinstance(start_ns, bool) or start_ns < 0:
        raise ValueError("start_ns 必须是非负整数")
    if not frames:
        raise ValueError("frames 不能为空")
    end_ns = start_ns + round(len(frames) * 1_000_000_000 / fps)
    if not config.enabled:
        return SceneDetectionRun(
            stage=stage,
            skipped=True,
            skip_reason="scene.enabled=false",
            frame_count=len(frames),
            fps=fps,
            start_ns=start_ns,
            end_ns=end_ns,
            config_hash=config.config_hash,
            profile=config.profile,
        )

    transitions: list[TransitionProposal] = []
    semantic_boundaries: list[BoundaryScore] = []
    scenes: list[SceneProposal] = []
    if stage in {"1", "all"}:
        transitions = _run_stage_a(
            frames,
            fps=fps,
            start_ns=start_ns,
            config=config,
            progress=progress,
        )
    if stage in {"2", "all"}:
        embedder = stage_b_backend or DinoV2SmallEmbedder(config.stage_b)
        effective_candidates = candidate_frame_indices
        if stage == "all" and effective_candidates is None and transitions:
            effective_candidates = tuple(
                transition.frame_index for transition in transitions
            )
        if progress is not None:
            progress("dino", 0, len(frames))
        semantic_boundaries = embedder.detect(
            frames,
            fps=fps,
            start_timestamp_ns=start_ns,
            candidate_frame_indices=effective_candidates,
        )
        if progress is not None:
            progress("dino", len(frames), len(frames))
        if stage == "all":
            if progress is not None:
                progress("scene_fusion", 0, 1)
            scenes = SceneBoundaryFusion(
                config.fusion,
                config_hash=config.config_hash,
                center_embedding_provider=_center_embedding_provider(
                    frames,
                    fps=fps,
                    start_ns=start_ns,
                    embedder=embedder,
                ),
            ).fuse(
                transitions,
                semantic_boundaries,
                start_ns=start_ns,
                end_ns=end_ns,
                fps=fps,
            )
            if progress is not None:
                progress("scene_fusion", 1, 1)
    return SceneDetectionRun(
        stage=stage,
        skipped=False,
        skip_reason=None,
        frame_count=len(frames),
        fps=fps,
        start_ns=start_ns,
        end_ns=end_ns,
        config_hash=config.config_hash,
        profile=config.profile,
        transitions=tuple(transitions),
        semantic_boundaries=tuple(semantic_boundaries),
        scenes=tuple(scenes),
    )


def _write_json_atomic(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(
        json.dumps(document, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def run(args: argparse.Namespace) -> int:
    quiet = bool(getattr(args, "quiet", False))
    progress = None if quiet else ConsoleProgress()
    if bool(getattr(args, "with_vlm", False)):
        if args.stage != "all":
            raise ValueError("--with-vlm 仅可与 --stage all 一起使用")
        raise RuntimeError(
            "--with-vlm 已保留为人员 B 流水线交接入口；"
            "当前 zpds.scene.pipeline/VLMReviewer 尚未实现，拒绝伪造复核结果"
        )
    profile_path = getattr(args, "profile", None)
    config = (
        SceneConfig.load_with_profile(args.config, profile_path)
        if profile_path
        else SceneConfig.load(args.config)
    )
    if not quiet:
        print(f"读取视频: {Path(args.input).expanduser().resolve()}", file=sys.stderr)
    video = read_video(args.input, max_frames=args.max_frames)
    if not quiet:
        print(
            f"已读取 {len(video.frames)} 帧，{video.width}x{video.height} @ {video.fps:g} FPS",
            file=sys.stderr,
        )
    started = time.perf_counter()
    result = run_scene_detection(
        video.frames,
        fps=video.fps,
        config=config,
        stage=args.stage,
        start_ns=args.start_ns,
        candidate_frame_indices=getattr(args, "candidate_frame", None),
        progress=progress,
    )
    document = result.to_document(input_path=str(video.path))
    if args.output_json:
        _write_json_atomic(Path(args.output_json).expanduser().resolve(), document)
    else:
        print(json.dumps(document, indent=2, ensure_ascii=False))
    if not quiet:
        print(f"完成，总耗时 {time.perf_counter() - started:.1f} 秒", file=sys.stderr)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run(args)
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"scene detection failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
