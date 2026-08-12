"""
ZPDS Prepare — 主入口。

从原始采集数据出发，运行检测器、生成统一 QualityIssue、
决定 trim/split/keep_with_flag、产出候选 Segment。

用法:
    # 墨现 (默认)
    python -m zpds_prepare.main /path/to/dataset/
    python -m zpds_prepare.main /path/to/dataset/ --profile guida

    # 遁甲
    python -m zpds_prepare.main /path/to/session.mcap --profile dunjia
"""

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np
import yaml

if TYPE_CHECKING:
    # 仅类型解析用（scene schemas 为纯 dataclass，无重依赖）。
    # 运行时名字来自各函数体内的局部 import，保持延迟加载。
    from zpds.scene.schemas import SceneProposal, VLMReviewResult

from zpds.hands.schemas import PreparedFrame
from zpds.qc import QCCascade
from zpds.qc.cascade import CascadeDistribution
from zpds_prepare.decisions.issue_model import QualityIssue
from zpds_prepare.decisions.segment_planner import (
    downgrade_split_issues,
    get_issue_summary,
    plan_segments,
)
from zpds_prepare.detectors.bad_frame import detect_bad_frames
from zpds_prepare.detectors.black_frame import detect_black_frames
from zpds_prepare.detectors.depth_coverage import detect_depth_coverage
from zpds_prepare.detectors.frame_count import detect_frame_count_mismatch
from zpds_prepare.detectors.imu_gap import detect_imu_gaps
from zpds_prepare.detectors.timestamp_gap import detect_timestamp_gaps
from zpds_prepare.writers.candidate_writer import write_segment_candidates
from zpds_prepare.writers.quality_writer import (
    derive_processing_status,
    derive_quality_status,
    write_quality_issues,
)


def _decisions_to_issues(decisions: list, stream_id: str) -> list:
    """将级联 Decision 转换为 QualityIssue，供 segment_planner 消费。

    只有 disposition 为 trim / split / keep_with_flag / quarantine 的决策才会被转换；
    keep / reject 或 disposition 为空的决策不影响分段，被跳过。
    """
    issues = []
    for d in decisions:
        disp = d.disposition.value if d.disposition else None
        if disp not in ("trim", "split", "keep_with_flag", "quarantine"):
            continue
        start_ns = d.timestamp_ns or d.detail.get("start_ns", 0)
        end_ns = d.end_timestamp_ns or d.detail.get("end_ns", start_ns)
        if end_ns <= start_ns:
            end_ns = start_ns + 1  # 确保有效区间
        issues.append(QualityIssue(
            issue_type=d.reason.value,
            stream_id=stream_id,
            start_ns=int(start_ns),
            end_ns=int(end_ns),
            severity=d.severity.value,
            decision=disp,
            details={
                "stage": d.stage,
                "message": d.message,
                "source": "qc_cascade",
                **{k: v for k, v in (d.detail or {}).items()
                   if k not in ("start_ns", "end_ns")},
            },
        ))
    return issues

CONFIG_PATH = "config.yaml"
OUTPUT_DIR = "output"


def _get_reader(profile: str):
    """返回 reader 模块。所有 reader 导出统一的 read_session() 入口。"""
    if profile == "guida":
        from zpds_prepare.readers import guida_reader as rd
    elif profile == "dunjia":
        from zpds_prepare.readers import dunjia_reader as rd
    elif profile == "umi":
        from zpds_prepare.readers import umi_reader as rd
    else:
        raise ValueError(f"未知 profile: {profile}，可选: guida, dunjia, umi")
    return rd


def load_config(config_path: str = CONFIG_PATH) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_header(n: int, title: str):
    print(f"\n{'=' * 60}")
    print(f"  Step {n}: {title}")
    print(f"{'=' * 60}")


def _load_dotenv(path: str | Path = ".env") -> None:
    """把 .env 的 KEY=VALUE 注入进程环境；已存在的变量不覆盖。"""

    env_path = Path(path).expanduser()
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


class _RawVideoFrameSource:
    """把原始视频包装成 PreparedFrameSource，供 HandsPipeline 消费。

    frame_source（SharedFrameSource，可选）：提供时不再自行打开
    VideoCapture，而是从共享帧源顺序读取——避免重复解码。
    """

    def __init__(
        self,
        video_path: str | Path,
        *,
        timestamps_ns: list[int] | tuple[int, ...] | None,
        session_id: str,
        stream_id: str,
        frame_source: Any | None = None,
    ) -> None:
        self._video_path = Path(video_path)
        self._timestamps = list(timestamps_ns or ())
        self._session_id = session_id
        self._stream_id = stream_id
        self._frame_source = frame_source

    @property
    def segment_id(self) -> str:
        return self._session_id

    @property
    def video_stream_id(self) -> str:
        return self._stream_id

    def __iter__(self) -> Iterator[PreparedFrame]:
        capture = None
        if self._frame_source is not None:
            frame_iter = enumerate(self._frame_source)
        else:
            capture = cv2.VideoCapture(str(self._video_path))
            if not capture.isOpened():
                raise FileNotFoundError(f"无法打开视频: {self._video_path}")

            def _read_frames():
                while True:
                    ok, frame_bgr = capture.read()
                    if not ok:
                        return
                    yield frame_bgr

            frame_iter = enumerate(_read_frames())
        try:
            for index, frame_bgr in frame_iter:
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                timestamp_ns = (
                    int(self._timestamps[index])
                    if index < len(self._timestamps)
                    else 0
                )
                yield PreparedFrame(
                    frame_rgb=frame_rgb,
                    output_frame_index=index,
                    timestamp_ns=timestamp_ns,
                    source_frame_index=index,
                    source_timestamp_ns=timestamp_ns,
                )
        finally:
            if capture is not None:
                capture.release()


def _hands_cache_ok(
    report_file: Path,
    hands_parquet_file: Path,
    video_path: str | Path,
    timestamps_ns: list[int] | tuple[int, ...],
) -> tuple[bool, list[str]]:
    """Hands 缓存复用判定：帧数 + 源视频内容 + 模型指纹全一致才复用。

    指纹来源（复用判定本身不触发模型推理）：
    - hand_cleaning_report.json: source.advertised_frame_count / provenance.source_video_sha256
    - hands_2d.parquet 列: config_sha256 / checkpoint_sha256（每行同值，取首行）
    期望指纹与 _run_hand_analysis 同源（HandsPipelineConfig.load，只读配置与权重
    文件元信息，不加载模型权重）。

    Returns:
        (是否复用, 不匹配原因列表)
    """
    import pandas as pd

    from zpds.hands.config import HandsPipelineConfig
    from zpds.utils.hash import sha256_hex

    reasons: list[str] = []
    try:
        rep = json.loads(report_file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return False, [f"报告不可读: {exc}"]
    src = rep.get("source", {})
    if int(src.get("advertised_frame_count", -1)) != len(timestamps_ns):
        reasons.append(
            f"帧数 {src.get('advertised_frame_count')} vs 当前 {len(timestamps_ns)}"
        )
    prov = rep.get("provenance", {})
    try:
        if sha256_hex(str(video_path)) != prov.get("source_video_sha256", ""):
            reasons.append("源视频内容指纹不匹配")
    except OSError as exc:
        reasons.append(f"源视频指纹计算失败: {exc}")
    try:
        meta = pd.read_parquet(
            hands_parquet_file, columns=["config_sha256", "checkpoint_sha256"]
        )
        if meta.empty:
            reasons.append("hands_2d.parquet 为空")
        else:
            runtime_config = HandsPipelineConfig.load(
                str(Path(CONFIG_PATH).expanduser().resolve()),
            )
            row = meta.iloc[0]
            if str(row["config_sha256"]) != str(runtime_config.config_sha256):
                reasons.append("config_sha256 不匹配")
            if str(row["checkpoint_sha256"]) != str(
                runtime_config.checkpoint_sha256
            ):
                reasons.append("checkpoint_sha256 不匹配")
    except Exception as exc:  # noqa: BLE001 - 指纹读取失败一律不复用
        reasons.append(f"hands_2d.parquet 读取失败: {exc}")
    return (not reasons), reasons


def _run_hand_analysis(
    *,
    video_path: str | Path,
    timestamps_ns: list[int] | tuple[int, ...] | None,
    output_dir: Path,
    session_id: str,
    stream_id: str,
    source_kind: str = "ego",
    frame_source: Any | None = None,
) -> dict[str, str]:
    """运行手部检测（WiLoR，按配置路由）与手部清洗，返回报告与 parquet 路径。"""
    from zpds.hands.backend_router import HandsBackendRouter
    from zpds.hands.cleaning import HandVideoCleaningConfig, clean_hand_video
    from zpds.hands.config import HandsPipelineConfig
    from zpds.hands.estimator_factory import (
        create_hand_estimator,
        validate_estimator_runtime,
    )
    from zpds.hands.pipeline import HandsPipeline
    from zpds.hands.wilor_preflight import check_wilor_assets
    from zpds.hands.writer import write_hand_observations

    runtime_config = HandsPipelineConfig.load(
        str(Path(CONFIG_PATH).expanduser().resolve()),
    )
    router = HandsBackendRouter(runtime_config.backend_policy)
    primary_model = router.select_backend(is_ego=source_kind == "ego")
    if primary_model == "wilor":
        preflight = check_wilor_assets(runtime_config.wilor)
        if not preflight.ready:
            raise RuntimeError(
                "WiLoR 资产预检失败（请在有 WiLoR 权重的工作机运行）: "
                + ("; ".join(preflight.errors) or "未知资产错误")
            )
    runtime = create_hand_estimator(primary_model, runtime_config)
    validate_estimator_runtime(primary_model, runtime, runtime_config)
    source = _RawVideoFrameSource(
        video_path,
        timestamps_ns=timestamps_ns,
        session_id=session_id,
        stream_id=stream_id,
        frame_source=frame_source,
    )
    pipeline = HandsPipeline(
        source,
        runtime.estimator,
        model_name=runtime.model_name,
        model_version=runtime.model_version,
        active_backend=runtime.active_backend,
    )
    # 手部产物放入 source-level staging 目录（analysis/hands/），
    # batch_prepare 按候选区间裁切后落入各 segment。
    hand_dir = _analysis_output_dir(output_dir, "hands")
    hand_dir.mkdir(parents=True, exist_ok=True)
    hands_parquet = hand_dir / "hands_2d.parquet"
    write_hand_observations(
        pipeline,
        hands_parquet,
        prep_revision="r0001",
        checkpoint_sha256=runtime_config.checkpoint_sha256,
        config_sha256=runtime_config.config_sha256,
        run_meta={"source": "zpds_prepare.main", "profile": "hand_integration"},
    )
    cleaning_config = HandVideoCleaningConfig.load(
        "configs/hands/cleaning_default.yaml"
    )
    result = clean_hand_video(
        str(video_path),
        str(hands_parquet),
        str(hand_dir),
        cleaning_config,
        frame_source=frame_source,
    )
    return {
        "hands_parquet": str(hands_parquet),
        "hand_cleaning_report_path": str(result.report_path),
        "video_stream_id": stream_id,
        "model": runtime.active_backend,
    }


def _run_privacy_analysis(
    *,
    video_path: str | Path,
    profile: str,
    output_dir: Path,
    session_id: str,
    frame_source: Any | None = None,
) -> dict:
    """运行隐私脱敏分析（人脸模糊 + PII 文本遮挡），返回 Stage 0 消费的 manifest dict。

    只做检测与统计（稀疏检测 + KLT 传播），不写脱敏视频——脱敏产物仍由
    scripts/run_privacy_redaction.py 单独产出；这里仅提供 Stage 0 QC 输入。

    frame_source（SharedFrameSource，可选）：提供时从帧源顺序读取帧，
    不再自行解码视频。
    """
    from zpds.privacy.backend_router import PrivacyBackendPolicy
    from zpds.privacy.config import PrivacyConfig
    from zpds.privacy.pipeline import PrivacyPipeline

    profile_name = {
        "guida": "guida_ego",
        "dunjia": "dunjia_ego",
        "umi": "jianzhi_umi",
        "epic": "epic100",
        "a2d": "a2d_robot",
    }.get(profile, profile)

    config_path = Path("configs/privacy/default.yaml").expanduser().resolve()
    try:
        pcfg = PrivacyConfig.load(config_path)
    except FileNotFoundError:
        print(f"  [warn] 隐私配置不存在 ({config_path})，使用默认值")
        pcfg = PrivacyConfig.defaults()

    policy = PrivacyBackendPolicy.from_profile(profile_name)
    pipeline = PrivacyPipeline(
        Path(video_path),
        config=pcfg,
        policy=policy,
        profile=profile_name,
        session_id=session_id,
        face_interval=pcfg.face_interval_frames,
        text_interval=pcfg.text_interval_frames,
        frames=frame_source,
        fps=float(frame_source.fps) if frame_source is not None else None,
    )
    records = pipeline.run_to_list()
    manifest = pipeline.build_manifest()
    stats = pipeline.stats
    print(f"  隐私脱敏分析: {stats.frames_processed} 帧, {stats.elapsed_seconds:.1f}s "
          f"({stats.average_fps:.1f} fps)")
    print(f"    人脸: {stats.frames_with_faces} 帧 / {stats.total_face_regions} 区域; "
          f"文本: {stats.frames_with_text} 帧 / {stats.total_text_regions} 区域; "
          f"PII: {stats.total_pii_masked} 区域; LLM 状态: {stats.llm_status}"
          f"（调用 {stats.llm_attempts} 次 / 成功 {stats.llm_successes} 次）")

    # Stage 0 期望的 manifest dict（与 scripts/run_privacy_redaction.py 对齐）
    manifest_dict = {
        "session_id": manifest.session_id,
        "source_uri": manifest.source_uri,
        "profile": manifest.profile,
        "producer": manifest.producer,
        "version": manifest.version,
        "config_hash": manifest.config_hash,
        "llm_available": manifest.llm_available,
        "llm_status": manifest.llm_status,
        "llm_attempts": manifest.llm_attempts,
        "llm_successes": manifest.llm_successes,
        "stats": {
            "total_frames": manifest.total_frames,
            "frames_with_faces": manifest.frames_with_faces,
            "frames_with_text": manifest.frames_with_text,
            "total_face_regions": manifest.total_face_regions,
            "total_text_regions": manifest.total_text_regions,
            "total_pii_masked": stats.total_pii_masked,
            "pii_categories_found": list(manifest.pii_categories_found),
        },
    }

    # 落盘 source-level manifest（analysis/privacy/manifest.json），
    # 供跨服务器消费/审计（与 hands/scene 同层的 staging 目录）。
    privacy_dir = _analysis_output_dir(output_dir, "privacy")
    privacy_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = privacy_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_dict, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"  → 隐私 manifest: {manifest_path}")
    return manifest_dict


def _read_video_frames(
    video_path: str | Path,
) -> tuple[list[np.ndarray], float]:
    """读取视频全部帧（BGR）与帧率，供场景分割使用。"""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise FileNotFoundError(f"无法打开视频: {video_path}")
    frames: list[np.ndarray] = []
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"视频没有可解码帧: {video_path}")
    return frames, fps


def _analysis_output_dir(output_dir: Path, kind: str) -> Path:
    """辅助分析产物 staging 目录（source-level，与 segment_candidates.json 同根）。

    hands / scene / privacy 的分析产物统一放这里，不再假装 batch 的
    prepared_segments/r0001/seg_000001 目录；batch_prepare 通过
    segment_candidates.json 的 analysis_artifacts 声明按 [start_ns, end_ns)
    裁切消费，任何 source-level → segment-level 的转换只发生在 batch 端。
    """
    return output_dir / "analysis" / kind


def _split_csv(value: object) -> tuple[str, ...]:
    """反序列化 scene writer 的逗号连接字符串字段。"""
    return tuple(part for part in str(value or "").split(",") if part)


def _scene_proposal_from_row(row: dict) -> "SceneProposal":
    """从 scene_proposals.parquet 行重建 SceneProposal（writer 列逆变换）。"""
    from zpds.scene.schemas import SceneProposal

    return SceneProposal(
        scene_id=str(row["scene_id"]),
        start_ns=int(row["start_ns"]),
        end_ns=int(row["end_ns"]),
        confidence=float(row["confidence"]),
        sources=_split_csv(row.get("sources")),
        boundary_scores=json.loads(str(row.get("boundary_scores") or "{}")),
        evidence_uris=_split_csv(row.get("evidence_uris")),
        short_span=bool(row.get("short_span", False)),
        producer=str(row.get("producer", "zpds.scene")),
        version=str(row.get("version", "v1")),
        config_hash=str(row.get("config_hash", "")),
    )


def _vlm_review_from_row(row: dict) -> "VLMReviewResult":
    """从 vlm_review.parquet 行重建 VLMReviewResult（writer 列逆变换）。"""
    from zpds.scene.schemas import VLMReviewResult

    return VLMReviewResult(
        scene_id=str(row["scene_id"]),
        scene_label=str(row["scene_label"]),
        task_label=str(row["task_label"]),
        decision=str(row["decision"]),
        confidence=float(row["confidence"]),
        reasons=str(row["reasons"]),
        evidence_frame_uris=_split_csv(row.get("evidence_frame_uris")),
        producer=str(row.get("producer", "zpds.scene.vlm")),
        version=str(row.get("version", "v1")),
        config_hash=str(row.get("config_hash", "")),
    )


def _try_reuse_scene_run(
    *,
    scene_config,
    profile: str,
    output_dir: Path,
    frames: list,
    fps: float,
) -> dict[str, object] | None:
    """复用已有场景产物：frame_count + config_hash 一致则跳过分割与 VLM。

    与手部复用同模式。返回与 _run_scene_analysis 同形的 dict；
    不可复用（产物缺失/校验失败/不一致）时返回 None。
    """
    from zpds.scene.pipeline import ScenePipelineRun

    scene_dir = _analysis_output_dir(Path(output_dir), "scene")
    summary_file = scene_dir / "run_summary.json"
    proposals_file = scene_dir / "scene_proposals.parquet"
    if not (summary_file.is_file() and proposals_file.is_file()):
        return None
    try:
        import pandas as _pd

        summary = json.loads(summary_file.read_text(encoding="utf-8"))
        if int(summary.get("frame_count", -1)) != len(frames):
            return None
        if str(summary.get("config_hash", "")) != scene_config.config_hash:
            return None
        scenes = tuple(
            _scene_proposal_from_row(dict(row))
            for row in _pd.read_parquet(str(proposals_file)).to_dict("records")
        )
        vlm_file = scene_dir / "vlm_review.parquet"
        vlm_results: tuple = ()
        if vlm_file.is_file():
            vlm_results = tuple(
                _vlm_review_from_row(dict(row))
                for row in _pd.read_parquet(str(vlm_file)).to_dict("records")
            )
    except (OSError, ValueError, KeyError) as exc:
        print(f"  场景产物校验失败，重新分析: {exc}")
        return None
    run = ScenePipelineRun(
        skipped=False,
        skip_reason=None,
        frame_count=len(frames),
        fps=fps,
        start_ns=int(summary.get("start_ns", 0)),
        end_ns=int(summary.get("end_ns", 0)),
        config_hash=scene_config.config_hash,
        profile=profile,
        scenes=scenes,
        vlm_results=vlm_results,
    )
    print(f"  场景产物已复用（帧数 {len(frames)}、config 一致）: {scene_dir}")
    print(f"  场景数: {len(scenes)}，VLM 复核: {len(vlm_results)}")
    return {"scene_pipeline_run": run, "scene_config": scene_config}


def _run_scene_analysis(
    *,
    video_path: str | Path,
    profile: str,
    output_dir: Path,
) -> dict[str, object]:
    """运行场景分割 + VLM 复核，返回 ScenePipelineRun 与 SceneConfig。

    运行结果落盘到 ``{output_dir}/analysis/scene/``（source-level staging：
    scene_proposals.parquet、vlm_review.parquet、run_summary.json），
    与 hands / privacy 目录同层；batch 端按候选区间消费。
    """
    from zpds.scene.config import SceneConfig
    from zpds.scene.pipeline import run_scene_pipeline
    from zpds.scene.vlm_review import (
        OpenAICompatibleVLMReviewer,
        VLMUnavailableError,
        load_scene_labels,
    )
    from zpds.scene.writer import write_scene_run

    scene_config_path = Path("configs/scene/default.yaml")
    profile_config_name = {
        "guida": "guida_ego",
        "dunjia": "dunjia_ego",
        "umi": "jianzhi_umi",
        "epic": "epic100",
        "a2d": "a2d_robot",
    }.get(profile, profile)
    profile_path = (
        Path("configs/qc_thresholds") / f"{profile_config_name}.yaml"
    )
    if profile_path.is_file():
        scene_config = SceneConfig.load_with_profile(
            scene_config_path, profile_path
        )
    else:
        scene_config = SceneConfig.load(scene_config_path)
    if not scene_config.enabled:
        return {"scene_pipeline_run": None, "scene_config": scene_config}
    frames, fps = _read_video_frames(video_path)

    # ---- 场景产物复用（帧数 + config_hash 一致则跳过分割与 VLM） ----
    reused = _try_reuse_scene_run(
        scene_config=scene_config,
        profile=profile,
        output_dir=output_dir,
        frames=frames,
        fps=fps,
    )
    if reused is not None:
        return reused

    reviewer = None
    if scene_config.vlm.enabled:
        if not scene_config.vlm.labels_path.strip():
            raise ValueError("scene.vlm.labels_path 未配置，无法运行 VLM 复核")
        labels = load_scene_labels(scene_config.vlm.labels_path)
        reviewer = OpenAICompatibleVLMReviewer(
            scene_config.vlm,
            labels=labels,
            config_hash=scene_config.config_hash,
        )
    vlm_unavailable_reason: str | None = None
    try:
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=scene_config,
            vlm_reviewer=reviewer,
        )
    except VLMUnavailableError as exc:
        vlm_unavailable_reason = str(exc)
        print(
            f"  [WARN] VLM 复核不可用，仅产出 scene 分割 "
            f"（Stage 10 记 SEMANTIC_NOT_RUN）: {exc}"
        )
        run = run_scene_pipeline(
            frames,
            fps=fps,
            config=scene_config,
            vlm_reviewer=None,
        )
    if run is not None and not run.skipped:
        scene_dir = _analysis_output_dir(Path(output_dir), "scene")
        scene_dir.mkdir(parents=True, exist_ok=True)
        written = write_scene_run(
            scene_dir,
            input_path=video_path,
            config_hash=run.config_hash,
            profile=run.profile or profile,
            fps=run.fps,
            frame_count=run.frame_count,
            start_ns=run.start_ns,
            end_ns=run.end_ns,
            scenes=run.scenes,
            vlm_results=run.vlm_results,
            review_queue=run.review_queue,
            skipped=run.skipped,
            skip_reason=run.skip_reason,
            vlm_unavailable_reason=vlm_unavailable_reason,
        )
        print(f"  → 场景报告: {written.summary_file}")
    return {"scene_pipeline_run": run, "scene_config": scene_config}


def main():
    parser = argparse.ArgumentParser(
        description="ZPDS Prepare — 从原始数据生成质量报告和候选分段方案"
    )
    parser.add_argument(
        "dataset",
        nargs="?",
        default=None,
        help="数据集路径 (墨现: 目录; 遁甲: .mcap 文件)",
    )
    parser.add_argument(
        "--profile", "-p",
        default="guida",
        choices=["guida", "dunjia", "umi", "epic"],
        help="数据源 profile (默认: guida)",
    )
    parser.add_argument(
        "--output", "-o",
        default=None,
        help="输出目录 (默认: output/moxian/ 或 output/dunjia/)",
    )
    parser.add_argument(
        "--config", "-c",
        default=CONFIG_PATH,
        help="YAML 配置路径",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="[Dunjia] H264 重建缓存目录（默认: 输出目录/.cache）",
    )
    parser.add_argument(
        "--formal-robot-qc",
        action="store_true",
        help="[UMI] 写入正式 quality views 和 revision manifest；保留原始数据不变",
    )
    parser.add_argument(
        "--with-hands",
        action="store_true",
        help="在主流程内运行手部检测与手部清洗，并把 hand_cleaning_report_path 传入 Stage 9"
             "（guida 默认开启；其他 profile 需显式指定）",
    )
    parser.add_argument(
        "--skip-hands",
        action="store_true",
        help="跳过手部检测与手部清洗（仅在 guida 默认开启时用于反向关闭）",
    )
    parser.add_argument(
        "--hands-fail-mode",
        choices=["strict", "degraded"],
        default="strict",
        help="手部分析失败时的处理（默认 strict）：strict=报错中断（human_hand=applicable "
             "Profile 要求 Hands 产物完整，禁止静默跳过）；degraded=降级跳过并继续",
    )
    parser.add_argument(
        "--with-privacy",
        action="store_true",
        help="在主流程内运行隐私脱敏分析（人脸/文本），把 PrivacyRunManifest 传入 Stage 0；"
             "不指定时 Stage 0 不参与级联",
    )
    parser.add_argument(
        "--with-scene",
        action="store_true",
        help="在主流程内运行场景分割与 VLM 复核，并把 scene 结果传入 Stage 10",
    )
    parser.add_argument(
        "--skip-cascade",
        action="store_true",
        help="跳过 QC 级联（Stage 3/5/6/7/8/9/11）",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="允许切分视频（默认不切分）：split 决策按长缺口等降级前逻辑执行，"
             "可能产出多个候选 segment",
    )
    parser.add_argument(
        "--review",
        default=None,
        help="平台审核返回的 quality_issues.json 路径（0.2.0）；提供时按审核结果"
             "（approved/rejected/modified/added）调整 issues 后重新切分",
    )
    parser.add_argument(
        "--epic-ho",
        default=None,
        help="[EPIC] hand-object .pkl 标注路径",
    )
    parser.add_argument(
        "--epic-mask",
        default=None,
        help="[EPIC] mask .pkl 标注路径",
    )
    parser.add_argument(
        "--epic-record",
        default=None,
        help="[EPIC] 单条 inventory record JSON (替代 --epic-ho/--epic-mask)",
    )
    args = parser.parse_args()

    # ---- 解析 profile 和 reader ----
    profile = args.profile
    rd = _get_reader(profile)

    # ---- 默认数据集路径 ----
    if args.dataset is None:
        if profile in ("dunjia", "umi"):
            parser.error(f"{profile} 模式必须指定 .mcap 文件路径")
        elif profile == "epic":
            parser.error(
                "epic 模式必须指定视频文件路径或 record JSON\n"
                "  --dataset /path/to/P01_01.mp4\n"
                "  --dataset output/epic/records/P01_01.json"
            )
        else:
            # 保持与旧版的兼容默认值
            dataset_path = "E:/datasets/egos/墨现"
    else:
        dataset_path = args.dataset

    # ---- EPIC: 从 record JSON 加载路径 ----
    epic_config: dict = {}
    if profile == "epic":
        record_json_path = args.epic_record or (
            dataset_path if dataset_path.endswith(".json") else None
        )
        if record_json_path:
            import json as _json
            with open(record_json_path, "r", encoding="utf-8") as _f:
                _record = _json.load(_f)
            video_path = _record.get("video_uri")
            if not video_path:
                parser.error(f"Record JSON 缺少 video_uri: {record_json_path}")
            dataset_path = video_path
            if _record.get("hand_object_uri"):
                epic_config["hand_object_path"] = _record["hand_object_uri"]
            if _record.get("mask_uri"):
                epic_config["mask_path"] = _record["mask_uri"]
            print(f"  从 record JSON 加载: {record_json_path}")
        else:
            # 命令行参数优先，否则由 caller 后续处理
            if args.epic_ho:
                epic_config["hand_object_path"] = args.epic_ho
            if args.epic_mask:
                epic_config["mask_path"] = args.epic_mask

    # 默认输出目录按 profile 分子目录
    if args.output is None:
        profile_subdirs = {"guida": "moxian", "dunjia": "dunjia", "umi": "umi", "epic": "epic"}
        subdir = profile_subdirs.get(profile, profile)
        output_dir = Path("output") / subdir
    else:
        output_dir = Path(args.output)
    config_path = args.config

    # EPIC: per-video 子目录 (output/epic/P01_01/)
    if profile == "epic":
        from zpds_prepare.readers.epic_inventory import parse_epic_id
        _, video_id = parse_epic_id(Path(dataset_path))
        output_dir = output_dir / video_id

    # 中文路径 fail-fast：输出 PNG/MP4 由 cv2.imread/imwrite 处理，
    # 不支持非 ASCII 路径（静默失败），命中立即报错而不是跑一段再挂。
    if not output_dir.as_posix().isascii():
        print(f"错误: 输出目录必须为纯 ASCII 路径（cv2 不支持中文路径）: {output_dir}")
        print("请改用英文目录，例如: --output output/taodai2/")
        return 1

    _load_dotenv()
    start_time = time.time()

    # ================================================================
    # Step 0: 加载配置
    # ================================================================
    cfg = load_config(config_path)

    # 黑屏检测参数
    bd = cfg.get("video", {}).get("black_detection", {})
    black_threshold = bd.get(
        "mean_intensity_threshold",
        cfg.get("video", {}).get("black_threshold", 5.0),
    )
    min_black_duration_s = bd.get(
        "min_duration_s",
        cfg.get("video", {}).get("min_black_duration_s", 0.5),
    )
    edge_tolerance_s = bd.get("edge_tolerance_s", 1.0)

    # 视频时间戳缺口参数
    tv = cfg.get("timestamp", {}).get("video", {})
    video_gap_factor = tv.get(
        "gap_factor",
        cfg.get("timestamp", {}).get("video_gap_factor", 2.0),
    )
    video_split_gap_s = tv.get("split_gap_s", 0.5)

    # IMU 时间戳缺口参数
    ti = cfg.get("timestamp", {}).get("imu", {})
    imu_gap_factor = ti.get(
        "gap_factor",
        cfg.get("timestamp", {}).get("imu_gap_factor", 3.0),
    )
    imu_split_gap_s = ti.get("split_gap_s", 1.0)

    # Segment 约束参数
    seg = cfg.get("segment", {})
    min_duration_s = seg.get("min_duration_s", 1.0)
    max_duration_s = seg.get("max_duration_s", 120.0)
    profile_depth = cfg.get(profile, {}).get("depth", {})
    depth_coverage_tolerance_ns = int(
        float(profile_depth.get("coverage_tolerance_s", 0.08))
        * 1_000_000_000
    )

    # ================================================================
    # Step 1: 读取数据
    # ================================================================
    step_header(1, "读取原始数据")

    print(f"  Profile:     {profile}")
    print(f"  数据集:      {dataset_path}")

    # EPIC: 传递 pickle 路径
    if profile == "epic":
        session = rd.read_session(dataset_path, config=epic_config if epic_config else None)
    elif profile == "dunjia":
        dunjia_depth = cfg.get("dunjia", {}).get("depth", {})
        cache_dir = Path(args.cache_dir) if args.cache_dir else output_dir / ".cache"
        session = rd.read_session(
            dataset_path,
            cache_dir=cache_dir,
            include_depth=bool(dunjia_depth.get("enabled", True)),
            require_depth=bool(dunjia_depth.get("required", True)),
        )
    elif profile == "umi":
        cache_dir = (
            Path(args.cache_dir)
            if args.cache_dir
            else output_dir / ".cache"
        )
        session = rd.read_session(dataset_path, cache_dir=cache_dir)
    else:
        session = rd.read_session(dataset_path)

    if args.formal_robot_qc:
        from zpds.qc.robot_integration import (
            run_a2d_formal_session,
            run_dunjia_formal_session,
            run_umi_formal_session,
        )

        if profile == "umi":
            delivery = run_umi_formal_session(session, output_dir)
        elif profile == "dunjia":
            delivery = run_dunjia_formal_session(session, output_dir)
        elif profile == "a2d":
            delivery = run_a2d_formal_session(
                session, dataset_path, output_dir
            )
        else:
            parser.error(
                f"--formal-robot-qc 不支持 {profile}；"
                f"可选: umi, dunjia, a2d"
            )
            return 1

        print(f"正式 QC revision 已写入: {(output_dir / 'revision.json').resolve()}")
        print("VLM 语义: not_run")
        print(f"质量视图: {', '.join(sorted(delivery.report.quality_views))}")
        return 0
    pv = session.primary_video

    # ---- 共享帧源（问题 16）：一次解码、各处消费 ----
    # 懒初始化：首次迭代才解码并写 JPEG 缓存；hands / privacy / scene /
    # stage3 / black_frame 等消费者共用同一实例，避免同一 MKV 被重复
    # 全量解码（默认 guida 运行 8 次 → 2 次，另 1 次为 bad_frame 子进程）。
    shared_frames = None
    if pv.video_path and Path(pv.video_path).is_file():
        from zpds_prepare.frame_source import SharedFrameSource

        shared_frames = SharedFrameSource(
            pv.video_path,
            cache_dir=output_dir / ".frame_cache",
        )

    meta = session.meta
    print(f"  设备:        {meta['device']}")
    print(f"  标称帧率:    {meta['fps']} fps")
    print(f"  分辨率:      {meta['width']}×{meta['height']}")
    print(f"  标称帧数:    {meta['frame_count']}")

    timestamps_ns = pv.timestamps_ns
    index_frames = pv.index_frames
    print(f"  Index 帧数:  {len(timestamps_ns)}")

    if len(timestamps_ns) >= 2:
        duration_s = (timestamps_ns[-1] - timestamps_ns[0]) / 1e9
        median_interval_ns = int(np.median(np.diff(timestamps_ns)))
        print(f"  时长:        {duration_s:.2f} s")
        print(f"  帧间隔中位数: {median_interval_ns:,} ns (~{1e9/median_interval_ns:.1f} fps)")

    session_start_ns = session.session_start_ns
    session_end_ns = session.session_end_ns
    session_id = session.session_id
    print(f"  Session ID:  {session_id}")

    # ================================================================
    # Step 1.5: 手部检测/清洗（Stage 9）、隐私脱敏（Stage 0）、场景分割（Stage 10）
    # ================================================================
    hand_report_path: str | None = None
    privacy_manifest: dict | None = None
    scene_pipeline_run = None
    scene_config = None
    # 处理状态（processing_status 输入）：记录各分析环节结果，
    # 随 quality_issues.json 落盘供平台区分「数据质量」与「处理过程」。
    processing_steps: dict[str, dict] = {}

    # guida（墨现）默认运行手部清洗；--skip-hands 反向关闭
    run_hands = (args.with_hands or profile == "guida") and not args.skip_hands
    hand_result: dict | None = None
    if run_hands and pv.video_path:
        step_header(15, "手部检测与手部清洗（Stage 9 输入）")
        from zpds.profiles.registry import get as _get_profile

        _profile_name = {
            "guida": "guida_ego",
            "dunjia": "dunjia_ego",
            "umi": "jianzhi_umi",
            "epic": "epic100",
            "a2d": "a2d_robot",
        }.get(profile, profile)
        _profile_obj = _get_profile(_profile_name)
        _hand_applicable = (
            _profile_obj is not None
            and _profile_obj.modalities.get("human_hand") == "applicable"
        )
        if not _hand_applicable:
            print("  手部分析跳过: human_hand=not_applicable")
            processing_steps["hands"] = {
                "status": "skipped",
                "detail": "human_hand=not_applicable",
            }
        else:
            hand_dir = _analysis_output_dir(output_dir, "hands")
            report_file = hand_dir / "hand_cleaning_report.json"
            hands_parquet_file = hand_dir / "hands_2d.parquet"
            reused = False
            if report_file.is_file() and hands_parquet_file.is_file():
                reused, reasons = _hands_cache_ok(
                    report_file, hands_parquet_file, pv.video_path, pv.timestamps_ns
                )
                if reused:
                    hand_report_path = str(report_file)
                    processing_steps["hands"] = {
                        "status": "complete",
                        "detail": "复用既有产物（指纹一致）",
                    }
                    print(f"  手部产物已存在，复用（指纹一致）: {hand_dir}")
                    print(f"  手部报告:    {hand_report_path}")
                    print(f"  hands 2D:    {hands_parquet_file}")
                else:
                    print(
                        "  手部产物指纹不匹配，重新推理: "
                        + ("; ".join(reasons) if reasons else "未知原因")
                    )
            if not reused:
                try:
                    hand_result = _run_hand_analysis(
                        video_path=pv.video_path,
                        timestamps_ns=pv.timestamps_ns,
                        output_dir=output_dir,
                        session_id=session_id,
                        stream_id=next(iter(session.video_streams), "ego_rgb"),
                        frame_source=shared_frames,
                    )
                    hand_report_path = hand_result.get("hand_cleaning_report_path")
                    processing_steps["hands"] = {
                        "status": "complete",
                        "detail": f"推理完成（{hand_result.get('model')}）",
                    }
                    print(f"  手部报告:    {hand_report_path}")
                    print(f"  hands 2D:    {hand_result.get('hands_parquet')}")
                except (
                    FileNotFoundError,
                    TypeError,
                    ValueError,
                    OSError,
                    ImportError,
                    RuntimeError,
                ) as exc:
                    if args.hands_fail_mode == "strict":
                        raise RuntimeError(
                            f"手部分析失败（--hands-fail-mode=strict）: {exc}"
                        ) from exc
                    processing_steps["hands"] = {
                        "status": "degraded",
                        "detail": f"Hands 失败降级跳过: {exc}",
                    }
                    print(f"  手部分析失败，degraded 模式降级跳过: {exc}")

    if args.with_privacy and pv.video_path:
        step_header(17, "隐私脱敏分析（Stage 0 输入）")
        try:
            privacy_manifest = _run_privacy_analysis(
                video_path=pv.video_path,
                profile=profile,
                output_dir=output_dir,
                session_id=session_id,
                frame_source=shared_frames,
            )
            processing_steps["privacy"] = {
                "status": "complete",
                "detail": f"llm_status={privacy_manifest.get('llm_status')}",
            }
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            OSError,
            ImportError,
            RuntimeError,
        ) as exc:
            processing_steps["privacy"] = {
                "status": "degraded",
                "detail": f"隐私脱敏分析跳过: {exc}",
            }
            print(f"  隐私脱敏分析跳过: {exc}")

    if args.with_scene and pv.video_path:
        step_header(16, "场景分割 + VLM 复核（Stage 10 输入）")
        try:
            scene_result = _run_scene_analysis(
                video_path=pv.video_path,
                profile=profile,
                output_dir=output_dir,
                frame_source=shared_frames,
            )
            scene_pipeline_run = scene_result["scene_pipeline_run"]
            scene_config = scene_result["scene_config"]
            print(
                f"  场景数:      {len(scene_pipeline_run.scenes)}，"
                f"VLM 复核: {len(scene_pipeline_run.vlm_results)}，"
                f"复核队列: {len(scene_pipeline_run.review_queue)}"
            )
            processing_steps["scene"] = {
                "status": "complete",
                "detail": (
                    f"{len(scene_pipeline_run.scenes)} 场景, "
                    f"{len(scene_pipeline_run.vlm_results)} VLM 复核"
                ),
            }
        except (
            FileNotFoundError,
            TypeError,
            ValueError,
            OSError,
            ImportError,
            RuntimeError,
        ) as exc:
            processing_steps["scene"] = {
                "status": "degraded",
                "detail": f"场景分析跳过: {exc}",
            }
            print(f"  场景分析跳过: {exc}")

    # MCAP profile：显示双时间戳信息
    if profile in ("dunjia", "umi") and index_frames:
        first = index_frames[0]
        print(f"  时间戳 (消息内):   {first['timestamp_ns']}")
        print(f"  log_time (MCAP):   {first.get('log_time_ns', 'N/A')}")
        print(f"  publish_time:      {first.get('publish_time_ns', 'N/A')}")

    # 显示所有流
    print(f"\n  视频流: {len(session.video_streams)} 个, "
          f"深度流: {len(session.depth_streams)} 个, "
          f"IMU 流: {len(session.imu_streams)} 个, "
          f"标注流: {len(session.annotation_streams)} 个")
    for stream_id, vs in session.video_streams.items():
        print(f"    [{stream_id}] {vs.frame_count} 帧, "
              f"{vs.width}×{vs.height}, {vs.fps} fps")
    for stream_id, depth_s in session.depth_streams.items():
        print(
            f"    [{stream_id}] {depth_s.frame_count} 帧, "
            f"{depth_s.width}×{depth_s.height}, {depth_s.fps} Hz, "
            f"dtype={depth_s.dtype}, unit={depth_s.unit}"
        )
    for stream_id, imu_s in session.imu_streams.items():
        print(f"    [{stream_id}] {len(imu_s.dataframe)} 样本, "
              f"{imu_s.sample_rate_hz} Hz")
    for stream_id, ts_s in session.time_series_streams.items():
        print(
            f"    [{stream_id}] {ts_s.num_samples} 样本, "
            f"{ts_s.expected_rate_hz} Hz, "
            f"unit={ts_s.metadata.get('unit')}, "
            f"semantic={ts_s.metadata.get('semantic_status')}"
        )
    for stream_id, ann_s in session.annotation_streams.items():
        print(f"    [{stream_id}] {ann_s.annotation_type}, "
              f"{len(ann_s.records)} 标注帧, "
              f"bbox={ann_s.bbox_format}")

    # ================================================================
    # Step 2: QC 级联（Stage 3/5/6/7/8/9/11）
    # ================================================================
    step_header(2, "QC 级联 (Stage 3/5/6/7/8/9/11)")

    # 启用 QC 级联进度日志（INFO 级别）
    logging.getLogger("zpds.qc").setLevel(logging.INFO)

    cascade_decisions: list = []
    cascade_issues: list = []  # Decision → QualityIssue 转换结果，供 segment_planner 消费
    cascade_overall_pass = True
    cascade_distribution = CascadeDistribution()
    if not args.skip_cascade:
        # 提取 IMU 数据供 Stage 6 使用（支持多 IMU 流）
        imu_context: dict[str, object] = {}
        imu_streams_data: list[dict] = []
        for imu_id, imu_s in session.imu_streams.items():
            df = imu_s.dataframe
            accel_cols = [c for c in df.columns if c in ("ax", "ay", "az")]
            gyro_cols = [c for c in df.columns if c in ("gx", "gy", "gz")]
            stream_data: dict[str, object] = {
                "stream_id": imu_id,
                "timestamps_ns": df["timestamp_ns"].tolist(),
            }
            if accel_cols + gyro_cols:
                stream_data["values"] = df[accel_cols + gyro_cols].to_numpy()
                stream_data["axis_names"] = accel_cols + gyro_cols
            imu_streams_data.append(stream_data)
            # 向后兼容：保留第一个 IMU 的 flat keys
            if not imu_context:
                imu_context["imu_timestamps_ns"] = stream_data["timestamps_ns"]
                imu_context["imu_stream_id"] = imu_id
                if "values" in stream_data:
                    imu_context["imu_values"] = stream_data["values"]
                    imu_context["imu_axis_names"] = stream_data["axis_names"]
        imu_context["imu_streams"] = imu_streams_data

        # 收集视频文件路径供 Stage 11 去重
        all_video_paths: list[str] = [
            str(vs.video_path) for vs in session.video_streams.values() if vs.video_path
        ]
        all_file_paths: list[str] = list(all_video_paths)
        # 读取已有 inventory 供跨 session 比对
        inventory_path = output_dir / "inventory.json"
        if inventory_path.exists():
            try:
                inv = _json.loads(inventory_path.read_text(encoding="utf-8"))
                for prev in inv.get("sessions", []):
                    for vp in prev.get("video_paths", []):
                        if vp not in all_video_paths:
                            all_video_paths.append(vp)
                    for fp in prev.get("file_paths", []):
                        if fp not in all_file_paths:
                            all_file_paths.append(fp)
            except Exception:
                pass

        robot_observation_checked_flag = False
        for stream_id, vs in session.video_streams.items():
            robot_observation_checked = robot_observation_checked_flag
            ctx: dict[str, object] = {
                "session_id": session_id,
                "session": session,
                "cfg": cfg,
                "robot_observation_checked": robot_observation_checked,
                "video_path": str(vs.video_path) if vs.video_path else "",
                "fps": float(vs.fps),
                "start_ns": int(vs.timestamps_ns[0]) if vs.timestamps_ns else 0,
                "profile": profile,
                "evidence_dir": str(output_dir / "evidence" / stream_id),
                "stream_id": stream_id,
                # Stage 11 去重：当前 session + 历史 inventory 的视频/文件路径
                "video_paths": all_video_paths,
                "file_paths": all_file_paths,
                # Stage 7 机器人信号：TimeSeriesStream 对象（session 级，幂等守卫防重复）
                "time_series_streams": session.time_series_streams,
                # Stage 8 标定：标定数据 + 视频流（用于分辨率一致性校验）
                "calibration": session.meta.get("calibration"),
                "video_streams_for_calib": session.video_streams,
                # Stage 9 手部清洗报告路径（guida 默认 / --with-hands 时传入）
                "hand_cleaning_report_path": hand_report_path,
                # Stage 0 隐私脱敏 manifest（--with-privacy 时传入；缺失时 Stage 0 被剔除）
                "privacy_manifest": privacy_manifest,
                # Stage 10 场景分割 + VLM 复核结果（--with-scene 时传入）
                "scene_pipeline_run": scene_pipeline_run,
                "scene_config": scene_config,
                # Stage 3 视觉检测共享帧源（问题 16）：提供时不再重复解码
                "frames": shared_frames,
            }
            # Stage 12 音频质量：把 Session 音频流转为 QC 可消费格式
            audio_streams_data: list[dict] = []
            for audio_id, audio_s in session.audio_streams.items():
                audio_streams_data.append({
                    "stream_id": audio_id,
                    "timestamps_ns": [p["timestamp_ns"] for p in audio_s.packets],
                    "packets": audio_s.num_packets,
                    "duration_s": (
                        audio_s.duration_ns / 1e9 if audio_s.num_packets >= 2 else 0.0
                    ),
                    "source_topic": "/robot0/sensor/audio",
                    "source_format": audio_s.format,
                })
            if audio_streams_data:
                ctx["audio_streams"] = audio_streams_data
                # 向后兼容 flat keys（清洗阶段无 WAV，只传时间戳/包数/时长）
                first_audio = audio_streams_data[0]
                ctx["audio_timestamps_ns"] = first_audio["timestamps_ns"]
                ctx["audio_duration_s"] = first_audio["duration_s"]
                ctx["audio_packets"] = first_audio["packets"]
            depth_s = session.depth_streams.get(stream_id)
            if depth_s is not None:
                if depth_s.timestamps_ns:
                    ctx["rgb_timestamps_ns"] = vs.timestamps_ns
                    ctx["depth_timestamps_ns"] = depth_s.timestamps_ns
                # 深度帧数据（供 Stage 5 使用，按优先级）：
                #   1. depth_frames — MCAP 等容器解码后的 numpy 采样帧（纯内存）
                #   2. depth_dir     — 已有 PNG 序列目录
                #   3. depth_source_files — 可独立读取的深度源文件列表
                if depth_s.depth_frames:
                    ctx["depth_frames"] = depth_s.depth_frames
                if depth_s.source_files and depth_s.source_kind != "mcap_compressed_image":
                    ctx["depth_dir"] = str(depth_s.source_files[0].parent)
                    ctx["depth_source_files"] = [str(p) for p in depth_s.source_files]
            ctx.update(imu_context)
            try:
                cascade = QCCascade.from_profile(profile)
                if privacy_manifest is None:
                    # 主流程未跑隐私脱敏 → 剔除 Stage 0，避免 manifest 缺失的假 quarantine
                    # （stage0_privacy 对 None manifest 返回 PRIVACY_COVERAGE_LOW + quarantine）
                    cascade.config.enabled_stages = [
                        s for s in cascade.config.enabled_stages if s != 0
                    ]
                report = cascade.run(ctx)
                cascade_decisions.extend(report.decisions)
                if not report.overall_pass:
                    cascade_overall_pass = False
                for d in report.decisions:
                    cascade_distribution.record(d)
                # Stage 7 无手观测视图是 session 级检查，只需跑一次
                robot_observation_checked_flag = True
                # 转换为 QualityIssue 供 segment_planner 消费
                cascade_issues.extend(
                    _decisions_to_issues(report.decisions, stream_id)
                )
                print(f"  [{stream_id}] Stage 级联完成: "
                      f"{len(report.decisions)} decisions, "
                      f"overall_pass={report.overall_pass}")
            except Exception as exc:
                print(f"  [{stream_id}] QC 级联异常: {exc}")
        if cascade_decisions:
            cascade_path = output_dir / "cascade_report.json"
            import json as _json
            cascade_path.parent.mkdir(parents=True, exist_ok=True)
            cascade_data = {
                "schema_version": "zpds.cascade_report.v1",
                "session_id": session_id,
                "overall_pass": cascade_overall_pass,
                # 数据质量状态（与 Prepared 阶段 package_validation 解耦）
                "quality_status": (
                    "fail"
                    if any(d.severity.value == "error" for d in cascade_decisions)
                    else (
                        "warn"
                        if any(d.severity.value == "warn" for d in cascade_decisions)
                        else "pass"
                    )
                ),
                "total_decisions": len(cascade_decisions),
                "distribution": cascade_distribution.to_dict(),
                "decisions": [
                    {
                        "stage": d.stage,
                        "reason": d.reason.value,
                        "severity": d.severity.value,
                        "message": d.message,
                        "disposition": d.disposition.value if d.disposition else None,
                        "detail": d.detail,
                    }
                    for d in cascade_decisions
                ],
            }
            cascade_path.write_text(
                _json.dumps(cascade_data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  级联报告: {cascade_path.resolve()}")
    else:
        print("  QC 级联已跳过 (--skip-cascade)")

    # 更新 inventory.json（跨 session 去重注册表）
    if not args.skip_cascade:
        try:
            import json as _json2
            inv: dict = {}
            if inventory_path.exists():
                inv = _json2.loads(inventory_path.read_text(encoding="utf-8"))
            current_entry = {
                "session_id": session_id,
                "profile": profile,
                "video_paths": [str(vs.video_path) for vs in session.video_streams.values() if vs.video_path],
                "file_paths": [str(vs.video_path) for vs in session.video_streams.values() if vs.video_path],
            }
            sessions: list = inv.get("sessions", [])
            # 覆盖同 session_id 的旧记录
            sessions = [s for s in sessions if s.get("session_id") != session_id]
            sessions.append(current_entry)
            inv["sessions"] = sessions
            inv["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            inventory_path.parent.mkdir(parents=True, exist_ok=True)
            inventory_path.write_text(
                _json2.dumps(inv, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"  去重注册表: {inventory_path.resolve()}")
        except Exception:
            pass

    # ================================================================
    # Step 3: 运行检测器（遍历所有流）
    # ================================================================
    step_header(3, "运行检测器")

    all_issues = []
    # 合并级联产出的 Decision（已转为 QualityIssue），让级联发现的问题影响分段方案
    if cascade_issues:
        all_issues.extend(cascade_issues)
        print(f"  从 QC 级联合并 {len(cascade_issues)} 条 issues")
    min_black_duration_ns = int(min_black_duration_s * 1_000_000_000)
    edge_tolerance_ns = int(edge_tolerance_s * 1_000_000_000)
    video_split_gap_ns = int(video_split_gap_s * 1_000_000_000)
    imu_split_gap_ns = int(imu_split_gap_s * 1_000_000_000)

    # ---- 视频流检测 ----
    for stream_id, vs in session.video_streams.items():
        print(f"\n  [{stream_id}]")

        # 2a. 帧数一致性
        if profile == "epic":
            # EPIC: 比较 ffprobe 声明 / 时间戳 / 标注范围
            ann_max = None
            for ann_s in session.annotation_streams.values():
                if ann_s.records:
                    max_idx = max(r["frame_index"] for r in ann_s.records)
                    ann_max = max(ann_max, max_idx + 1) if ann_max is not None else max_idx + 1
            fc_issues = detect_frame_count_mismatch(
                stream_id=stream_id,
                timestamps_ns=vs.timestamps_ns,
                declared_count=vs.frame_count,
                timestamp_count=len(vs.timestamps_ns),
                annotation_max_frame=ann_max,
            )
        else:
            fc_issues = detect_frame_count_mismatch(
                stream_id=stream_id,
                timestamps_ns=vs.timestamps_ns,
                index_frame_count=len(vs.timestamps_ns),
                meta_frame_count=vs.frame_count,
            )
        all_issues.extend(fc_issues)
        if fc_issues:
            for iss in fc_issues:
                if iss.issue_type == "annotation_frame_out_of_range":
                    print(f"    标注越界 [{iss.decision}]: "
                          f"标注最大帧={iss.details.get('annotation_max_frame')} > "
                          f"可解码帧数={iss.details.get('decoded_frame_count')}")
                else:
                    src_a = iss.details.get("source_a", "index_frame_count")
                    src_b = iss.details.get("source_b", "meta_frame_count")
                    val_a = iss.details.get("value_a", iss.details.get("index_frame_count"))
                    val_b = iss.details.get("value_b", iss.details.get("meta_frame_count"))
                    print(f"    帧数不一致 [{iss.decision}]: "
                          f"{src_a}={val_a} vs "
                          f"{src_b}={val_b}, "
                          f"diff={iss.details.get('difference')}")
        else:
            print(f"    帧数一致: {len(vs.timestamps_ns)}")

        # 2b. 坏帧检测
        bad_issues = detect_bad_frames(
            video_path=vs.video_path,
            timestamps_ns=vs.timestamps_ns,
            stream_id=stream_id,
        )
        all_issues.extend(bad_issues)
        if bad_issues:
            for iss in bad_issues:
                print(f"    坏帧 [{iss.decision}]: "
                      f"{iss.details.get('bad_frame_count')} 帧 "
                      f"({iss.details.get('bad_ratio', 0)*100:.1f}%)")

        # 2c. 黑屏检测
        black_issues = detect_black_frames(
            video_path=vs.video_path,
            timestamps_ns=vs.timestamps_ns,
            mean_intensity_threshold=black_threshold,
            min_duration_ns=min_black_duration_ns,
            edge_tolerance_ns=edge_tolerance_ns,
            frames=shared_frames,
        )
        all_issues.extend(black_issues)
        if black_issues:
            for iss in black_issues:
                print(f"    黑屏 [{iss.decision}]: "
                      f"{(iss.end_ns - iss.start_ns)/1e9:.2f}s "
                      f"({iss.details.get('frame_count', '?')} 帧)")

        # 2d. 视频时间戳缺口
        expected_interval_ns = int(1_000_000_000 / vs.fps)
        gap_issues = detect_timestamp_gaps(
            timestamps_ns=vs.timestamps_ns,
            expected_interval_ns=expected_interval_ns,
            gap_factor=video_gap_factor,
            split_gap_ns=video_split_gap_ns,
            stream_id=stream_id,
        )
        all_issues.extend(gap_issues)
        if gap_issues:
            for iss in gap_issues:
                print(f"    时间戳缺口 [{iss.decision}]: "
                      f"Frame {iss.details.get('frame_index', '?')}, "
                      f"gap={iss.details.get('gap_ms', '?')}ms")

        if not any([fc_issues, bad_issues, black_issues, gap_issues]):
            print("    ✓ 无异常")

    # ---- 必需深度流覆盖范围 ----
    depth_coverage_issues = detect_depth_coverage(
        video_streams=session.video_streams,
        depth_streams=session.depth_streams,
        tolerance_ns=depth_coverage_tolerance_ns,
    )
    all_issues.extend(depth_coverage_issues)
    for stream_id, depth_s in session.depth_streams.items():
        print(f"\n  [{stream_id}]")
        stream_issues = [
            issue
            for issue in depth_coverage_issues
            if issue.stream_id == stream_id
        ]
        if stream_issues:
            for issue in stream_issues:
                missing_ns = (
                    issue.details.get("missing_tail_ns")
                    or issue.details.get("missing_head_ns")
                    or 0
                )
                print(
                    f"    覆盖边界 [{issue.decision}]: {issue.issue_type}, "
                    f"缺失 {missing_ns / 1e9:.3f}s, "
                    f"覆盖率 {issue.details['coverage_ratio']:.2%}"
                )
        else:
            print("    ✓ 覆盖 RGB 公共时间范围")

    # ---- IMU 流检测 ----
    for stream_id, imu_s in session.imu_streams.items():
        print(f"\n  [{stream_id}]")

        expected_interval_ns = int(1_000_000_000 / imu_s.sample_rate_hz)
        imu_issues = detect_imu_gaps(
            imu=imu_s.dataframe,
            expected_interval_ns=expected_interval_ns,
            gap_factor=imu_gap_factor,
            split_gap_ns=imu_split_gap_ns,
            stream_id=stream_id,
        )
        all_issues.extend(imu_issues)
        if imu_issues:
            for iss in imu_issues:
                print(f"    IMU 缺口 [{iss.decision}]: "
                      f"Sample {iss.details.get('sample_index', '?')}, "
                      f"gap={iss.details.get('gap_s', '?')}s")
        else:
            print("    ✓ 无异常")

    # ================================================================
    # Step 4: 汇总分析
    # ================================================================
    step_header(4, "汇总分析")

    if not args.split:
        downgrade_split_issues(all_issues)

    summary = get_issue_summary(all_issues)
    print(f"  总异常数: {summary['total']}")
    if summary["total"] > 0:
        print(f"  按类型: {summary['by_type']}")
        print(f"  按处置: {summary['by_decision']}")

    # ================================================================
    # Step 5: 写出 quality_issues.json
    # ================================================================
    step_header(5, "写出 quality_issues.json")

    qi_path = write_quality_issues(
        output_path=output_dir / "quality_issues.json",
        issues=all_issues,
        source_session_id=session_id,
        processing_steps=processing_steps,
    )
    _processing_status, _ = derive_processing_status(processing_steps)
    print(f"  输出: {qi_path.resolve()}")
    print(f"  quality_status: {derive_quality_status(all_issues)} / "
          f"processing_status: {_processing_status}")

    # ================================================================
    # Step 5.5: 应用平台审核结果（--review）后重新切分
    # ================================================================
    if args.review:
        step_header(5.5, "应用平台审核结果 (--review)")
        from zpds_prepare.writers.review_applier import apply_review

        with open(args.review, "r", encoding="utf-8") as _rf:
            reviewed_payload = json.load(_rf)
        all_issues, review_stats = apply_review(all_issues, reviewed_payload)
        s = review_stats
        print(f"  审核结果: approved={s.approved} rejected={s.rejected} "
              f"modified={s.modified} added={s.added} kept={s.kept}")
        print(f"  调整后 issues: {len(all_issues)} 条（继续生成候选）")

    # ================================================================
    # Step 6: 生成候选 Segment
    # ================================================================
    step_header(6, "生成候选 Segment")

    candidates = plan_segments(
        issues=all_issues,
        session_start_ns=session_start_ns,
        session_end_ns=session_end_ns,
        min_duration_ns=int(min_duration_s * 1_000_000_000),
        max_duration_ns=int(max_duration_s * 1_000_000_000),
        no_split=not args.split,
    )

    print(f"  候选数: {len(candidates)}")
    for c in candidates:
        print(f"    {c.candidate_id}: "
              f"{c.source_start_ns:,} → {c.source_end_ns:,} "
              f"({c.duration_ns / 1e9:.2f}s, {c.reason})")
        for iss in c.issues_in_span:
            print(f"      ⚠ [{iss['decision']}] {iss['issue_type']}: "
                  f"{(iss['end_ns'] - iss['start_ns']) / 1e9:.2f}s")

    # ================================================================
    # Step 7: 写出 segment_candidates.json
    # ================================================================
    step_header(7, "写出 segment_candidates.json")

    # source-level 辅助产物声明（analysis/hands|scene|privacy/），
    # batch_prepare 据此按每个候选 [start_ns, end_ns) 裁切消费；
    # 不写路径进每个 candidate——产物是 session 级，candidate 是切分级。
    analysis_artifacts: dict[str, Any] = {}
    if hand_report_path:
        analysis_artifacts["hands"] = {
            "schema_version": "zpds.hands.v1",
            "timebase": "source_clock",
            "uri": "analysis/hands/hands_2d.parquet",
            "frames_uri": "analysis/hands/hand_cleaning_frames.parquet",
            "report_uri": "analysis/hands/hand_cleaning_report.json",
            "video_stream_id": (
                hand_result.get("video_stream_id") if hand_result else None
            ),
            "model": hand_result.get("model") if hand_result else None,
            "source_fps": float(pv.fps),
        }
    if scene_pipeline_run is not None and not scene_pipeline_run.skipped:
        analysis_artifacts["scene"] = {
            "schema_version": "zpds.scene.v1",
            "timebase": "source_clock",
            "uri": "analysis/scene/scene_proposals.parquet",
            "summary_uri": "analysis/scene/run_summary.json",
            "vlm_uri": "analysis/scene/vlm_review.parquet",
            "config_hash": scene_pipeline_run.config_hash,
            "fps": scene_pipeline_run.fps,
            "frame_count": scene_pipeline_run.frame_count,
        }
    if privacy_manifest:
        analysis_artifacts["privacy"] = {
            "schema_version": "zpds.privacy.manifest.v1",
            "uri": "analysis/privacy/manifest.json",
            "config_hash": privacy_manifest.get("config_hash"),
        }

    sc_path = write_segment_candidates(
        output_path=output_dir / "segment_candidates.json",
        candidates=candidates,
        source_session_id=session_id,
        source_start_ns=session_start_ns,
        source_end_ns=session_end_ns,
        analysis_artifacts=analysis_artifacts,
    )
    print(f"  输出: {sc_path.resolve()}")

    # ================================================================
    # 完成
    # ================================================================
    elapsed = time.time() - start_time
    print(f"\n{'=' * 60}")
    print("  完成")
    print(f"  耗时:        {elapsed:.1f}s")
    print(f"  QC 级联:     {'已跳过' if args.skip_cascade else f'{len(cascade_decisions)} decisions'}")
    print(f"  发现异常:    {summary['total']}")
    print(f"  候选 Segment: {len(candidates)}")
    if hand_report_path:
        print(f"  手部报告:    {hand_report_path}")
    if scene_pipeline_run is not None and not scene_pipeline_run.skipped:
        print(
            f"  场景/VLM:    {len(scene_pipeline_run.scenes)} scenes, "
            f"{len(scene_pipeline_run.vlm_results)} reviewed, "
            f"{len(scene_pipeline_run.review_queue)} in review queue"
        )
    print(f"  输出目录:    {output_dir.resolve()}")
    print(f"{'=' * 60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
