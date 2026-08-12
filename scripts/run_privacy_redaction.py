"""隐私脱敏 CLI 入口。

用法:
    python scripts/run_privacy_redaction.py --input video.mp4 --profile guida_ego
    python scripts/run_privacy_redaction.py --input video.mp4 --profile dunjia_ego --output output/privacy/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser(description="ZPDS 视频隐私脱敏：人脸模糊 + PII 文本遮挡")
    parser.add_argument("--input", "-i", required=True, help="输入视频路径")
    parser.add_argument("--profile", "-p", default="guida_ego",
                        choices=["guida_ego", "dunjia_ego", "jianzhi_umi", "a2d_robot", "epic100"])
    parser.add_argument("--config", "-c", default=None, help="隐私配置 YAML（默认 configs/privacy/default.yaml）")
    parser.add_argument("--output", "-o", default=None, help="输出目录（默认 output/privacy/<session_id>/）")
    parser.add_argument("--skip-faces", action="store_true", help="跳过人脸模糊")
    parser.add_argument("--skip-text", action="store_true", help="跳过文本检测与遮挡")
    parser.add_argument("--max-frames", type=int, default=None, help="最多处理帧数（调试用）")
    parser.add_argument("--reset-frames", default=None,
                        help="强制检测帧号（逗号分隔，如 100,250；"
                             "场景边界等画面布局剧变点，命中时重置 KLT 传播缓存）")
    args = parser.parse_args()

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.is_file():
        print(f"输入视频不存在: {input_path}", file=sys.stderr)
        return 1

    # ---- 加载配置 ----
    config_path = Path(args.config) if args.config else Path("configs/privacy/default.yaml")
    from zpds.privacy.config import PrivacyConfig
    try:
        pcfg = PrivacyConfig.load(config_path)
    except FileNotFoundError:
        print(f"[warn] 隐私配置不存在 ({config_path})，使用默认值")
        pcfg = PrivacyConfig.defaults()

    # ---- 路由 ----
    from zpds.privacy.backend_router import PrivacyBackendPolicy
    policy = PrivacyBackendPolicy.from_profile(args.profile)

    # 命令行覆盖：跳过某类检测
    if args.skip_faces or args.skip_text:
        face_app = "not_applicable" if args.skip_faces else policy.face_applicability
        text_app = "not_applicable" if args.skip_text else policy.text_applicability
        policy = PrivacyBackendPolicy(
            face_applicability=face_app,   # type: ignore[arg-type]
            text_applicability=text_app,   # type: ignore[arg-type]
        )

    # 不适合的检测器提前告知
    if not policy.face_enabled:
        print(f"[{args.profile}] 人脸检测: not_applicable — 跳过")
    if not policy.text_enabled:
        print(f"[{args.profile}] 文本检测: not_applicable — 跳过")

    # ---- 输出目录 ----
    session_id = input_path.stem
    output_dir = Path(args.output) if args.output else Path("output/privacy") / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- 运行 Pipeline ----
    from zpds.privacy.pipeline import PrivacyPipeline

    reset_frames = None
    if args.reset_frames:
        reset_frames = {
            int(v) for v in args.reset_frames.split(",") if v.strip()
        }

    pipeline = PrivacyPipeline(
        input_path,
        config=pcfg,
        policy=policy,
        profile=args.profile,
        session_id=session_id,
        max_frames=args.max_frames,
        # 稀疏检测：间隔从配置读取，中间帧 KLT 光流传播
        face_interval=pcfg.face_interval_frames,
        text_interval=pcfg.text_interval_frames,
        reset_frames=reset_frames,
    )
    # 真脱敏必须严格：LLM 未配置时文本 PII 无法分类，拒绝执行（pipeline 层已
    # 降级为 llm_status=not_configured，这里显式把关）
    if not pipeline.llm_configured:
        raise SystemExit(
            "错误: LLM API key 未配置（DASHSCOPE_API_KEY），拒绝执行脱敏"
            "（文本 PII 将无法分类）"
        )

    print(f"\n脱敏开始: {input_path}")
    print(f"  Profile: {args.profile}")
    print(f"  人脸: {'启用' if policy.face_enabled else '跳过'} ({policy.face_applicability})")
    print(f"  文本: {'启用' if policy.text_enabled else '跳过'} ({policy.text_applicability})")
    print(f"  输出: {output_dir.resolve()}")
    print()

    records = pipeline.run_to_list()
    stats = pipeline.stats
    manifest = pipeline.build_manifest()

    print(f"处理完成: {stats.frames_processed} 帧, {stats.elapsed_seconds:.1f}s ({stats.average_fps:.1f} fps)")
    print(f"  人脸: {stats.frames_with_faces} 帧, {stats.total_face_regions} 区域")
    print(f"  文本: {stats.frames_with_text} 帧, {stats.total_text_regions} 区域")
    print(f"  PII 脱敏: {stats.total_pii_masked} 区域, 类别: {list(stats.pii_categories_found)}")
    print(f"  LLM 状态: {stats.llm_status}"
          f"（调用 {stats.llm_attempts} 次 / 成功 {stats.llm_successes} 次）")

    # ---- 写出产物 ----
    from zpds.privacy.writer import write_manifest, write_redacted_video, write_run_summary

    video_path = write_redacted_video(records, output_dir / "redacted.mp4")
    print(f"\n脱敏视频: {video_path.resolve()}")

    manifest_path = write_manifest(records, manifest, output_dir / "redaction_manifest.parquet")
    print(f"Manifest:  {manifest_path.resolve()}")

    summary_path = write_run_summary(manifest, output_dir / "run_summary.json")
    print(f"Summary:   {summary_path.resolve()}")

    # ---- QC 决策 ----
    from zpds.qc.stage0_privacy import build_privacy_view, check as stage0_check

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
            "total_pii_masked": manifest.total_pii_masked,
            "pii_categories_found": list(manifest.pii_categories_found),
        },
    }

    privacy_cfg = {
        "face": {"applicability": policy.face_applicability},
        "text": {"applicability": policy.text_applicability},
    }

    decisions = stage0_check(manifest=manifest_dict, stage_config={"enabled": True, **privacy_cfg})
    view = build_privacy_view(decisions)

    print(f"\nQC Stage 0:")
    print(f"  privacy_ready: {view.ready} ({view.disposition.value})")
    for d in decisions:
        print(f"  [{d.severity.value}] {d.reason.value}: {d.message}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
