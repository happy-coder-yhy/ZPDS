"""Prepared Segment 视频脱敏的公共入口：检测 → 等长重渲染 → manifest。

供两条 Segment 生成链路复用：
- ``batch_prepare.py``（guida/dunjia/umi/epic）
- ``scripts/pipeline_a2d.py`` / ``scripts/prepare_a2d_segment.py``（A2D 真机）

对已转码的 segment 视频原地脱敏（人脸模糊 + 文本遮挡），覆盖原转码产物，
脱敏版即训练用产物。按 Profile 路由 face/text 适用性（如 A2D 人脸不适用、
文本适用）。

等长重渲染由 :func:`zpds.privacy.writer.write_redacted_video` 承担
（source_video 补帧，无遮挡帧写原帧，保证帧数与时序一致）。
"""

from __future__ import annotations

from pathlib import Path


def redact_video_in_place(
    records,
    video_path: Path,
    *,
    face_method: str = "blur",
    text_method: str = "black_rect",
    blur_ksize: int = 41,
    blur_sigma: int = 15,
) -> None:
    """按逐帧遮挡区域重渲染视频并原地覆盖（等长，无丢帧）。

    遮挡已在 PrivacyPipeline 检测阶段完成（``records.redacted_frame``）；
    写出统一走 :func:`zpds.privacy.writer.write_redacted_video`，以
    源视频为底帧补齐无遮挡帧（source_video=video_path）。

    face_method/text_method/blur_ksize/blur_sigma 参数仅为兼容历史签名
    保留（检测阶段已按 PrivacyConfig 遮挡，这里不生效）。
    """
    from zpds.privacy.writer import write_redacted_video

    write_redacted_video(
        records,
        video_path,
        source_video=video_path,
        recode_h264=True,
    )


def redact_segment_videos(
    video_meta: list[dict],
    video_results: list[dict],
    output_dir: Path,
    profile: str,
    reset_frames: set[int] | None = None,
    config_path: str | Path | None = None,
) -> int:
    """对转码后的 segment 视频做隐私脱敏，原地覆盖 ``output_mp4``。

    按 Profile 路由（face/text 适用性）：
    - 两者都不适用（如遁甲/UMI/A2D 人脸不适用但仍可能拍文本，text 默认 applicable）
    - 人脸模糊 + 文本遮挡后覆盖原转码产物，脱敏版即训练用产物

    Args:
        video_meta: 每个元素含 ``output_mp4``（转码产物路径）。
        video_results: 与 video_meta 对齐的 segment 流结果，逐项补写
            ``redacted`` / ``redaction_manifest_uri`` / ``redaction_stats``
            （build_segment_json 会透传进 segment.json 的 redaction 条目）。
        output_dir: 未使用（manifest 随视频所在目录写出），保留参数以
            batch_prepare 兼容。
        reset_frames: 相对本视频的强制检测帧号（场景边界等布局剧变点），
            命中时重新完整检测并重置 KLT 传播缓存。
        config_path: privacy 配置路径；默认 ``configs/privacy/default.yaml``
            （相对项目根解析）。

    Returns:
        完成脱敏的视频流数量。
    """
    from zpds.privacy.backend_router import PrivacyBackendPolicy
    from zpds.privacy.config import PrivacyConfig
    from zpds.privacy.pipeline import PrivacyPipeline
    from zpds.privacy.writer import write_manifest

    policy = PrivacyBackendPolicy.from_profile(profile)
    if not policy.face_enabled and not policy.text_enabled:
        for vr in video_results:
            vr["redaction_skipped"] = True
            vr["redaction_skip_reason"] = (
                f"profile {profile}: face={policy.face_applicability}, "
                f"text={policy.text_applicability}"
            )
        return 0

    # 项目根绝对路径（batch_prepare 可能从任意 cwd 启动）
    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent.parent.parent
            / "configs/privacy/default.yaml"
        )
    try:
        pcfg = PrivacyConfig.load(config_path)
    except FileNotFoundError:
        print(f"[warn] 隐私配置不存在 ({config_path})，使用默认值")
        pcfg = PrivacyConfig.defaults()

    redacted_count = 0
    for vm, vr in zip(video_meta, video_results):
        stream_id = vr["stream_id"]
        mp4_path = Path(vm["output_mp4"])
        if not mp4_path.is_file():
            vr["redaction_skipped"] = True
            vr["redaction_skip_reason"] = f"转码产物不存在: {mp4_path}"
            continue
        print(
            f"  脱敏 [{stream_id}]: "
            f"人脸={'启用' if policy.face_enabled else '跳过'} "
            f"文本={'启用' if policy.text_enabled else '跳过'}"
        )
        pipeline = PrivacyPipeline(
            mp4_path,
            config=pcfg,
            policy=policy,
            profile=profile,
            session_id=stream_id,
            # 稀疏检测：检测帧之间用 KLT 光流传播遮挡区域
            face_interval=pcfg.face_interval_frames,
            text_interval=pcfg.text_interval_frames,
            reset_frames=reset_frames,
        )
        records = pipeline.run_to_list()
        manifest = pipeline.build_manifest()

        # 原地覆盖：脱敏视频即为训练用产物（等长重渲染，无丢帧）
        redact_video_in_place(
            records,
            mp4_path,
            face_method=pcfg.face_method,
            text_method=pcfg.redaction_text_method,
            blur_ksize=pcfg.face_blur_ksize,
            blur_sigma=pcfg.face_blur_sigma,
        )
        manifest_path = mp4_path.parent / f"{stream_id}_redaction_manifest.parquet"
        write_manifest(records, manifest, manifest_path)

        vr["redacted"] = True
        vr["redaction_manifest_uri"] = f"data/{stream_id}_redaction_manifest.parquet"
        vr["redaction_face"] = policy.face_enabled
        vr["redaction_text"] = policy.text_enabled
        vr["redaction_stats"] = {
            "frames_processed": manifest.total_frames,
            "frames_with_faces": manifest.frames_with_faces,
            "frames_with_text": manifest.frames_with_text,
            "face_regions": manifest.total_face_regions,
            "text_regions": manifest.total_text_regions,
            "pii_categories_found": list(manifest.pii_categories_found),
            "llm_available": manifest.llm_available,
        }
        redacted_count += 1
    return redacted_count


__all__ = [
    "redact_segment_videos",
    "redact_video_in_place",
]
