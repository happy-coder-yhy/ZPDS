"""Stage 8: 相机标定完整性检查。

检查 Session 中标定数据的存在性和基本合理性：
  1. 标定数据是否存在
  2. 每路相机是否有 intrinsics（fx, fy, cx, cy）
  3. 内参值是否在合理范围
  4. distortion_model 是否可识别
  5. 分辨率是否与视频流一致

不做重投影误差计算（属于标定模块的职责）。
"""

from __future__ import annotations

from zpds.core.decisions import Decision, Disposition, ReasonCode, Severity
from zpds.qc.cascade import register_stage

# 已知畸变模型
_KNOWN_DISTORTION_MODELS: frozenset[str] = frozenset({
    "pinhole",
    "brown_conrady",
    "equidistant",
    "rational_polynomial",
    "plumb_bob",
    "fisheye",
})


def check(
    calibration: dict | None = None,
    video_streams: dict | None = None,
    *,
    stage_config: dict | None = None,
) -> list[Decision]:
    """Stage 8 统一检查入口：标定完整性。

    Parameters
    ----------
    calibration : Optional[dict]
        session.meta["calibration"]，格式：
        {"calibration_id": str, "cameras": [{camera_id, model, intrinsics, resolution}]}
    video_streams : Optional[dict]
        {stream_id: VideoStream} 映射，用于分辨率一致性校验。
    stage_config : Optional[dict]
        阈值覆盖。

    Returns
    -------
    list[Decision]
    """
    cfg = stage_config or {}
    if not cfg.get("enabled", True):
        return []

    if not calibration:
        return [
            Decision(
                stage=8,
                reason=ReasonCode.CHECK_NOT_APPLICABLE,
                severity=Severity.INFO,
                message="Calibration check skipped: no calibration data",
                disposition=Disposition.KEEP,
                detail={"applicability": "not_applicable"},
            ),
        ]

    cameras = calibration.get("cameras", [])
    if not cameras:
        return [
            Decision(
                stage=8,
                reason=ReasonCode.INTRINSICS_MISSING,
                severity=Severity.WARN,
                message=(
                    "标定数据存在但 cameras 列表为空"
                ),
                disposition=Disposition.KEEP_WITH_FLAG,
                detail={
                    "calibration_id": calibration.get("calibration_id"),
                },
            ),
        ]

    video_streams = video_streams or {}
    decisions: list[Decision] = []

    for cam in cameras:
        cam_id = str(cam.get("camera_id", "unknown"))
        intr = cam.get("intrinsics", {})
        if not isinstance(intr, dict):
            decisions.append(
                Decision(
                    stage=8,
                    reason=ReasonCode.INTRINSICS_MISSING,
                    severity=Severity.ERROR,
                    message=f"相机 {cam_id}: intrinsics 缺失或格式错误",
                    disposition=Disposition.QUARANTINE,
                    detail={"camera_id": cam_id},
                ),
            )
            continue

        fx = intr.get("fx")
        fy = intr.get("fy")
        cx = intr.get("cx")
        cy = intr.get("cy")

        # ---- 1. 必填字段存在 + 正数 ----
        for key, val in [("fx", fx), ("fy", fy), ("cx", cx), ("cy", cy)]:
            if val is None:
                decisions.append(
                    Decision(
                        stage=8,
                        reason=ReasonCode.INTRINSICS_MISSING,
                        severity=Severity.ERROR,
                        message=f"相机 {cam_id}: 缺少内参 {key}",
                        disposition=Disposition.QUARANTINE,
                        detail={
                            "camera_id": cam_id,
                            "missing_param": key,
                        },
                    ),
                )
            elif not isinstance(val, (int, float)) or val <= 0:
                decisions.append(
                    Decision(
                        stage=8,
                        reason=ReasonCode.INTRINSICS_MISSING,
                        severity=Severity.ERROR,
                        message=(
                            f"相机 {cam_id}: 内参 {key}={val} 无效（需 > 0）"
                        ),
                        disposition=Disposition.QUARANTINE,
                        detail={
                            "camera_id": cam_id,
                            "param": key,
                            "value": val,
                        },
                    ),
                )

        # ---- 2. 分辨率一致性 ----
        vs = video_streams.get(cam_id)
        if vs is not None and isinstance(vs, object):
            res = cam.get("resolution", {})
            calib_w = res.get("width") if isinstance(res, dict) else None
            calib_h = res.get("height") if isinstance(res, dict) else None
            vs_w = getattr(vs, "width", 0)
            vs_h = getattr(vs, "height", 0)
            if calib_w is not None and vs_w > 0 and abs(calib_w - vs_w) > 10:
                decisions.append(
                    Decision(
                        stage=8,
                        reason=ReasonCode.INTRINSICS_MISSING,
                        severity=Severity.WARN,
                        message=(
                            f"相机 {cam_id}: 标定宽度 {calib_w} 与视频宽度 "
                            f"{vs_w} 不一致（偏差 {abs(calib_w - vs_w)}px）"
                        ),
                        disposition=Disposition.KEEP_WITH_FLAG,
                        detail={
                            "camera_id": cam_id,
                            "calibration_width": calib_w,
                            "video_width": vs_w,
                        },
                    ),
                )
            if calib_h is not None and vs_h > 0 and abs(calib_h - vs_h) > 10:
                decisions.append(
                    Decision(
                        stage=8,
                        reason=ReasonCode.INTRINSICS_MISSING,
                        severity=Severity.WARN,
                        message=(
                            f"相机 {cam_id}: 标定高度 {calib_h} 与视频高度 "
                            f"{vs_h} 不一致（偏差 {abs(calib_h - vs_h)}px）"
                        ),
                        disposition=Disposition.KEEP_WITH_FLAG,
                        detail={
                            "camera_id": cam_id,
                            "calibration_height": calib_h,
                            "video_height": vs_h,
                        },
                    ),
                )

        # ---- 3. 畸变模型识别 ----
        distortion_model = intr.get("distortion_model", "")
        if distortion_model and distortion_model not in _KNOWN_DISTORTION_MODELS:
            decisions.append(
                Decision(
                    stage=8,
                    reason=ReasonCode.INTRINSICS_MISSING,
                    severity=Severity.WARN,
                    message=(
                        f"相机 {cam_id}: 未知畸变模型 "
                        f"{distortion_model!r}"
                    ),
                    disposition=Disposition.KEEP_WITH_FLAG,
                    detail={
                        "camera_id": cam_id,
                        "distortion_model": distortion_model,
                        "known_models": sorted(_KNOWN_DISTORTION_MODELS),
                    },
                ),
            )

    # ---- 汇总 ----
    if not decisions:
        decisions.append(
            Decision(
                stage=8,
                reason=ReasonCode.SOURCE_QUALITY_FLAG,
                severity=Severity.INFO,
                message=(
                    f"Calibration check passed: "
                    f"{len(cameras)} camera(s) verified"
                ),
                disposition=Disposition.KEEP,
                detail={
                    "calibration_id": calibration.get("calibration_id"),
                    "camera_count": len(cameras),
                    "camera_ids": [c.get("camera_id") for c in cameras],
                },
            ),
        )

    return decisions


# ---------------------------------------------------------------------------
# QCCascade 注册入口
# ---------------------------------------------------------------------------


@register_stage(8)
def _check_stage8(context: dict) -> list[Decision]:
    """Stage 8 QCCascade 入口：从 context dict 提取标定数据并检查。

    Stage 8 是 session 级检查，只需执行一次。
    """
    if context.get("_stage8_done"):
        return []
    context["_stage8_done"] = True

    stage_config = context.get("stage_config", {})

    # 适用性守卫：仅 robot/end_effector profile 运行
    profile = context.get("profile")
    if profile:
        from zpds.profiles.registry import get

        registered = get(str(profile))
        if registered is not None:
            modalities = registered.modalities
            if (
                modalities.get("end_effector") != "applicable"
                and modalities.get("human_hand") != "applicable"
            ):
                return [
                    Decision(
                        stage=8,
                        reason=ReasonCode.CHECK_NOT_APPLICABLE,
                        severity=Severity.INFO,
                        message=(
                            "Calibration check skipped: "
                            "profile does not require camera calibration"
                        ),
                        disposition=Disposition.KEEP,
                        detail={"applicability": "not_applicable"},
                    ),
                ]

    calibration = context.get("calibration")
    video_streams = context.get("video_streams_for_calib")
    return check(
        calibration=calibration,
        video_streams=video_streams,
        stage_config=stage_config,
    )


__all__ = ["check", "_check_stage8"]
