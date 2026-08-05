"""Stage 10: 场景分割 + VLM 复核 → QC Decision。

挂在 Stage 10 语义门下。scene.enabled=false、流水线跳过或缺少
scene 运行结果时整个 scene 级联跳过；软阈值只出 WARN/quarantine。
"""

from __future__ import annotations

from zpds.core.decisions import Decision
from zpds.qc.cascade import register_stage
from zpds.scene.pipeline import ScenePipelineRun
from zpds.scene.qc_integration import build_scene_decisions


@register_stage(10)
def _check_stage10(context: dict) -> list[Decision]:
    """从 cascade context 读取 scene 运行结果并转成决策。"""

    stage_config = context.get("stage_config", {})
    if not stage_config.get("enabled", True):
        return []
    run = context.get("scene_pipeline_run")
    if not isinstance(run, ScenePipelineRun):
        return []
    if run.skipped:
        return []
    scene_config = context.get("scene_config")
    if scene_config is None or not getattr(scene_config, "enabled", True):
        return []
    return build_scene_decisions(
        run.scenes,
        run.vlm_results,
        config=scene_config,
        vlm_enabled=scene_config.vlm.enabled,
    )


__all__ = ["_check_stage10"]
