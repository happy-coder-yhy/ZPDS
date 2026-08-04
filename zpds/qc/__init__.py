"""质量检查级联（Stage 0–12）。

使用 ``register_stage`` 装饰器将各 stage 的 check 函数注册到级联调度器中。

已实现 Stages：
  - Stage 3: visual (D13 过曝 + D14 模糊)
  - Stage 5: depth (D15 深度有效性)
  - Stage 6: imu (D16 IMU 异常)
  - Stage 9: hand video cleaning report
  - Stage 11: dedup (D18 跨 Session 近重复)
"""

from zpds.qc.cascade import (
    CascadeConfig,
    CascadeDistribution,
    QCCascade,
    get_stage_checker,
    register_stage,
)


def _register_builtin_stages():
    """导入各 stage 模块触发 @register_stage 装饰器执行。"""
    from zpds.qc import stage0_privacy as _s0  # noqa: F401
    from zpds.qc import stage3_visual as _s3  # noqa: F401
    from zpds.qc import stage5_depth as _s5  # noqa: F401
    from zpds.qc import stage6_imu as _s6  # noqa: F401
    from zpds.qc import stage9_hand as _s9  # noqa: F401
    from zpds.qc import stage11_dedup as _s11  # noqa: F401


_register_builtin_stages()

__all__ = [
    "CascadeConfig",
    "CascadeDistribution",
    "QCCascade",
    "get_stage_checker",
    "register_stage",
]
