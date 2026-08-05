"""QCCascade — 质量检查级联调度器。

按 Stage 0–12 依次执行质量检查，支持：
- 按 stage 依次调度
- FATAL 停止条件
- Decision / Severity / ReasonCode 聚合
- overall_pass 计算
- pass / quarantine / reject 分布统计
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from zpds.core.decisions import Decision, ReasonCode, Severity
from zpds.core.quality import QualityMetric, QualityReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

# 每个 stage 的检查函数签名: (context: dict) -> list[Decision]
StageChecker = Callable[[dict], list[Decision]]

_STAGE_REGISTRY: dict[int, StageChecker] = {}


def register_stage(stage: int):
    """装饰器：将函数注册为指定 stage 的检查器。"""

    def decorator(fn: StageChecker):
        _STAGE_REGISTRY[stage] = fn
        return fn

    return decorator


def get_stage_checker(stage: int) -> StageChecker | None:
    """获取已注册的 stage 检查器。"""
    return _STAGE_REGISTRY.get(stage)


# ---------------------------------------------------------------------------
# Cascade configuration
# ---------------------------------------------------------------------------

# 默认级联停止条件：FATAL 即停
DEFAULT_STOP_ON_SEVERITY: set[Severity] = {Severity.FATAL}

# 默认的 quarantine 触发 severity
DEFAULT_QUARANTINE_SEVERITIES: set[Severity] = {Severity.ERROR, Severity.WARN}


@dataclass
class CascadeConfig:
    """级联调度配置。"""

    enabled_stages: list[int] = field(
        default_factory=lambda: list(range(13))
    )
    stop_on_severity: set[Severity] = field(
        default_factory=lambda: DEFAULT_STOP_ON_SEVERITY.copy()
    )
    quarantine_severities: set[Severity] = field(
        default_factory=lambda: DEFAULT_QUARANTINE_SEVERITIES.copy()
    )
    config_overrides: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Distribution counters
# ---------------------------------------------------------------------------


@dataclass
class CascadeDistribution:
    """级联清洗结果分布统计。"""

    total_decisions: int = 0
    pass_count: int = 0
    quarantine_count: int = 0
    reject_count: int = 0

    severity_counts: dict[str, int] = field(default_factory=dict)
    reason_counts: dict[str, int] = field(default_factory=dict)
    stage_counts: dict[int, int] = field(default_factory=dict)

    def record(self, decision: Decision) -> None:
        """将一条 Decision 计入分布。"""
        self.total_decisions += 1
        sev = decision.severity.value if isinstance(decision.severity, Severity) else str(decision.severity)
        self.severity_counts[sev] = self.severity_counts.get(sev, 0) + 1
        reason = decision.reason.value if isinstance(decision.reason, ReasonCode) else str(decision.reason)
        self.reason_counts[reason] = self.reason_counts.get(reason, 0) + 1
        self.stage_counts[decision.stage] = self.stage_counts.get(decision.stage, 0) + 1

    def to_dict(self) -> dict:
        return {
            "total_decisions": self.total_decisions,
            "pass_count": self.pass_count,
            "quarantine_count": self.quarantine_count,
            "reject_count": self.reject_count,
            "by_severity": dict(self.severity_counts),
            "by_reason": dict(self.reason_counts),
            "by_stage": {str(k): v for k, v in self.stage_counts.items()},
        }


# ---------------------------------------------------------------------------
# QCCascade
# ---------------------------------------------------------------------------


class QCCascade:
    """按 stage 0–12 依次执行质量检查。

    Usage::

        cascade = QCCascade.from_profile("guida_ego")
        report = cascade.run(context={"session_path": "/data/session"})
        print(report.overall_pass)
    """

    def __init__(
        self,
        config: CascadeConfig | None = None,
        stage_configs: dict[int, dict] | None = None,
    ):
        self.config = config or CascadeConfig()
        self._stage_configs = stage_configs or {}
        self._distribution = CascadeDistribution()

    # ---- factories --------------------------------------------------------

    @classmethod
    def from_profile(cls, profile: str) -> QCCascade:
        """从 profile 名称加载阈值配置并构建级联。

        依次查找 `configs/qc_thresholds/{profile}.yaml`。
        """
        import zpds

        pkg_root = Path(zpds.__file__).parent.parent
        config_path = pkg_root / "configs" / "qc_thresholds" / f"{profile}.yaml"
        if not config_path.exists():
            logger.warning("QC threshold config not found: %s, using defaults", config_path)
            return cls()

        with open(config_path, "r", encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}

        # 解析各 stage 配置
        stage_configs: dict[int, dict] = {}
        stage_map = {
            "stage0_registry": 0,
            "stage1_structure": 1,
            "stage2_time": 2,
            "stage3_visual": 3,
            "stage4_video_timing": 4,
            "stage5_depth": 5,
            "stage6_imu": 6,
            "stage7_robot": 7,
            "stage8_calibration": 8,
            "stage9_hand": 9,
            "stage10_semantic": 10,
            "stage11_dedup": 11,
            "stage12_delivery": 12,
        }
        for key, stage_num in stage_map.items():
            if key in raw:
                stage_configs[stage_num] = raw[key]

        cascade_cfg = raw.get("cascade", {})
        config = CascadeConfig(
            enabled_stages=cascade_cfg.get("enabled_stages", list(range(13))),
            config_overrides=stage_configs,
        )

        return cls(config=config, stage_configs=stage_configs)

    # ---- run --------------------------------------------------------------

    def run(self, context: dict) -> QualityReport:
        """执行全级联检查，返回 QualityReport。

        Parameters
        ----------
        context : dict
            至少包含 ``session_id`` 键。各 stage 可从中读取所需字段。

        Returns
        -------
        QualityReport
        """
        session_id = context.get("session_id", "unknown")
        segment_id = context.get("segment_id", "")
        report = QualityReport(session_id=session_id, segment_id=segment_id)
        self._distribution = CascadeDistribution()

        for stage in self.config.enabled_stages:
            checker = get_stage_checker(stage)
            if checker is None:
                logger.debug("Stage %d: no checker registered, skipping", stage)
                continue

            stage_cfg = self._stage_configs.get(stage, {})
            ctx = {**context, "stage_config": stage_cfg, "stage": stage}

            logger.info(
                "Stage %d (%s): 开始检查...", stage, checker.__name__
            )
            try:
                decisions = checker(ctx)
            except Exception:
                logger.exception("Stage %d checker raised exception", stage)
                decisions = [
                    Decision(
                        stage=stage,
                        reason=ReasonCode.DELIVERY_CHECK_FAIL,
                        severity=Severity.ERROR,
                        message=f"Stage {stage} checker raised an unhandled exception",
                    )
                ]

            if not isinstance(decisions, list):
                decisions = []

            logger.info(
                "Stage %d (%s): 完成, %d decisions",
                stage,
                checker.__name__,
                len(decisions),
            )

            for d in decisions:
                report.decisions.append(d)
                self._distribution.record(d)

            # 检查是否需要提前终止
            if self._should_stop(decisions):
                logger.warning(
                    "Cascade stopped at stage %d due to stop condition (severity in %s)",
                    stage,
                    self.config.stop_on_severity,
                )
                break

        # 聚合 metric 与 overall_pass
        report.metrics = self._build_metrics(report.decisions)
        report.overall_pass = self._compute_overall_pass(report.decisions)

        # 计算 pass / quarantine / reject
        self._compute_disposition(report)

        return report

    # ---- internal ---------------------------------------------------------

    def _should_stop(self, decisions: list[Decision]) -> bool:
        """判断是否因 FATAL 等严重等级提前终止级联。"""
        for d in decisions:
            if d.severity in self.config.stop_on_severity:
                return True
        return False

    def _build_metrics(self, decisions: list[Decision]) -> list[QualityMetric]:
        """从 decisions 中提取 / 构造 QualityMetric 列表。"""
        metrics: list[QualityMetric] = []
        # 从 decision.detail 中提取数值型指标
        for d in decisions:
            detail = d.detail or {}
            for key, val in detail.items():
                if isinstance(val, (int, float)) and not isinstance(val, bool):
                    threshold = detail.get(f"{key}_threshold", 0.8)
                    metrics.append(
                        QualityMetric(
                            name=f"{d.reason.value}_{key}",
                            value=float(val),
                            threshold=float(threshold),
                        )
                    )
        return metrics

    @staticmethod
    def _compute_overall_pass(decisions: list[Decision]) -> bool:
        """从 decisions 计算 overall_pass。

        FATAL 或 ERROR -> 不通过。
        """
        for d in decisions:
            if d.severity in (Severity.FATAL, Severity.ERROR):
                return False
        return True

    def _compute_disposition(self, report: QualityReport) -> None:
        """根据 decisions 的 severity 分布计算 pass / quarantine / reject 计数。

        - FATAL -> reject
        - ERROR / WARN -> quarantine（软阈值，不自动 reject）
        - 无异常 -> pass
        """
        has_fatal = False
        has_issue = False
        for d in report.decisions:
            if d.severity == Severity.FATAL:
                has_fatal = True
            if d.severity in (Severity.ERROR, Severity.WARN):
                has_issue = True

        if has_fatal:
            self._distribution.reject_count = 1
        elif has_issue:
            self._distribution.quarantine_count = 1
        else:
            self._distribution.pass_count = 1

    @property
    def distribution(self) -> CascadeDistribution:
        return self._distribution

    def distribution_dict(self) -> dict:
        """返回清洗分布摘要（适合序列化到 JSON）。"""
        return self._distribution.to_dict()
