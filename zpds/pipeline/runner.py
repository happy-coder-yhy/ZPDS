"""具备幂等复用、自动重试和断点恢复能力的 Pipeline Runner。"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from zpds.observability.events import NullObserver, PipelineEvent, PipelineObserver

from .ledger import FileRunLedger, RunStatus
from .stage import (
    PipelineStage,
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
    validate_stage_contract,
)


class RunnerError(RuntimeError):
    """Runner 配置或 Stage 执行契约错误。"""


class RunnerConfigurationError(RunnerError, ValueError):
    """配置与传入 Stage 集合不匹配。"""


@dataclass(frozen=True)
class PipelineRunResult:
    run_id: str
    status: RunStatus
    stage_results: tuple[StageResult, ...]
    executed_stage_ids: tuple[int, ...]
    reused_stage_ids: tuple[int, ...]


class PipelineRunner:
    """顺序运行 Stage，并将每次状态变化交给 Run Ledger 持久化。"""

    def __init__(
        self,
        stages: Sequence[PipelineStage],
        ledger: FileRunLedger,
        *,
        clock: Callable[[], datetime] | None = None,
        sleeper: Callable[[float], None] = time.sleep,
        observer: PipelineObserver | None = None,
    ) -> None:
        descriptors = [validate_stage_contract(stage) for stage in stages]
        stage_ids = [descriptor.stage_id for descriptor in descriptors]
        if len(stage_ids) != len(set(stage_ids)):
            raise RunnerConfigurationError("stage_id values must be unique")
        self._stages = tuple(
            stage
            for _, stage in sorted(
                zip(stage_ids, stages, strict=True),
                key=lambda pair: pair[0],
            )
        )
        self._ledger = ledger
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._sleeper = sleeper
        self._observer = observer or NullObserver()

    def run(self, context: StageContext) -> PipelineRunResult:
        stages = self._enabled_stages(context)
        descriptors = tuple(stage.descriptor for stage in stages)
        self._ledger.initialize(context, descriptors)
        retry_options = context.config.section("runner")
        max_retries = int(retry_options["max_retries"])
        retry_backoff_seconds = float(retry_options["retry_backoff_seconds"])

        current_refs = context.input_refs
        results: list[StageResult] = []
        executed_stage_ids: list[int] = []
        reused_stage_ids: list[int] = []

        self._emit(
            "run_started",
            context,
            details={
                "config_hash": context.config.config_hash,
                "code_version": context.code_version,
                "stage_count": len(stages),
            },
        )
        try:
            for stage in stages:
                stage_context = StageContext(
                    run_id=context.run_id,
                    session_id=context.session_id,
                    input_refs=current_refs,
                    config=context.config,
                    code_version=context.code_version,
                )
                key = execution_key(stage.descriptor, stage_context)
                completed = self._ledger.completed_result(
                    context.run_id,
                    stage.descriptor,
                    key,
                )
                if completed is not None:
                    results.append(completed)
                    reused_stage_ids.append(stage.descriptor.stage_id)
                    self._emit(
                        "stage_reused",
                        stage_context,
                        descriptor=stage.descriptor,
                        details={"execution_key": key, "status": completed.status.value},
                    )
                    current_refs = completed.output_refs or completed.input_refs
                    continue

                executed_stage_ids.append(stage.descriptor.stage_id)
                result = self._execute_with_retries(
                    stage,
                    stage_context,
                    key,
                    max_retries=max_retries,
                    retry_backoff_seconds=retry_backoff_seconds,
                )
                results.append(result)
                if result.status == StageStatus.FAILED:
                    break
                current_refs = result.output_refs or result.input_refs

            snapshot = self._ledger.snapshot(context.run_id)
            run_result = PipelineRunResult(
                run_id=context.run_id,
                status=snapshot.status,
                stage_results=tuple(results),
                executed_stage_ids=tuple(executed_stage_ids),
                reused_stage_ids=tuple(reused_stage_ids),
            )
            self._emit(
                "run_finished",
                context,
                details={
                    "status": snapshot.status.value,
                    "executed_stage_ids": executed_stage_ids,
                    "reused_stage_ids": reused_stage_ids,
                },
            )
            return run_result
        except BaseException as error:
            self._emit(
                "run_interrupted",
                context,
                details={"error": f"{type(error).__name__}: {error}"},
            )
            raise

    def _execute_with_retries(
        self,
        stage: PipelineStage,
        context: StageContext,
        key: str,
        *,
        max_retries: int,
        retry_backoff_seconds: float,
    ) -> StageResult:
        result: StageResult | None = None
        for retry_index in range(max_retries + 1):
            started_at = self._now()
            attempt = self._ledger.begin_attempt(
                context.run_id,
                stage.descriptor,
                key,
                context.input_refs,
                started_at=started_at,
            )
            self._emit(
                "stage_attempt_started",
                context,
                descriptor=stage.descriptor,
                attempt=attempt,
                details={"execution_key": key},
            )
            try:
                result = stage.execute(context)
                _validate_result(stage.descriptor, context, result)
            # Stage 是插件边界：普通异常必须转换为可持久化的失败结果；
            # KeyboardInterrupt/SystemExit 等 BaseException 仍向外传播以支持断点恢复。
            except Exception as error:  # noqa: BLE001
                result = StageResult(
                    descriptor=stage.descriptor,
                    status=StageStatus.FAILED,
                    input_refs=context.input_refs,
                    output_refs=(),
                    config_hash=context.config.config_hash,
                    started_at=started_at,
                    finished_at=self._now(),
                    error=f"{type(error).__name__}: {error}",
                )
            except BaseException as error:
                self._emit(
                    "stage_interrupted",
                    context,
                    descriptor=stage.descriptor,
                    attempt=attempt,
                    details={"error": f"{type(error).__name__}: {error}"},
                )
                raise
            self._ledger.record_result(context.run_id, key, result)
            self._emit(
                "stage_attempt_finished",
                context,
                descriptor=stage.descriptor,
                attempt=attempt,
                details={
                    "status": result.status.value,
                    "duration_seconds": result.duration_seconds,
                    "error": result.error,
                },
            )
            if result.status != StageStatus.FAILED:
                return result
            if retry_index < max_retries:
                self._emit(
                    "stage_retry_scheduled",
                    context,
                    descriptor=stage.descriptor,
                    attempt=attempt,
                    details={"backoff_seconds": retry_backoff_seconds},
                )
                if retry_backoff_seconds > 0:
                    self._sleeper(retry_backoff_seconds)
        if result is None:
            raise AssertionError("retry loop did not execute")
        return result

    def _enabled_stages(self, context: StageContext) -> tuple[PipelineStage, ...]:
        configured = context.config.section("pipeline")["stages"]
        enabled_ids = {int(stage_id) for stage_id in configured}
        stages = tuple(
            stage for stage in self._stages if stage.descriptor.stage_id in enabled_ids
        )
        if not stages:
            raise RunnerConfigurationError("no supplied stages are enabled by pipeline.stages")
        return stages

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None or value.utcoffset() is None:
            raise RunnerError("runner clock must return a timezone-aware datetime")
        return value

    def _emit(
        self,
        event: str,
        context: StageContext,
        *,
        descriptor: StageDescriptor | None = None,
        attempt: int | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self._observer.emit(
            PipelineEvent(
                event=event,
                timestamp=self._now(),
                run_id=context.run_id,
                session_id=context.session_id,
                stage_id=descriptor.stage_id if descriptor is not None else None,
                stage_name=descriptor.name if descriptor is not None else None,
                attempt=attempt,
                details=details or {},
            )
        )


def execution_key(descriptor: StageDescriptor, context: StageContext) -> str:
    """计算不包含 run_id 的确定性执行键，支持同一 run 的断点复用。"""

    payload: dict[str, Any] = {
        "stage": {
            "stage_id": descriptor.stage_id,
            "name": descriptor.name,
            "version": descriptor.version,
        },
        "session_id": context.session_id,
        "input_refs": list(context.input_refs),
        "config_hash": context.config.config_hash,
        "code_version": context.code_version,
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(canonical).hexdigest()}"


def _validate_result(
    descriptor: StageDescriptor,
    context: StageContext,
    result: StageResult,
) -> None:
    if not isinstance(result, StageResult):
        raise TypeError("stage.execute() must return StageResult")
    if result.descriptor != descriptor:
        raise RunnerError("StageResult descriptor differs from stage descriptor")
    if result.input_refs != context.input_refs:
        raise RunnerError("StageResult input_refs differ from StageContext input_refs")
    if result.config_hash != context.config.config_hash:
        raise RunnerError("StageResult config_hash differs from loaded config")
