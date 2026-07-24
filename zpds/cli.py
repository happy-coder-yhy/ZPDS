"""ZPDS 平台基础命令行入口。"""

import argparse
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import TextIO

from zpds import __version__
from zpds.adapters import create_adapter, list_adapter_profiles
from zpds.config import ConfigError, PipelineConfigLoader
from zpds.observability import JsonLinesObserver, build_run_metrics, persist_run_metrics
from zpds.pipeline import (
    FileRunLedger,
    LedgerError,
    PipelineRunner,
    PipelineStage,
    RunnerError,
    RunStatus,
    StageContext,
    validate_stage_contract,
)
from zpds.prepared import GuidaBasicCleaner, PreparedValidator
from zpds.qc.adapter_stage_common import inventory_to_dict, report_to_dict
from zpds.qc.stage0_registry import InventoryStage
from zpds.qc.stage1_structure import StructureStage
from zpds.qc.stage2_time import TimeStage
from zpds.storage import LocalStorage, StorageError

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GUIDA_THRESHOLDS = (
    REPOSITORY_ROOT / "configs" / "qc_thresholds" / "guida_ego.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="zpds",
        description="ZPDS 数据标准化平台基础 CLI",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    config = commands.add_parser("config", help="Pipeline 配置操作")
    config_commands = config.add_subparsers(dest="config_command", required=True)
    config_validate = config_commands.add_parser("validate", help="校验配置并输出哈希")
    config_validate.add_argument("path", type=Path)
    config_validate.set_defaults(handler=_handle_config_validate)

    source = commands.add_parser("source", help="只读探测或校验真实数据源")
    source_commands = source.add_subparsers(dest="source_command", required=True)
    for name, handler in (
        ("inspect", _handle_source_inspect),
        ("validate", _handle_source_validate),
        ("scan", _handle_source_scan),
    ):
        source_command = source_commands.add_parser(name)
        source_command.add_argument("--profile", choices=list_adapter_profiles(), required=True)
        source_command.add_argument("--raw-root", type=Path, required=True)
        source_command.add_argument("--input-ref", required=True)
        source_command.set_defaults(handler=handler)

    run = commands.add_parser("run", help="执行或查询 Pipeline Run")
    run_commands = run.add_subparsers(dest="run_command", required=True)

    execute = run_commands.add_parser("execute", help="运行显式指定的可信 Stage 插件")
    execute.add_argument("--config", type=Path, required=True)
    execute.add_argument("--raw-root", type=Path, required=True)
    execute.add_argument("--artifact-root", type=Path, required=True)
    execute.add_argument("--run-id", required=True)
    execute.add_argument("--session-id", required=True)
    execute.add_argument("--input-ref", action="append", required=True)
    execute.add_argument("--code-version", required=True)
    execute.add_argument(
        "--stage",
        action="append",
        required=True,
        help="可信插件的 module:attribute；attribute 可为 Stage 实例或无参工厂",
    )
    execute.add_argument(
        "--log-file",
        type=Path,
        help="可选 JSONL 日志路径；默认写到 artifact run 目录",
    )
    execute.set_defaults(handler=_handle_run_execute)

    source_run = run_commands.add_parser(
        "source",
        help="使用内置 Adapter Stage 0～2 运行真实数据源",
    )
    source_run.add_argument("--profile", choices=list_adapter_profiles(), required=True)
    source_run.add_argument("--config", type=Path, required=True)
    source_run.add_argument("--raw-root", type=Path, required=True)
    source_run.add_argument("--artifact-root", type=Path, required=True)
    source_run.add_argument("--run-id", required=True)
    source_run.add_argument("--session-id", required=True)
    source_run.add_argument("--input-ref", action="append", required=True)
    source_run.add_argument("--code-version", required=True)
    source_run.add_argument("--log-file", type=Path)
    source_run.set_defaults(handler=_handle_source_run)

    status = run_commands.add_parser("status", help="读取已有 Run Ledger 和指标")
    status.add_argument("--artifact-root", type=Path, required=True)
    status.add_argument("--run-id", required=True)
    status.set_defaults(handler=_handle_run_status)

    clean = commands.add_parser("clean", help="执行基础清洗纵向闭环")
    clean_commands = clean.add_subparsers(dest="clean_command", required=True)
    clean_guida = clean_commands.add_parser(
        "guida",
        help="Guida 时钟恢复、硬质量检查、切分和 Prepared 写出",
    )
    clean_guida.add_argument("--config", type=Path, required=True)
    clean_guida.add_argument(
        "--thresholds",
        type=Path,
        default=DEFAULT_GUIDA_THRESHOLDS,
    )
    clean_guida.add_argument("--raw-root", type=Path, required=True)
    clean_guida.add_argument("--input-ref", required=True)
    clean_guida.add_argument("--output-root", type=Path, required=True)
    clean_guida.add_argument("--prep-revision", default="r0001")
    clean_guida.add_argument("--code-version", required=True)
    clean_guida.set_defaults(handler=_handle_clean_guida)

    prepared = commands.add_parser("prepared", help="Prepared Segment 操作")
    prepared_commands = prepared.add_subparsers(
        dest="prepared_command",
        required=True,
    )
    prepared_validate = prepared_commands.add_parser(
        "validate",
        help="回读校验 Schema、引用、时间和可选 Raw hash",
    )
    prepared_validate.add_argument("--segment-dir", type=Path, required=True)
    prepared_validate.add_argument("--raw-root", type=Path)
    prepared_validate.set_defaults(handler=_handle_prepared_validate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except KeyboardInterrupt:
        _write_json(
            sys.stderr,
            {"status": "interrupted", "error": "KeyboardInterrupt"},
        )
        return 130
    except (
        AttributeError,
        ConfigError,
        ImportError,
        LedgerError,
        OSError,
        RunnerError,
        StorageError,
        TypeError,
        ValueError,
    ) as error:
        _write_json(
            sys.stderr,
            {
                "status": "error",
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        return 2


def _handle_config_validate(args: argparse.Namespace) -> int:
    config = PipelineConfigLoader().load(args.path)
    _write_json(
        sys.stdout,
        {
            "status": "valid",
            "source": str(config.source),
            "version": config.version,
            "config_hash": config.config_hash,
        },
    )
    return 0


def _handle_source_inspect(args: argparse.Namespace) -> int:
    storage = LocalStorage(args.raw_root, args.raw_root)
    session_path = storage.raw_path(args.input_ref)
    inventory = create_adapter(args.profile).inspect(str(session_path))
    _write_json(
        sys.stdout,
        {
            "status": "inspected",
            "profile": args.profile,
            "inventory": inventory_to_dict(inventory),
        },
    )
    return 0


def _handle_source_validate(args: argparse.Namespace) -> int:
    storage = LocalStorage(args.raw_root, args.raw_root)
    session_path = storage.raw_path(args.input_ref)
    report = create_adapter(args.profile).validate(str(session_path))
    _write_json(
        sys.stdout,
        {
            "status": "passed" if report.passed else "failed",
            "profile": args.profile,
            "report": report_to_dict(report),
        },
    )
    return 0 if report.passed else 1


def _handle_source_scan(args: argparse.Namespace) -> int:
    storage = LocalStorage(args.raw_root, args.raw_root)
    session_path = storage.raw_path(args.input_ref)
    report = create_adapter(args.profile).scan(str(session_path))
    _write_json(
        sys.stdout,
        {
            "status": "passed" if report.passed else "failed",
            "profile": args.profile,
            "report": report_to_dict(report),
        },
    )
    return 0 if report.passed else 1


def _handle_run_execute(args: argparse.Namespace) -> int:
    stages = tuple(_load_stage(specification) for specification in args.stage)
    return _run_stages(args, stages)


def _handle_source_run(args: argparse.Namespace) -> int:
    storage = LocalStorage(args.raw_root, args.artifact_root)
    adapter = create_adapter(args.profile)
    stages = (
        InventoryStage(adapter, storage),
        StructureStage(adapter, storage),
        TimeStage(adapter, storage),
    )
    return _run_stages(args, stages, storage=storage)


def _run_stages(
    args: argparse.Namespace,
    stages: tuple[PipelineStage, ...],
    *,
    storage: LocalStorage | None = None,
) -> int:
    config = PipelineConfigLoader().load(args.config)
    storage = storage or LocalStorage(args.raw_root, args.artifact_root)
    ledger = FileRunLedger(storage)
    context = StageContext(
        run_id=args.run_id,
        session_id=args.session_id,
        input_refs=tuple(args.input_ref),
        config=config,
        code_version=args.code_version,
    )
    log_path = args.log_file or storage.artifact_path(
        f"artifact://runs/{context.run_id}/events.jsonl"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_context = log_path.open("a", encoding="utf-8")
    result = None
    try:
        with log_context as log_stream:
            runner = PipelineRunner(
                stages,
                ledger,
                observer=JsonLinesObserver(log_stream),
            )
            result = runner.run(context)
    finally:
        reference = f"artifact://runs/{context.run_id}/ledger.json"
        if storage.exists(reference):
            persist_run_metrics(storage, ledger.snapshot(context.run_id))
    if result is None:
        raise AssertionError("runner did not return a result")
    metrics_reference = f"artifact://runs/{context.run_id}/metrics.json"
    _write_json(
        sys.stdout,
        {
            "run_id": result.run_id,
            "status": result.status.value,
            "executed_stage_ids": list(result.executed_stage_ids),
            "reused_stage_ids": list(result.reused_stage_ids),
            "ledger": str(
                storage.artifact_path(
                    f"artifact://runs/{context.run_id}/ledger.json"
                )
            ),
            "events": str(log_path),
            "metrics": str(storage.artifact_path(metrics_reference)),
        },
    )
    return 0 if result.status is RunStatus.SUCCEEDED else 1


def _handle_run_status(args: argparse.Namespace) -> int:
    storage = LocalStorage(None, args.artifact_root)
    snapshot = FileRunLedger(storage).snapshot(args.run_id)
    metrics_reference = f"artifact://runs/{snapshot.run_id}/metrics.json"
    metrics = (
        storage.read_json(metrics_reference)
        if storage.exists(metrics_reference)
        else build_run_metrics(snapshot)
    )
    _write_json(
        sys.stdout,
        {
            "run_id": snapshot.run_id,
            "session_id": snapshot.session_id,
            "status": snapshot.status.value,
            "config_hash": snapshot.config_hash,
            "code_version": snapshot.code_version,
            "stages": [
                {
                    "stage_id": entry.descriptor.stage_id,
                    "name": entry.descriptor.name,
                    "version": entry.descriptor.version,
                    "status": entry.status.value,
                    "attempts": entry.attempts,
                    "last_error": entry.last_error,
                }
                for entry in snapshot.stages
            ],
            "metrics": metrics,
        },
    )
    return 0


def _handle_clean_guida(args: argparse.Namespace) -> int:
    storage = LocalStorage(args.raw_root, args.output_root)
    session_path = storage.raw_path(args.input_ref)
    config = PipelineConfigLoader().load(args.config)
    cleaner = GuidaBasicCleaner(
        pipeline_config=config,
        thresholds_path=args.thresholds,
        code_version=args.code_version,
        config_uri=_logical_config_uri(args.config),
    )
    result = cleaner.clean(
        session_path,
        args.output_root,
        raw_session_uri=args.input_ref,
        prep_revision=args.prep_revision,
    )
    _write_json(
        sys.stdout,
        {
            "status": "completed",
            "profile": "guida_ego",
            "revision_dir": str(result.revision_dir),
            "segment_ids": list(result.segment_ids),
            "source_frame_count": result.source_frame_count,
            "imu_sample_count": result.imu_sample_count,
            "issues": [issue.to_report() for issue in result.issues],
            "removed_spans": [
                {
                    "start_ns": span.start_ns,
                    "end_ns": span.end_ns,
                    "reason_code": span.code,
                    "evidence_uri": span.evidence_uri,
                }
                for span in result.removed_spans
            ],
        },
    )
    return 0


def _handle_prepared_validate(args: argparse.Namespace) -> int:
    errors = PreparedValidator().validate(
        str(args.segment_dir),
        raw_root=args.raw_root,
    )
    _write_json(
        sys.stdout,
        {
            "status": "valid" if not errors else "invalid",
            "segment_dir": str(args.segment_dir.resolve()),
            "raw_hashes_checked": args.raw_root is not None,
            "errors": errors,
        },
    )
    return 0 if not errors else 1


def _logical_config_uri(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPOSITORY_ROOT).as_posix()
    except ValueError:
        return f"runtime-configs/{resolved.name}"


def _load_stage(specification: str) -> PipelineStage:
    module_name, separator, attribute_name = specification.partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError("stage must use module:attribute syntax")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    try:
        validate_stage_contract(candidate)
    except TypeError:
        if not callable(candidate):
            raise TypeError(
                f"Stage plugin {specification!r} is not a Stage or callable factory"
            ) from None
        candidate = candidate()
        validate_stage_contract(candidate)
    return candidate


def _write_json(stream: TextIO, value: dict[str, object]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
    stream.flush()


if __name__ == "__main__":
    raise SystemExit(main())
