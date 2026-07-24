"""Pipeline Stage 公共契约。"""

from .ledger import (
    FileRunLedger,
    LedgerConflictError,
    LedgerError,
    RunLedgerSnapshot,
    RunStatus,
    StageLedgerEntry,
)
from .runner import (
    PipelineRunner,
    PipelineRunResult,
    RunnerConfigurationError,
    RunnerError,
    execution_key,
)
from .stage import (
    PipelineStage,
    StageContext,
    StageDescriptor,
    StageResult,
    StageStatus,
    validate_stage_contract,
)

__all__ = [
    "FileRunLedger",
    "LedgerConflictError",
    "LedgerError",
    "PipelineRunResult",
    "PipelineRunner",
    "PipelineStage",
    "RunLedgerSnapshot",
    "RunStatus",
    "RunnerConfigurationError",
    "RunnerError",
    "StageContext",
    "StageDescriptor",
    "StageLedgerEntry",
    "StageResult",
    "StageStatus",
    "execution_key",
    "validate_stage_contract",
]
