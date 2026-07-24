"""Stage 1: 容器头 / 索引 / schema 校验。"""

from datetime import datetime, timezone

from zpds.adapters import BaseAdapter
from zpds.core.quality import MetricDirection, QualityMetric
from zpds.pipeline import StageContext, StageDescriptor, StageResult, StageStatus
from zpds.storage import LocalStorage

from .adapter_stage_common import (
    decisions_from_report,
    raw_session_path,
    report_to_dict,
    retained_outputs,
    validate_validation_report,
)


class StructureStage:
    descriptor = StageDescriptor(1, "source_structure", "0.1.0")

    def __init__(self, adapter: BaseAdapter, storage: LocalStorage) -> None:
        self._adapter = adapter
        self._storage = storage

    def execute(self, context: StageContext) -> StageResult:
        started_at = datetime.now(timezone.utc)
        _, session_path = raw_session_path(self._storage, context)
        report = self._adapter.validate(str(session_path))
        serialized_report = report_to_dict(report)
        validate_validation_report(serialized_report)
        report_reference = f"artifact://runs/{context.run_id}/stage-1/structure.json"
        self._storage.atomic_write_json(
            report_reference,
            {
                "zpds_version": "0.1.0",
                "run_id": context.run_id,
                "stage": 1,
                "config_hash": context.config.config_hash,
                "report": serialized_report,
            },
        )
        finished_at = datetime.now(timezone.utc)
        return StageResult(
            descriptor=self.descriptor,
            status=StageStatus.SUCCEEDED,
            input_refs=context.input_refs,
            output_refs=retained_outputs(context, report_reference),
            config_hash=context.config.config_hash,
            started_at=started_at,
            finished_at=finished_at,
            metrics=(
                QualityMetric(
                    name="structure_issue_count",
                    value=float(len(report.issues)),
                    unit="issues",
                    direction=MetricDirection.LOWER_IS_BETTER,
                    threshold=0.0,
                ),
            ),
            decisions=decisions_from_report(
                report,
                stage_id=1,
                evidence_uri=report_reference,
            ),
        )
