"""Stage 0: 文件清单 / hash / license 校验。"""

from dataclasses import replace
from datetime import datetime, timezone

from zpds.adapters import BaseAdapter
from zpds.adapters.common import sha256_file
from zpds.core.quality import MetricDirection, QualityMetric
from zpds.pipeline import StageContext, StageDescriptor, StageResult, StageStatus
from zpds.storage import LocalStorage

from .adapter_stage_common import (
    inventory_to_dict,
    raw_session_path,
    retained_outputs,
    validate_source_inventory,
)


class InventoryStage:
    descriptor = StageDescriptor(0, "source_inventory", "0.1.0")

    def __init__(self, adapter: BaseAdapter, storage: LocalStorage) -> None:
        self._adapter = adapter
        self._storage = storage

    def execute(self, context: StageContext) -> StageResult:
        started_at = datetime.now(timezone.utc)
        _, session_path = raw_session_path(self._storage, context)
        inventory = self._adapter.inspect(str(session_path))
        root = session_path if session_path.is_dir() else session_path.parent
        inventory.assets = [
            replace(
                asset,
                sha256=sha256_file(root / asset.relative_path),
            )
            for asset in inventory.assets
        ]
        report_reference = f"artifact://runs/{context.run_id}/stage-0/inventory.json"
        serialized_inventory = inventory_to_dict(inventory)
        validate_source_inventory(serialized_inventory)
        value = {
            "zpds_version": "0.1.0",
            "run_id": context.run_id,
            "stage": 0,
            "config_hash": context.config.config_hash,
            "code_version": context.code_version,
            "inventory": serialized_inventory,
            "license_status": "not_evaluated",
            "privacy_status": "not_evaluated",
        }
        self._storage.atomic_write_json(report_reference, value)
        total_bytes = sum(asset.size_bytes for asset in inventory.assets)
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
                    name="source_asset_count",
                    value=float(len(inventory.assets)),
                    unit="files",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=1.0,
                ),
                QualityMetric(
                    name="source_bytes",
                    value=float(total_bytes),
                    unit="bytes",
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    threshold=1.0,
                ),
            ),
        )
