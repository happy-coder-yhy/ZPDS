"""EPIC-KITCHENS-100 衍生标注只读 Adapter。"""

from pathlib import Path

from zpds.core.types import (
    ClockDescriptor,
    ClockDomain,
    SessionInventory,
    SourceStream,
    StreamKind,
)

from .base import BaseAdapter
from .common import require_directory, source_asset
from .contracts import IssueLevel, ValidationIssue, ValidationReport
from .pickle import inspect_pickle, summarize_primitive_pickle


class Epic100Adapter(BaseAdapter):
    profile_name = "epic100"
    quick_validation_limit = 20

    def inspect(self, path: str) -> SessionInventory:
        root = require_directory(path)
        pickle_paths = self._pickle_paths(root)
        assets = [
            source_asset(item, root)
            for item in sorted(
                {
                    *pickle_paths,
                    *(item for pattern in ("**/*.json", "**/*.csv") for item in root.glob(pattern)),
                }
            )
        ]
        streams = [
            SourceStream(
                kind=StreamKind.ROBOT_STATE,
                stream_id="model_annotations",
                role="annotation",
                clock_id="annotation_frame_index",
                container="pickle",
                encoding="python_pickle_untrusted",
                metadata={"origin": "model_estimated"},
            )
        ]
        return SessionInventory(
            session_id=root.name,
            source_profile=self.profile_name,
            session_uri=str(root),
            assets=assets,
            streams=streams,
            clocks=self.read_clock_catalog(str(root)),
            clock_domain=ClockDomain.CUSTOM_EPOCH,
            metadata={
                "pickle_count": len(pickle_paths),
                "untrusted_input": True,
                "requires_video_hash_link": True,
            },
        )

    def validate(self, path: str) -> ValidationReport:
        root = Path(path)
        if not root.is_dir():
            return ValidationReport(
                issues=(
                    ValidationIssue(
                        code="epic_annotation_root_missing",
                        level=IssueLevel.FATAL,
                        message=f"EPIC annotation root not found: {root}",
                    ),
                )
            )
        pickle_paths = self._pickle_paths(root)
        issues: list[ValidationIssue] = []
        if not pickle_paths:
            issues.append(
                ValidationIssue(
                    code="epic_pickle_missing",
                    level=IssueLevel.ERROR,
                    message="No .pkl or .pickle annotations were found",
                    path=str(root),
                )
            )
        sampled_paths = _even_sample(pickle_paths, self.quick_validation_limit)
        issues.extend(self._inspect_pickles(sampled_paths))
        if len(sampled_paths) < len(pickle_paths):
            issues.append(
                ValidationIssue(
                    code="pickle_validation_sampled",
                    level=IssueLevel.INFO,
                    message=(
                        f"Quick validation inspected {len(sampled_paths)} of "
                        f"{len(pickle_paths)} pickle files; use scan() for all files"
                    ),
                    path=str(root),
                    details={
                        "sampled": len(sampled_paths),
                        "total": len(pickle_paths),
                    },
                )
            )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=len(sampled_paths),
            checked_records=len(sampled_paths),
            metadata={
                "main_process_unpickle": False,
                "total_pickle_count": len(pickle_paths),
                "quick_validation": True,
            },
        )

    def scan(self, path: str) -> ValidationReport:
        root = require_directory(path)
        pickle_paths = self._pickle_paths(root)
        issues: list[ValidationIssue] = []
        decoded_records = 0
        summary_types: dict[str, int] = {}
        summary_examples: list[dict[str, object]] = []
        for source in pickle_paths:
            try:
                inspection = inspect_pickle(source)
            except (OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="pickle_opcode_invalid",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(source),
                    )
                )
                continue
            if inspection.requires_isolated_review:
                issues.append(
                    ValidationIssue(
                        code="untrusted_pickle",
                        level=IssueLevel.WARN,
                        message=(
                            "Pickle contains object construction opcodes; "
                            "content parsing was blocked"
                        ),
                        path=str(source),
                        details={
                            "globals": list(inspection.global_references),
                            "unsafe_opcodes": list(inspection.unsafe_opcodes),
                        },
                    )
                )
                continue
            try:
                summary = summarize_primitive_pickle(source)
            except (OSError, TypeError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="pickle_content_invalid",
                        level=IssueLevel.ERROR,
                        message=f"Isolated primitive-only parse failed: {error}",
                        path=str(source),
                    )
                )
                continue
            decoded_records += 1
            summary_type = str(summary.get("type", "unknown"))
            summary_types[summary_type] = summary_types.get(summary_type, 0) + 1
            if len(summary_examples) < 20:
                summary_examples.append(
                    {
                        "path": source.relative_to(root).as_posix(),
                        "summary": summary,
                    }
                )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=len(pickle_paths),
            checked_records=len(pickle_paths),
            decoded_records=decoded_records,
            metadata={
                "main_process_unpickle": False,
                "isolated_content_parse": True,
                "total_pickle_count": len(pickle_paths),
                "quick_validation": False,
                "summary_types": summary_types,
                "summary_examples": summary_examples,
                "summary_examples_truncated": len(pickle_paths) > len(summary_examples),
            },
        )

    @staticmethod
    def _inspect_pickles(
        pickle_paths: tuple[Path, ...],
    ) -> list[ValidationIssue]:
        issues: list[ValidationIssue] = []
        for source in pickle_paths:
            try:
                inspection = inspect_pickle(source)
            except (OSError, ValueError) as error:
                issues.append(
                    ValidationIssue(
                        code="pickle_opcode_invalid",
                        level=IssueLevel.ERROR,
                        message=str(error),
                        path=str(source),
                    )
                )
                continue
            if inspection.requires_isolated_review:
                issues.append(
                    ValidationIssue(
                        code="untrusted_pickle",
                        level=IssueLevel.WARN,
                        message="Pickle contains object construction opcodes; main process did not load it",
                        path=str(source),
                        details={
                            "globals": list(inspection.global_references),
                            "unsafe_opcodes": list(inspection.unsafe_opcodes),
                        },
                    )
                )
        return issues

    def read_clock_catalog(self, path: str) -> list[ClockDescriptor]:
        del path
        return [
            ClockDescriptor(
                clock_id="annotation_frame_index",
                domain=ClockDomain.CUSTOM_EPOCH,
                source="derived annotation frame index",
                notes="Orphaned until original video hash/version is proven",
            )
        ]

    @staticmethod
    def _pickle_paths(root: Path) -> tuple[Path, ...]:
        return tuple(
            sorted({*root.glob("**/*.pkl"), *root.glob("**/*.pickle")})
        )


def _even_sample(paths: tuple[Path, ...], limit: int) -> tuple[Path, ...]:
    if len(paths) <= limit:
        return paths
    if limit < 2:
        return paths[:limit]
    indexes = {
        round(index * (len(paths) - 1) / (limit - 1))
        for index in range(limit)
    }
    return tuple(paths[index] for index in sorted(indexes))
