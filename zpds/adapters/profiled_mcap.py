"""遁甲与 UMI 的 Profile-aware MCAP 包装。"""

from zpds.core.types import SessionInventory
from zpds.profiles.registry import get as get_profile

from .contracts import IssueLevel, ValidationIssue, ValidationReport
from .mcap import McapInspector


class ProfiledMcapAdapter(McapInspector):
    def __init__(self, profile_name: str) -> None:
        if profile_name not in {"dunjia_ego", "jianzhi_umi"}:
            raise ValueError(f"Unsupported MCAP profile: {profile_name}")
        self.profile_name = profile_name

    def inspect(self, path: str) -> SessionInventory:
        inventory = super().inspect(path)
        inventory.source_profile = self.profile_name
        inventory.metadata["preserve_log_and_publish_time"] = True
        if self.profile_name == "jianzhi_umi":
            inventory.metadata.update(
                {
                    "robot_groups": ["robot0", "robot1"],
                    "magnetic_encoder_semantics": "raw_scalar_until_confirmed",
                    "forbid_interpolation_across_vio_reset": True,
                }
            )
        return inventory

    def validate(self, path: str) -> ValidationReport:
        report = super().validate(path)
        if not report.passed:
            return report
        topics = self.topics(path)
        profile = get_profile(self.profile_name)
        if profile is None:
            raise RuntimeError(f"Profile is not registered: {self.profile_name}")
        suffixes = tuple(profile.metadata.get("required_topic_suffixes", ()))
        fragments = tuple(profile.metadata.get("required_topic_fragments", ()))
        issues = list(report.issues)
        for required in suffixes:
            if not any(topic.endswith(str(required)) for topic in topics):
                issues.append(
                    ValidationIssue(
                        code="mcap_required_topic_missing",
                        level=IssueLevel.ERROR,
                        message=f"Required topic is missing: {required}",
                        path=str(path),
                        stream_id=str(required),
                    )
                )
        for required in fragments:
            if not any(str(required) in topic for topic in topics):
                issues.append(
                    ValidationIssue(
                        code="mcap_required_topic_missing",
                        level=IssueLevel.ERROR,
                        message=f"Required topic fragment is missing: {required}",
                        path=str(path),
                        stream_id=str(required),
                    )
                )
        return ValidationReport(
            issues=tuple(issues),
            checked_assets=report.checked_assets,
            checked_records=report.checked_records,
            decoded_records=report.decoded_records,
            metadata={**report.metadata, "profile": self.profile_name, "topics": topics},
        )

    def topics(self, path: str) -> list[str]:
        return sorted(stream.topic or stream.stream_id for stream in self.read_stream_catalog(path))

    def scan(self, path: str) -> ValidationReport:
        container_report = super().scan(path)
        media_report = self.scan_embedded_media(path)
        return ValidationReport(
            issues=(*container_report.issues, *media_report.issues),
            checked_assets=1,
            checked_records=container_report.checked_records,
            decoded_records=container_report.decoded_records,
            metadata={
                **container_report.metadata,
                "embedded_media": {
                    **media_report.metadata,
                    "passed": media_report.passed,
                    "issues": [issue.code for issue in media_report.issues],
                },
            },
        )
