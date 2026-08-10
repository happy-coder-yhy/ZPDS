"""dataset.json 与 revision.json 构建/写出测试。"""

from __future__ import annotations

import json
from pathlib import Path

from segment.segment_writer import (
    build_dataset_json,
    build_revision_json,
    build_segment_json,
    package_version,
    write_dataset_json,
    write_revision_json,
)
from zpds.prepared.conventions import LENGTH_UNIT


class TestBuildDatasetJson:
    def test_minimal_fields(self) -> None:
        document = build_dataset_json(
            dataset_id="moxian",
            prep_revision="r0001",
            source_types=["ego"],
        )
        assert document["zpds_version"] == "0.1.0"
        assert "zrds_version" not in document
        assert document["dataset_id"] == "moxian"
        assert document["dataset_version"] == "0.1.0"
        assert document["default_prep_revision"] == "r0001"
        assert document["source_types"] == ["ego"]
        assert document["created_at"].endswith("Z")

    def test_experience_version_only_when_given(self) -> None:
        base = build_dataset_json(dataset_id="m", prep_revision="r0001")
        assert "default_experience_version" not in base
        with_version = build_dataset_json(
            dataset_id="m",
            prep_revision="r0001",
            default_experience_version="v0.1.0",
        )
        assert with_version["default_experience_version"] == "v0.1.0"

    def test_dataset_version_derived_from_package(self) -> None:
        document = build_dataset_json(dataset_id="m", prep_revision="r0001")
        assert document["dataset_version"] == package_version()


class TestBuildRevisionJson:
    def test_fields_and_length_unit_source(self) -> None:
        document = build_revision_json(
            prep_revision="r0001",
            pipeline_name="zpds.batch_prepare",
            pipeline_version="0.1.0",
            config_hash="sha256:abc",
            changes=["删除无效区间"],
        )
        assert document["prep_revision"] == "r0001"
        assert document["zpds_version"] == "0.1.0"
        assert document["parent_revision"] is None
        assert document["pipeline"]["name"] == "zpds.batch_prepare"
        assert document["pipeline"]["config_hash"] == "sha256:abc"
        assert document["changes"] == ["删除无效区间"]
        conventions = document["conventions"]
        assert conventions["time_unit"] == "ns"
        # 长度单位以 prepared/conventions.py 为权威来源，不静默假设。
        assert conventions["length_unit"] == LENGTH_UNIT
        assert conventions["length_unit"] == "m"
        assert conventions["length_unit_source"] == "zpds/prepared/conventions.py"

    def test_pipeline_version_derived_from_package(self) -> None:
        document = build_revision_json(
            prep_revision="r0001",
            pipeline_name="zpds.batch_prepare",
            config_hash="sha256:abc",
        )
        assert document["pipeline"]["version"] == package_version()

    def test_run_stats_included_when_provided(self) -> None:
        document = build_revision_json(
            prep_revision="r0001",
            pipeline_name="zpds.batch_prepare",
            config_hash="sha256:abc",
            run_stats={
                "profile": "dunjia",
                "segment_count": 1,
                "video_stream_ids": ["camera0", "camera1", "camera2"],
                "rgb_frames_total": 374,
            },
        )
        assert document["run_stats"]["profile"] == "dunjia"
        assert document["run_stats"]["video_stream_ids"] == [
            "camera0",
            "camera1",
            "camera2",
        ]

    def test_run_stats_omitted_when_absent(self) -> None:
        document = build_revision_json(
            prep_revision="r0001",
            pipeline_name="p",
            config_hash="sha256:abc",
        )
        assert "run_stats" not in document


class TestBuildSegmentJson:
    def test_standard_version_and_revision_field_names(self, tmp_path: Path) -> None:
        document = build_segment_json(
            dataset_path=str(tmp_path),
            span={"source_start_ns": 0, "source_end_ns": 1},
            prep_revision="r0002",
        )

        assert document["zpds_version"] == "0.1.0"
        assert document["prep_revision"] == "r0002"
        assert "zrds_version" not in document
        assert "record_revision" not in document


class TestWriteRoundTrip:
    def test_dataset_and_revision_round_trip(self, tmp_path: Path) -> None:
        dataset_doc = build_dataset_json(
            dataset_id="moxian",
            prep_revision="r0001",
        )
        dataset_path = write_dataset_json(dataset_doc, tmp_path)
        assert Path(dataset_path).name == "dataset.json"
        assert json.loads(Path(dataset_path).read_text(encoding="utf-8"))[
            "dataset_id"
        ] == "moxian"

        revision_doc = build_revision_json(
            prep_revision="r0001",
            pipeline_name="p",
            pipeline_version="1",
            config_hash="sha256:abc",
        )
        revision_path = write_revision_json(revision_doc, tmp_path / "r0001")
        assert Path(revision_path).name == "revision.json"
        loaded = json.loads(Path(revision_path).read_text(encoding="utf-8"))
        assert loaded["prep_revision"] == "r0001"
        assert loaded["pipeline"]["code_commit"]
