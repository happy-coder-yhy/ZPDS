"""Tests for importing Prepared annotation streams into an Experience."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.import_existing_annotations import main as import_annotations_main
from zpds.annotation.importer import ANNOTATION_MANIFEST_KEY, import_segment_annotations


def _write_parquet(path: Path, values: list[int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"timestamp_ns": values, "value": values}), path)


def _write_segment(segment_dir: Path, streams: list[dict]) -> None:
    segment_dir.mkdir(parents=True, exist_ok=True)
    (segment_dir / "segment.json").write_text(
        json.dumps(
            {
                "segment_id": "seg_000001",
                "record_revision": "r0001",
                "source_session": {"session_id": "a2d_8032"},
                "streams": streams,
            }
        ),
        encoding="utf-8",
    )


def _annotation_stream(
    stream_id: str = "review_actions",
    uri: str = "annotations/review_actions.parquet",
    **extra: object,
) -> dict:
    return {
        "stream_id": stream_id,
        "role": "annotation",
        "format": "parquet",
        "uri": uri,
        "modality": "action_label",
        "ground_truth_status": "human_reviewed",
        "origin": {
            "kind": "imported_human_annotation",
            "operation": "frame_index_to_output_frame",
        },
        **extra,
    }


def test_imports_declared_parquet_assets_and_preserves_existing_annotations(tmp_path: Path) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    _write_parquet(segment_dir / "annotations" / "review_actions.parquet", [10, 20])
    _write_parquet(segment_dir / "annotations" / "masks.parquet", [30])
    _write_segment(
        segment_dir,
        [
            _annotation_stream(),
            _annotation_stream(
                "masks",
                "annotations/masks.parquet",
                modality="instance_segmentation",
                ground_truth_status="model_generated",
            ),
        ],
    )
    experience_dir = tmp_path / "experiences" / "source_labels_v1"
    experience_dir.mkdir(parents=True)
    (experience_dir / "experience_manifest.json").write_text(
        json.dumps(
            {
                "experience_version": "source_labels_v1",
                "annotations": {"hands_v1": {"rows": 3}},
            }
        ),
        encoding="utf-8",
    )

    manifest_path = import_segment_annotations(segment_dir, experience_dir)

    assert manifest_path == str((experience_dir / "experience_manifest.json").resolve())
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    assert manifest["annotations"]["hands_v1"] == {"rows": 3}
    group = manifest["annotations"][ANNOTATION_MANIFEST_KEY]
    assert group["schema_version"] == 1
    assert group["asset_count"] == 2
    assets = {asset["annotation_id"]: asset for asset in group["assets"]}

    review = assets["a2d_8032:seg_000001:review_actions"]
    assert review["rows"] == 2
    assert review["ground_truth_status"] == "human_reviewed"
    assert review["source_stream_uri"] == "annotations/review_actions.parquet"
    assert review["uri"] == "assets/annotations/a2d_8032__seg_000001/review_actions.parquet"
    assert len(review["sha256"]) == 64
    assert review["origin"]["operation"] == "frame_index_to_output_frame"
    assert (experience_dir / review["uri"]).is_file()

    masks = assets["a2d_8032:seg_000001:masks"]
    assert masks["rows"] == 1
    assert masks["ground_truth_status"] == "model_generated"


def test_reimport_is_idempotent_but_changed_content_is_rejected(tmp_path: Path) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    parquet_path = segment_dir / "annotations" / "review_actions.parquet"
    _write_parquet(parquet_path, [10])
    _write_segment(segment_dir, [_annotation_stream()])
    experience_dir = tmp_path / "experiences" / "labels_v1"

    first_manifest = import_segment_annotations(segment_dir, experience_dir)
    assert import_segment_annotations(segment_dir, experience_dir) == first_manifest

    _write_parquet(parquet_path, [10, 20])
    with pytest.raises(ValueError, match="内容哈希不同"):
        import_segment_annotations(segment_dir, experience_dir)


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("../outside.parquet", "相对路径"),
        ("/tmp/outside.parquet", "相对路径"),
    ],
)
def test_rejects_annotation_uri_outside_segment(
    tmp_path: Path,
    uri: str,
    expected: str,
) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    _write_segment(segment_dir, [_annotation_stream(uri=uri)])

    with pytest.raises(ValueError, match=expected):
        import_segment_annotations(segment_dir, tmp_path / "experience")


def test_rejects_non_parquet_annotation_stream(tmp_path: Path) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    _write_segment(
        segment_dir,
        [_annotation_stream(uri="annotations/review_actions.json", format="json")],
    )

    with pytest.raises(ValueError, match="非 Parquet"):
        import_segment_annotations(segment_dir, tmp_path / "experience")


def test_segment_without_declared_annotations_does_not_create_experience(tmp_path: Path) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    _write_segment(segment_dir, [{"stream_id": "ego_rgb", "role": "observation"}])
    experience_dir = tmp_path / "experiences" / "labels_v1"

    assert import_segment_annotations(segment_dir, experience_dir) is None
    assert not experience_dir.exists()


def test_cli_imports_multiple_segments(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    segment_dir = tmp_path / "prepared" / "seg_000001"
    _write_parquet(segment_dir / "annotations" / "review_actions.parquet", [10])
    _write_segment(segment_dir, [_annotation_stream()])
    experience_dir = tmp_path / "experiences" / "labels_v1"

    status = import_annotations_main(
        [
            "--segment",
            str(segment_dir),
            "--experience-dir",
            str(experience_dir),
        ]
    )

    assert status == 0
    assert "Imported segments: 1" in capsys.readouterr().out
    assert (experience_dir / "experience_manifest.json").is_file()
