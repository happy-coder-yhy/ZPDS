import hashlib
from pathlib import Path

import pytest

from zpds.storage import ArtifactExistsError, InvalidStorageReference, LocalStorage


def _storage(tmp_path: Path) -> LocalStorage:
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    return LocalStorage(raw_root, tmp_path / "artifacts")


def test_raw_read_and_hash_are_streaming_and_namespaced(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    source = storage.raw_root / "session" / "index.jsonl"
    source.parent.mkdir()
    source.write_bytes(b"one\ntwo\n")

    with storage.open_read("raw://session/index.jsonl") as file:
        assert file.read() == b"one\ntwo\n"
    expected_hash = hashlib.sha256(b"one\ntwo\n").hexdigest()
    assert storage.sha256("raw://session/index.jsonl") == f"sha256:{expected_hash}"


@pytest.mark.parametrize(
    "reference",
    [
        "raw://../secret",
        "raw:///absolute",
        "raw://session\\file",
        "raw://C:/secret",
        "missing-scheme",
    ],
)
def test_reference_rejects_escape_and_ambiguous_paths(
    tmp_path: Path,
    reference: str,
) -> None:
    storage = _storage(tmp_path)

    with pytest.raises(InvalidStorageReference):
        storage.exists(reference)


def test_atomic_file_validation_failure_leaves_no_target_or_temp(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)

    def reject(_: Path) -> None:
        raise ValueError("invalid artifact")

    with pytest.raises(ValueError, match="invalid artifact"):
        storage.atomic_write_file(
            "artifact://reports/stage0.json",
            lambda path: path.write_text("partial", encoding="utf-8"),
            validator=reject,
        )

    report_dir = storage.artifact_root / "reports"
    assert not storage.exists("artifact://reports/stage0.json")
    assert list(report_dir.iterdir()) == []


def test_atomic_file_can_replace_previous_complete_file(tmp_path: Path) -> None:
    storage = _storage(tmp_path)
    reference = "artifact://reports/stage0.json"

    first = storage.atomic_write_json(reference, {"revision": 1})
    second = storage.atomic_write_json(reference, {"revision": 2})

    assert first.sha256 != second.sha256
    assert storage.read_json(reference) == {"revision": 2}


def test_atomic_directory_validates_before_publish_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    storage = _storage(tmp_path)
    reference = "artifact://prepared/rev001/segment001"

    def write(directory: Path) -> None:
        (directory / "segment.json").write_text("{}", encoding="utf-8")

    def validate(directory: Path) -> None:
        assert (directory / "segment.json").is_file()

    published = storage.atomic_write_directory(reference, write, validator=validate)

    assert published.is_dir()
    assert (published / "segment.json").is_file()
    with pytest.raises(ArtifactExistsError):
        storage.atomic_write_directory(reference, write)
