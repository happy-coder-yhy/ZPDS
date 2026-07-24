import json
import pickle
from pathlib import Path

from zpds.adapters import A2DAdapter, Epic100Adapter, create_adapter
from zpds.profiles.registry import get, list_all


def test_five_profiles_are_registered_with_legacy_epic_alias() -> None:
    assert list_all() == [
        "a2d_robot",
        "dunjia_ego",
        "epic100",
        "guida_ego",
        "jianzhi_umi",
    ]
    assert get("epic100_auto_annotation") is get("epic100")
    assert create_adapter("epic100_auto_annotation").profile_name == "epic100"


def test_a2d_completeness_matrix_reports_missing_camera_member(tmp_path: Path) -> None:
    (tmp_path / "meta_info.json").write_text(
        json.dumps({"episode_token": "a2d-demo"}),
        encoding="utf-8",
    )
    frame = tmp_path / "camera" / "0"
    frame.mkdir(parents=True)
    (frame / "head_color.jpg").write_bytes(b"\xff\xd8\xff")

    report = A2DAdapter().validate(str(tmp_path))

    assert report.passed
    assert report.metadata["incomplete_frames"] == 1
    assert report.issues[0].code == "a2d_camera_tuple_incomplete"


def test_epic_pickle_is_never_loaded_in_main_process(tmp_path: Path) -> None:
    (tmp_path / "annotation.pkl").write_bytes(
        pickle.dumps({"frame_ids": [1, 2, 3]}, protocol=4)
    )

    inventory = Epic100Adapter().inspect(str(tmp_path))
    report = Epic100Adapter().validate(str(tmp_path))

    assert inventory.metadata["untrusted_input"] is True
    assert report.metadata["main_process_unpickle"] is False
    assert report.passed

