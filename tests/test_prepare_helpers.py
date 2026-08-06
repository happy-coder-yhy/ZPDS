"""zpds_prepare.main 的路径助手测试。"""

from __future__ import annotations

from pathlib import Path

from zpds_prepare.main import _hands_output_dir


def test_hands_output_dir_under_first_segment() -> None:
    output_dir = Path("output/moxian")
    assert _hands_output_dir(output_dir) == (
        output_dir / "prepared_segments" / "r0001" / "seg_000001" / "hands"
    )
