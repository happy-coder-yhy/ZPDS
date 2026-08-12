"""zpds_prepare.main 的路径助手测试。"""

from __future__ import annotations

from pathlib import Path

from zpds_prepare.main import _analysis_output_dir


def test_analysis_output_dir_under_analysis() -> None:
    output_dir = Path("output/moxian")
    assert _analysis_output_dir(output_dir, "hands") == (
        output_dir / "analysis" / "hands"
    )
    assert _analysis_output_dir(output_dir, "scene") == (
        output_dir / "analysis" / "scene"
    )
    assert _analysis_output_dir(output_dir, "privacy") == (
        output_dir / "analysis" / "privacy"
    )
