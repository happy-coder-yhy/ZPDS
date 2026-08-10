"""EPIC 链路接入 QCCascade 的配置测试。

覆盖：stage 剔除逻辑（stage0 恒剔除、stage3 由 skip_pixel_qc 控制）、
cascade 可链式调用。
"""

from __future__ import annotations

from zpds.qc.cascade import QCCascade

from scripts.batch_prepare_epic import _configure_epic_cascade


def _cascade() -> QCCascade:
    return QCCascade.from_profile("epic")


def test_stage0_always_removed() -> None:
    """EPIC 不跑脱敏（privacy unavailable）→ stage0 恒剔除，避免假 quarantine。"""
    c = _configure_epic_cascade(_cascade(), skip_pixel_qc=True)
    assert 0 not in c.config.enabled_stages


def test_stage3_removed_when_pixel_qc_skipped() -> None:
    """skip_pixel_qc=True（默认）→ 像素级 stage3 剔除。"""
    c = _configure_epic_cascade(_cascade(), skip_pixel_qc=True)
    assert 3 not in c.config.enabled_stages


def test_stage3_kept_when_pixel_qc_enabled() -> None:
    """skip_pixel_qc=False → 保留 stage3（过曝/模糊检测）。"""
    c = _configure_epic_cascade(_cascade(), skip_pixel_qc=False)
    assert 3 in c.config.enabled_stages
    assert 0 not in c.config.enabled_stages


def test_no_data_stages_removed() -> None:
    """EPIC 无深度/IMU/机器人/标定/音频 → stage5/6/7/8/12 剔除，避免空输入噪声决策。"""
    c = _configure_epic_cascade(_cascade(), skip_pixel_qc=True)
    for s in (0, 3, 5, 6, 7, 8, 12):
        assert s not in c.config.enabled_stages, f"stage {s} 应被剔除"


def test_other_stages_untouched() -> None:
    """去重/场景等 stage 不受影响。"""
    c = _configure_epic_cascade(_cascade(), skip_pixel_qc=True)
    assert 11 in c.config.enabled_stages  # 去重
    assert 10 in c.config.enabled_stages  # 场景
    assert 9 in c.config.enabled_stages   # 手部（无报告时为空，无害）
    assert 1 in c.config.enabled_stages   # 时间戳


def test_returns_same_instance() -> None:
    c = _cascade()
    assert _configure_epic_cascade(c, skip_pixel_qc=True) is c
