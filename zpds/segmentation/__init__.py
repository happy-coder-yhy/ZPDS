"""Prepared Segment 生成：区间提议、裁剪、切分。"""

from .robot_spans import IdleCandidate, propose_edge_idle, propose_physical_spans

__all__ = ["IdleCandidate", "propose_edge_idle", "propose_physical_spans"]
