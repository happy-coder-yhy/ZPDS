"""单位转换工具。"""


def mm_to_m(mm: float) -> float:
    return mm / 1000.0


def m_to_mm(m: float) -> float:
    return m * 1000.0


def deg_to_rad(deg: float) -> float:
    import math
    return deg * math.pi / 180.0


def rad_to_deg(rad: float) -> float:
    import math
    return rad * 180.0 / math.pi
