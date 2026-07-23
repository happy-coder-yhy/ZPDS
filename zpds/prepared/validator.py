"""12 级 prepared segment 校验规则。"""


class PreparedValidator:
    """Prepared Segment 校验器。"""

    def validate(self, segment_dir: str) -> list:
        """运行全部 12 级校验，返回问题列表。"""
        raise NotImplementedError
