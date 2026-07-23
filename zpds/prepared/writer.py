"""写 data / calibration / segment.json。"""


class PreparedSegmentWriter:
    """Prepared Segment 写入器。"""

    def write(self, output_dir: str, segment_data: dict) -> str:
        """写入一个 prepared segment，返回 segment_id。"""
        raise NotImplementedError
