"""视频解码、seek、逐帧读取。"""


class VideoDecoder:
    """视频解码器封装。"""

    def __init__(self, path: str):
        self.path = path

    def read_frame(self, idx: int):
        """读取指定帧。"""
        raise NotImplementedError

    def iter_frames(self):
        """逐帧迭代器。"""
        raise NotImplementedError
