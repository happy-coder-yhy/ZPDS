"""单位 / 时间 / 坐标约定常量。"""

# 时间
TIME_UNIT = "ns"
TIME_EPOCH = "device_monotonic"

# 长度
LENGTH_UNIT = "m"

# 角度
ANGLE_UNIT = "rad"

# 坐标系
COORDINATE_SYSTEM = "right-handed"  # x-forward, y-left, z-up
QUATERNION_ORDER = "xyzw"
POSE_NOTATION = "T_parent_child"

# 图像格式
RGB_CODEC = "h264"
RGB_CONTAINER = "mp4"
RGB_FPS_TARGET = 30
DEPTH_FORMAT = "zarr"  # uint16, mm
