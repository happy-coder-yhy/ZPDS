from zpds.segmentation.robot_spans import propose_edge_idle


def test_middle_idle_is_not_proposed_for_trim() -> None:
    candidates = propose_edge_idle(
        [0, 1, 2, 3, 4],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        [1, 0, 0, 0, 1],
        motion_max=0,
        gripper_max=0,
        visual_change_max=0,
    )
    assert candidates == []
