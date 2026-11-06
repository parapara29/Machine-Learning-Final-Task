# Input features, shared by training (table columns) and simulation (live state).
import numpy as np

ARM_JOINTS = range(4, 11)  # motors 0..3 are the wheels, the arm is motors 4..10

def state_from_table(table, columns):
    # Table -> dict of named columns.
    return {name: table[:, i] for i, name in enumerate(columns)}


def raw_features(s):
    # The 21 raw inputs: base pose, joint angles and velocities, target.
    return np.column_stack(
        [s["base_x"], s["base_y"], s["base_z"], s["base_bearing"]]
        + [s[f"q_mes_{i}"] for i in ARM_JOINTS]
        + [s[f"qd_mes_{i}"] for i in ARM_JOINTS]
        + [s["target_x"], s["target_y"], s["target_z"]]
    )

def wheel_features(s, extra):
    # extra adds the goal and the goal-relative features (29 inputs).
    x = raw_features(s)
    if not extra:
        return x
    dx = s["goal_x"] - s["base_x"]
    dy = s["goal_y"] - s["base_y"]
    heading_to_goal = np.arctan2(dy, dx) - s["base_bearing"]  # heading error toward the goal
    final_heading = s["goal_bearing"] - s["base_bearing"]  # orientation error at the goal
    return np.column_stack([
        x,
        s["goal_x"], s["goal_y"], s["goal_bearing"],
        np.hypot(dx, dy),
        np.sin(heading_to_goal), np.cos(heading_to_goal),
        np.sin(final_heading), np.cos(final_heading),
    ])

def arm_features(s, extra):
    # extra adds nav_done, end-effector, target error, sin/cos of q and hold error (49 inputs).
    x = raw_features(s)
    if not extra:
        return x
    return np.column_stack(
        [x, s["nav_done"]]
        + [s["ee_x"], s["ee_y"], s["ee_z"]]
        + [s[f"target_{c}"] - s[f"ee_{c}"] for c in "xyz"]
        + [np.sin(s[f"q_mes_{i}"]) for i in ARM_JOINTS]
        + [np.cos(s[f"q_mes_{i}"]) for i in ARM_JOINTS]
        + [s[f"q_init_{i}"] - s[f"q_mes_{i}"] for i in ARM_JOINTS]
    )

RAW_NAMES = (
    ["base_x", "base_y", "base_z", "base_bearing"]
    + [f"q_mes_{i}" for i in ARM_JOINTS]
    + [f"qd_mes_{i}" for i in ARM_JOINTS]
    + ["target_x", "target_y", "target_z"]
)


def wheel_feature_names(extra):
    # Names in the same order as wheel_features.
    if not extra:
        return RAW_NAMES
    return RAW_NAMES + [
        "goal_x", "goal_y", "goal_bearing", "goal_dist",
        "sin_heading_to_goal", "cos_heading_to_goal",
        "sin_final_heading", "cos_final_heading",
    ]


def arm_feature_names(extra):
    # Names in the same order as arm_features.
    if not extra:
        return RAW_NAMES
    return (
        RAW_NAMES + ["nav_done", "ee_x", "ee_y", "ee_z"]
        + [f"err_{c}" for c in "xyz"]
        + [f"sin_q_{i}" for i in ARM_JOINTS]
        + [f"cos_q_{i}" for i in ARM_JOINTS]
        + [f"hold_err_{i}" for i in ARM_JOINTS]
    )


if __name__ == "__main__":
    from pathlib import Path
    d = np.load(Path(__file__).parent / "data_randomized_cache.npz")
    s = state_from_table(d["table"][:1000], list(d["columns"]))
    s["nav_done"] = d["nav_done"][:1000]
    for extra in (False, True):
        print(extra, wheel_features(s, extra).shape, arm_features(s, extra).shape)
        assert wheel_features(s, extra).shape[1] == len(wheel_feature_names(extra))
        assert arm_features(s, extra).shape[1] == len(arm_feature_names(extra))
