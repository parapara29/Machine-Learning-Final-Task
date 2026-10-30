import numpy as np

ARM_COLUMNS = [f"q_mes_{i}" for i in range(4, 11)]
VELOCITY_COLUMNS = [f"qd_mes_{i}" for i in range(4, 11)]
INITIAL_COLUMNS = [f"q_init_{i}" for i in range(4, 11)]
OUTPUT_COLUMNS = [f"torque_arm_{i}" for i in range(7)] + [f"wheel_cmds_{i}" for i in range(4)]
STATE_COLUMNS = (
    ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z",
     "ee_x", "ee_y", "ee_z", "goal_x", "goal_y", "goal_bearing"]
    + ARM_COLUMNS + VELOCITY_COLUMNS + INITIAL_COLUMNS
)
INPUT_COLUMNS = (
    ARM_COLUMNS + VELOCITY_COLUMNS + INITIAL_COLUMNS
    + [f"sin_q_{i}" for i in range(4, 11)]
    + [f"cos_q_{i}" for i in range(4, 11)]
    + ["target_forward", "target_left", "target_height", "goal_forward", "goal_left",
       "goal_distance", "sin_heading_error", "cos_heading_error",
       "sin_goal_bearing_error", "cos_goal_bearing_error",
       "ee_error_forward", "ee_error_left", "ee_error_z"]
)


def make_features(data):
    q = np.column_stack([data[column] for column in ARM_COLUMNS])
    qd = np.column_stack([data[column] for column in VELOCITY_COLUMNS])
    q_init = np.column_stack([data[column] for column in INITIAL_COLUMNS])
    yaw = np.asarray(data["base_bearing"])
    c, s = np.cos(yaw), np.sin(yaw)
    target_dx = np.asarray(data["target_x"]) - data["base_x"]
    target_dy = np.asarray(data["target_y"]) - data["base_y"]
    goal_dx = np.asarray(data["goal_x"]) - data["base_x"]
    goal_dy = np.asarray(data["goal_y"]) - data["base_y"]
    heading_error = np.arctan2(goal_dy, goal_dx) - yaw
    bearing_error = np.asarray(data["goal_bearing"]) - yaw
    ee_dx = np.asarray(data["target_x"]) - data["ee_x"]
    ee_dy = np.asarray(data["target_y"]) - data["ee_y"]

    # Rotate horizontal offsets from world coordinates into the robot's yaw frame.
    return np.column_stack((
        q, qd, q_init, np.sin(q), np.cos(q),
        c * target_dx + s * target_dy, -s * target_dx + c * target_dy,
        np.asarray(data["target_z"]) - data["base_z"],
        c * goal_dx + s * goal_dy, -s * goal_dx + c * goal_dy,
        np.hypot(goal_dx, goal_dy), np.sin(heading_error), np.cos(heading_error),
        np.sin(bearing_error), np.cos(bearing_error),
        c * ee_dx + s * ee_dy, -s * ee_dx + c * ee_dy,
        np.asarray(data["target_z"]) - data["ee_z"],
    ))
