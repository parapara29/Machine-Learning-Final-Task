"""
Phase 2: 7-DOF Kinova Arm Linear Regression Training
=====================================================
This script trains the linear regression model for the 7-DOF Kinova robotic arm.
It maps 25 engineered features to the 7 arm motor joint torques (torque_arm_0..6).

Where the Data Comes From:
- Data was collected by running `collect_data.py` on the expert simulation (`robotnik_arm_task.py`).
- 2,000 randomized episodes were saved as CSV files in `robotnik_arm_task/data/data_randomized/`.
- Each episode started with the mobile base navigating toward a random 3D target across the room,
  stopping at the target standoff (~0.70m), and then deploying the 7-DOF Kinova arm to touch the target.
- Data was recorded at 100 Hz (every 10th step of the 1,000 Hz PyBullet simulation).

Key Training & Physics Decisions:
1. Phase 2 Regime Separation (Arm Reaching Phase Only):
   - In the first 77.4% of an episode, the robot is driving its mobile base while the arm is locked
     at its home configuration. We completely throw away these driving rows (`active_wheels > 1e-6`).
   - We only train on rows where the wheels are completely stopped (`|wheel_cmds| <= 1e-6`). This ensures
     the linear arm model specializes purely in arm kinematics and inverse dynamics, without interference
     from base movement.

2. Body-Frame Target Offsets (Rotation Invariance):
   - The arm is mounted on top of the mobile base. When the base stops at an angle (yaw), target coordinates
     in world space (X, Y) are meaningless to the arm.
   - We rotate the target offset into the robot's local body frame (forward, left, height): (t_f, t_l, t_h).
     Now the arm knows exactly where the target is relative to its own shoulder, regardless of where the
     robot parked in the room.

3. Dropping Gripper World Position (Fixing the "Anti-Spring" Explosion):
   - Standard end-effector tracking features (x_ee) create a positive feedback catastrophe.
     When the arm moves forward toward the target, x_ee increases, which tricked standard linear models into
     commanding *more* forward torque, causing explosive unconstrained divergence.
   - By completely omitting x_ee and instead using measured joint angles q_mes_4..10, the linear model
     directly learns physical restoring spring torques: tau = -Kp * (q - q_target).

4. Measured Joint Velocities for Physical Damping:
   - 7 joint velocities (qd_mes_4..10) allow the linear model to learn physical damping: tau = -Kd * qd.
     This stops the arm from oscillating or overshooting when reaching for the target.

5. Quadratic Cross Terms for Arm Kinematics:
   - 6 quadratic interaction terms (t_f^2, t_l^2, t_h^2, t_f*t_h, t_f*t_l, t_l*t_h) capture the spherical
     geometry and nonlinear curvature of the Kinova arm's workspace, allowing a linear model to approximate
     nonlinear inverse kinematics accurately.

6. Low-Z Hinge Feature for Elbow Bifurcation:
   - When reaching for low targets (t_h < 0.46m), Shoulder Joint 2 reaches its hard physical lower limit
     (q5 = 0.88 rad). Below this height, reaching lower requires twisting the elbow roll joint (Joint 3 / q6).
   - We engineer hinge features: low_z = max(0, 0.46 - t_h) and low_z^2. These act like a ReLU hinge, giving
     the linear model an explicit mathematical signal to twist the elbow once Shoulder 2 bottoms out.

7. Pruning Failed Episodes and Mirror Postures:
   - Discards episodes where the expert timed out or had end-effector error > 2cm.
   - For low targets, the arm can twist its elbow either left or right (two valid mirror IK solutions).
     If we train on both, linear regression averages them to 0 (commanding 0 torque, freezing the arm).
     We filter out the negative elbow branch (q6 < 0.2 rad when q5 < 0.88 rad), teaching the model to
     consistently choose the positive elbow twist branch.

8. DART Noise Injection (State Perturbation):
   - Expert demonstrations only follow smooth, perfect trajectories. A model trained only on nominal data
     learns zero restoring stiffness if bumped off course.
   - We inject Gaussian noise into joint angles (std = 0.08 rad) and velocities (std = 0.15 rad/s) and
     recalculate target torques using the expert PD law: tau_target = tau_final + Kp*(q_final - q_pert) - Kd*qd_pert.
     This forces the linear model to learn the true physical stiffness (Kp = 100 Nm/rad) and damping (Kd = 30 Nms/rad).

Run:
  uv run train_arm_model.py
"""

import csv
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# Directory paths:
# DATA_DIR: Folder containing raw CSV episodes collected from expert runs.
# MODEL_PATH: Where the final 26x7 weight matrix and scaling constants will be saved.
# RESULTS_DIR: Where evaluation plots and metrics.csv will be saved.
DATA_DIR = Path(__file__).parent / "data_randomized"
MODEL_PATH = Path(__file__).parent.parent / "trained_model" / "arm_linear_regression.npz"
RESULTS_DIR = Path(__file__).parent / "arm_training_results"
TEST_FRACTION = 0.2
SEED = 42

# EPD gains from the expert controller code.
KP_ARM = 100.0
KD_ARM = 30.0

# Shoulder offset in the robot base frame [forward, left, up] (m) and physical reach limit (m)
SHOULDER_OFFSET = np.array([0.204, 0.0, 0.495])
ARM_REACH_LIMIT = 0.85

# Column definitions in the CSV:
# In this 11-DOF mobile manipulator:
# - Joints 0..3 are the 4 mobile base wheels.
# - Joints 4..10 are the 7 revolute joints of the Kinova Gen3 robotic arm:
#     q_mes_4: Arm Base / Shoulder Pan (yaw rotation around base vertical)
#     q_mes_5: Shoulder Joint 1 / Shoulder Pitch (elevation)
#     q_mes_6: Shoulder Joint 2 / Elbow Roll (lateral twist)
#     q_mes_7: Elbow Joint 1 / Elbow Pitch (flexion/extension)
#     q_mes_8: Wrist Joint 1 / Wrist Roll
#     q_mes_9: Wrist Joint 2 / Wrist Pitch
#     q_mes_10: Wrist Joint 3 / Gripper Roll
JOINT_POS_COLUMNS = [f"q_mes_{i}" for i in range(4, 11)]
JOINT_VEL_COLUMNS = [f"qd_mes_{i}" for i in range(4, 11)]
WHEEL_COLUMNS = [f"wheel_cmds_{i}" for i in range(4)]
OUTPUT_COLUMNS = [f"torque_arm_{i}" for i in range(7)]
STATE_COLUMNS = (
    ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z"]
    + JOINT_POS_COLUMNS
    + JOINT_VEL_COLUMNS
)

# The 25 engineered input features fed into the arm linear regression model:
# - 7 measured joint positions (learns -Kp * q spring stiffness)
# - 7 measured joint velocities (learns -Kd * qd damping)
# - 3 body-frame target offsets (t_forward, t_left, t_height)
# - 6 quadratic interaction terms (workspace curvature)
# - 2 low-Z hinge features (elbow roll trigger when shoulder bottoms out)
INPUT_COLUMNS = (
    JOINT_POS_COLUMNS
    + JOINT_VEL_COLUMNS
    + [
        "target_forward",
        "target_left",
        "target_height",
        "target_forward_sq",
        "target_left_sq",
        "target_height_sq",
        "target_forward_height",
        "target_forward_left",
        "target_left_height",
        "low_z_hinge",
        "low_z_hinge_sq",
    ]
)


def target_in_base_frame(data):
    """
    Transforms the 3D world target into the robot base frame (forward, left, height).
    
    Why this is required:
    - In the CSV files, target_x, target_y, target_z are global room coordinates.
    - The arm is mounted directly onto the mobile chassis. When the base parks at an angle
      (base_bearing = yaw), the target's relative position must be rotated by yaw so that
      the arm knows where the target sits relative to its shoulder.
      
    Math:
      dx = target_x - base_x
      dy = target_y - base_y
      target_forward =  cos(yaw) * dx + sin(yaw) * dy  (meters ahead of base center)
      target_left    = -sin(yaw) * dx + cos(yaw) * dy  (meters to the left of base center)
      target_height  = target_z - base_z               (meters above the base floor)
    """
    yaw = np.asarray(data["base_bearing"])
    c, s = np.cos(yaw), np.sin(yaw)
    dx = np.asarray(data["target_x"]) - data["base_x"]
    dy = np.asarray(data["target_y"]) - data["base_y"]
    #Same thing as train base model code, we need to convert the target position from world frame to robot's body frame, so that the model can learn to reach the target in the robot's body frame.
    target_forward = c * dx + s * dy
    target_left = -s * dx + c * dy
    target_height = np.asarray(data["target_z"]) - data["base_z"]
    return target_forward, target_left, target_height


def make_arm_features(data):
    """
    Constructs the 25 physical features for the 7-DOF arm linear regression model:
    
    1. Measured Joint Positions (7 features): q_mes_4..10
       - We explicitly OMIT end-effector world position (x_ee, y_ee, z_ee) to eliminate
         the positive-feedback "anti-spring" explosion.
       - Joint angles provide the reference for stabilizing -Kp * q restoring torques.
       
    2. Measured Joint Velocities (7 features): qd_mes_4..10
       - Provides velocity state for the model to learn physical damping (-Kd * qd),
         stopping oscillations and overshoot during arm reaching.
         
    3. Body-Frame Target Offsets (3 features): t_f, t_l, t_h
       - Linear offsets from robot base center to 3D target. Tells the arm where to reach.
       
    4. Quadratic Cross Terms (6 features):
       - t_f^2, t_l^2, t_h^2, t_f*t_h, t_f*t_l, t_l*t_h
       - The Kinova arm workspace is curved/spherical. These quadratic combinations fit
         the nonlinear workspace geometry so linear regression can solve inverse kinematics.
         
    5. Low-Height Hinge Features (2 features):
       - low_z = max(0.0, 0.46 - target_height) and low_z^2
       - When the target height is below 0.46m, Shoulder Joint 2 bottoms out at its physical
         limit (0.88 rad). To reach lower, the elbow roll joint (Joint 3 / q6) must twist.
       - low_z acts like a rectified linear hinge that turns on only below 0.46m, commanding
         the elbow roll joint to twist downward.
         
    Total: 7 + 7 + 3 + 6 + 2 = 25 features (+ 1 bias = 26 inputs).
    """
    target_forward, target_left, target_height = target_in_base_frame(data)
    q = np.column_stack([np.asarray(data[c]) for c in JOINT_POS_COLUMNS])
    qd = np.column_stack([np.asarray(data[c]) for c in JOINT_VEL_COLUMNS])
    low_z = np.maximum(0.0, 0.46 - target_height)

    return np.column_stack((
        q,                              # 7 measured joint angles (positions)
        qd,                             # 7 measured joint velocities
        target_forward,                 # 1 linear forward target offset
        target_left,                    # 1 linear lateral target offset
        target_height,                  # 1 linear vertical target offset
        target_forward ** 2,            # 1 quadratic forward term
        target_left ** 2,               # 1 quadratic lateral term
        target_height ** 2,             # 1 quadratic vertical term
        target_forward * target_height, # 1 forward-height interaction term
        target_forward * target_left,   # 1 forward-lateral interaction term
        target_left * target_height,    # 1 lateral-height interaction term
        low_z,                          # 1 low-height hinge feature: max(0, 0.46 - h)
        low_z ** 2,                     # 1 quadratic low-height hinge feature
    ))


def load_episode(file):
    """
    Loads a single episode CSV file into a dictionary of numpy arrays.
    Only extracts the columns needed for training to minimize memory and load time.
    """
    with file.open() as fp:
        header = next(csv.reader(fp))
    needed = ["time", "ee_error", "ee_x", "ee_y", "ee_z"] + STATE_COLUMNS + WHEEL_COLUMNS + OUTPUT_COLUMNS
    indices = [header.index(col) for col in needed]
    raw = np.loadtxt(file, delimiter=",", skiprows=1, usecols=indices)
    return {col: raw[:, i] for i, col in enumerate(needed)}


def load_arm_dataset(files, rng):
    """
    Loads and processes arm demonstration data across all CSV episode files:
    
    1. Filters Out Incomplete / Failed Episodes:
       - Skips episodes where final end-effector error > 2cm (expert failed or timed out).
       
    2. Filters Out the Negative Elbow Branch (Mirror Posture Bifurcation):
       - When targets are low, the arm can reach them by twisting its elbow roll either
         left (+q6) or right (-q6). Both are valid kinematic solutions.
       - But linear regression averages multimodal data! If trained on both, it predicts
         tau_elbow = 0, causing the arm to freeze.
       - We prune the negative branch (q_mes_6 < 0.2 rad when q_mes_5 < 0.88 rad), keeping
         only the clean positive twist posture.
         
    3. Isolates Phase 2 (Arm Reaching Phase Only):
       - Only rows where the base wheels are completely stopped (|wheel_cmds| <= 1e-6)
         are used. All base driving rows are discarded.
         
    4. Downsamples Snapshots:
       - Takes 25 evenly spaced snapshots across each reaching phase to prevent over-representing
         slow terminal settling times.
         
    5. DART State Perturbations (Noise Injection):
       - Nominal demonstration trajectories are clean, so standard regression learns zero restoring
         stiffness off-trajectory.
       - We inject Gaussian noise into angles (dq ~ N(0, 0.08^2)) and speeds (dqd ~ N(0, 0.15^2)).
       - Ground truth torque is recomputed using the expert PD law:
           tau_target = tau_final + Kp * (q_final - q_pert) - Kd * qd_pert
         where tau_final is the gravity holding torque at target contact.
       - This forces the linear model to learn the true physical stiffness (100 Nm/rad) and damping (30 Nms/rad).
    """
    X_chunks, y_chunks = [], []
    workspace_points = []
    skipped = 0

    for file in files:
        ep = load_episode(file)

        # Throw away bad episodes:
        # If the arm didn't reach the target (error > 2cm at the end), skip it!
        # If the target is super low (Joint 2 hits its bottom limit at 0.88 rad), the arm has two mirror ways
        #   to twist its elbow (left vs right). If we train on both, linear regression gets confused and averages
        #   them to 0 (arm gets stuck). So we only keep the episodes that twisted to the positive branch (q6 >= 0.2).
        if ep["ee_error"][-1] > 0.02 or (ep["q_mes_5"][-1] < 0.88 and ep["q_mes_6"][-1] < 0.2):
            skipped += 1
            continue

        # Only look at the Arm Phase:
        # We only want rows where wheels are completely stopped.
        wheels = np.column_stack([ep[c] for c in WHEEL_COLUMNS])
        arm_phase = np.max(np.abs(wheels), axis=1) <= 1e-6
        idx = np.where(arm_phase)[0]
        if len(idx) == 0:
            skipped += 1
            continue

        # Record where the target sits in the base frame once the wheels stop
        tf, tl, th = target_in_base_frame({k: ep[k][idx[-1]] for k in ["base_x", "base_y", "base_z", "base_bearing", "target_x", "target_y", "target_z"]})
        workspace_points.append([tf, tl, th])

        # We pick 25 evenly spaced snapshots during the arm reach. To not have too much data
        step = max(1, len(idx) // 25)
        sel = idx[::step]

        # Grab the final goal pose and final gravity holding torque:
        # At the very end of the episode, the arm is stopped right at the target.
        # tau_final is the exact torque needed to hold the arm up against gravity.
        q_final = np.array([ep[c][-1] for c in JOINT_POS_COLUMNS])
        tau_final = np.array([ep[c][-1] for c in OUTPUT_COLUMNS])

        sub = {col: ep[col][sel].copy() for col in STATE_COLUMNS}
        q_clean = np.column_stack([sub[c] for c in JOINT_POS_COLUMNS])
        qd_clean = np.column_stack([sub[c] for c in JOINT_VEL_COLUMNS])

        # DART noise trick (teach the model restoring spring stiffness and damping):
        # If the arm only sees perfect smooth trajectories, it never learns how to fight back if pushed.
        # So we add small random perturbations to the angles (dq) and speeds (dqd).
        dq = rng.normal(0.0, 0.08, size=q_clean.shape)
        dqd = rng.normal(0.0, 0.15, size=qd_clean.shape)
        q_pert = q_clean + dq
        qd_pert = qd_clean + dqd

        for j, col in enumerate(JOINT_POS_COLUMNS):
            sub[col] = q_pert[:, j]
        for j, col in enumerate(JOINT_VEL_COLUMNS):
            sub[col] = qd_pert[:, j]

        # Ground truth PD restoring torque:
        # tau_target = tau_gravity_hold + Kp * (q_desired - q_actual) - Kd * qd_actual
        tau_target = tau_final + KP_ARM * (q_final - q_pert) - KD_ARM * qd_pert

        # Add the features and target torques to our training dataset
        X_chunks.append(make_arm_features(sub))
        y_chunks.append(tau_target)

    return np.vstack(X_chunks), np.vstack(y_chunks), np.array(workspace_points), skipped


def save_workspace_plot(workspace_points, bounds):
    """
    Plots the top-down and side views of reached target positions in the robot base frame.
    Saves the visualization to arm_training_results/arm_workspace.png.
    """
    tf, tl, th = workspace_points[:, 0], workspace_points[:, 1], workspace_points[:, 2]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Top-Down view: target forward vs target left
    axes[0].scatter(tf, tl, s=10, alpha=0.35, label="Reached targets (base frame)")
    axes[0].axvline(bounds["forward_min"], color="r", linestyle="--", label="Stop window")
    axes[0].axvline(bounds["forward_max"], color="r", linestyle="--")
    axes[0].axhline(-bounds["left_max"], color="r", linestyle="--")
    axes[0].axhline(bounds["left_max"], color="r", linestyle="--")
    axes[0].set_xlabel("Target Forward in Base Frame (m)")
    axes[0].set_ylabel("Target Left in Base Frame (m)")
    axes[0].set_title("Top-Down Arm Reachable Workspace")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Side-View: target forward vs target height
    axes[1].scatter(tf, th, s=10, alpha=0.35, color="tab:orange", label="Reached targets (base frame)")
    theta = np.linspace(-np.pi / 3, np.pi / 2, 200)
    axes[1].plot(
        SHOULDER_OFFSET[0] + ARM_REACH_LIMIT * np.cos(theta),
        SHOULDER_OFFSET[2] + ARM_REACH_LIMIT * np.sin(theta),
        "k--",
        label=f"Shoulder {ARM_REACH_LIMIT}m reach sphere",
    )
    axes[1].axvline(bounds["forward_min"], color="r", linestyle="--", label="Forward stop window")
    axes[1].axvline(bounds["forward_max"], color="r", linestyle="--")
    axes[1].set_xlabel("Target Forward in Base Frame (m)")
    axes[1].set_ylabel("Target Height above Base (m)")
    axes[1].set_title("Side-View Arm Reachable Workspace")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "arm_workspace.png", dpi=150)
    plt.close(fig)


def save_plots(metrics, weights, input_mean, input_scale, train_file, test_file):
    """
    Saves evaluation bar charts (RMSE and R^2 scores) and time-series torque prediction plots
    comparing the trained linear arm model against the expert controller.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    x = np.arange(len(OUTPUT_COLUMNS))
    width = 0.35

    # Bar chart 1: RMSE in Nm per joint
    axes[0].bar(x - width / 2, [metrics[o]["train_rmse"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[0].bar(x + width / 2, [metrics[o]["test_rmse"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[0].set_xticks(x, OUTPUT_COLUMNS, rotation=25)
    axes[0].set_ylabel("RMSE (Nm)")
    axes[0].set_title("Arm Torque RMSE")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Bar chart 2: R^2 Score per joint
    axes[1].bar(x - width / 2, [metrics[o]["train_r2"] for o in OUTPUT_COLUMNS], width, label="Train")
    axes[1].bar(x + width / 2, [metrics[o]["test_r2"] for o in OUTPUT_COLUMNS], width, label="Test")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xticks(x, OUTPUT_COLUMNS, rotation=25)
    axes[1].set_ylabel("R² score")
    axes[1].set_title("Arm Torque R²")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics.png", dpi=150)
    plt.close(fig)

    # Time-series prediction plots for a training episode and a test episode
    for split, file in [("train", train_file), ("test", test_file)]:
        ep = load_episode(file)
        wheels = np.column_stack([ep[c] for c in WHEEL_COLUMNS])
        arm_phase = np.max(np.abs(wheels), axis=1) <= 1e-6
        features = (make_arm_features(ep) - input_mean) / input_scale
        predictions = np.clip(np.column_stack((np.ones(len(ep["time"])), features)) @ weights, -30.0, 30.0)

        fig, axes = plt.subplots(4, 2, figsize=(12, 10), sharex=True)
        for i, output in enumerate(OUTPUT_COLUMNS):
            ax = axes.flat[i]
            ax.plot(ep["time"][arm_phase], ep[output][arm_phase], label="Expert (arm phase)")
            ax.plot(ep["time"][arm_phase], predictions[arm_phase, i], "--", label="Predicted (arm model)")
            ax.set_title(output)
            ax.set_ylabel("Nm")
            ax.grid(True, alpha=0.3)
        axes.flat[-1].set_visible(False)
        axes.flat[0].legend()
        fig.suptitle(f"{split.capitalize()} episode (arm reaching phase): {file.name}")
        fig.supxlabel("Simulation time (s)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{split}_predictions.png", dpi=150)
        plt.close(fig)


def main():
    # 1. Load list of all CSV demonstration files
    files = sorted(DATA_DIR.glob("episode_*.csv"))
    if len(files) < 2:
        raise ValueError(f"Need at least two episode CSVs in {DATA_DIR}")

    # 2. Episode-Level Train / Test Split (80% train, 20% test):
    # We split by entire EPISODES (not by individual rows). Splitting whole episodes
    # ensures the test set evaluates true generalisation on completely unseen targets.
    rng = np.random.default_rng(SEED)
    rng.shuffle(files)
    num_test = int(np.ceil(len(files) * TEST_FRACTION))
    test_files = files[:num_test]
    train_files = files[num_test:]

    # 3. Load features, targets, and reachable workspace data:
    raw_train_X, y_train, ws_train, skipped_train = load_arm_dataset(train_files, rng)
    raw_test_X, y_test, ws_test, skipped_test = load_arm_dataset(test_files, rng)
    workspace_points = np.vstack((ws_train, ws_test))

    # 4. Empirical Workspace Bounds (The Arm Sweet-Spot Window):
    # From all reached targets across the dataset, we extract the 1st to 99th percentile
    # of forward reach (typically [0.60, 0.85]m) and lateral reach (typically [-0.20, +0.20]m).
    # These empirical boundaries are stored in the model file so the combined controller knows
    # exactly when the base has entered the arm's sweet spot and should hand over control.
    workspace_bounds = {
        "forward_min": float(np.percentile(workspace_points[:, 0], 1)),
        "forward_max": float(np.percentile(workspace_points[:, 0], 99)),
        "left_max": float(np.percentile(np.abs(workspace_points[:, 1]), 95)),
        "height_min": float(np.min(workspace_points[:, 2])),
        "height_max": float(np.max(workspace_points[:, 2])),
    }

    print("==================================================================")
    print("Arm Reachable Workspace in Robot Base Frame (from expert data):")
    print(f"  Forward reach sweet spot : [{workspace_bounds['forward_min']:.3f}, {workspace_bounds['forward_max']:.3f}] m")
    print(f"  Lateral alignment window : [-{workspace_bounds['left_max']:.3f}, +{workspace_bounds['left_max']:.3f}] m")
    print(f"  Height above base        : [{workspace_bounds['height_min']:.3f}, {workspace_bounds['height_max']:.3f}] m")
    print(f"  Max 3D distance from shoulder {SHOULDER_OFFSET.tolist()} : {ARM_REACH_LIMIT:.2f} m")
    print("==================================================================\n")

    # 5. Feature Normalization (Z-score):
    # Scales all 25 features to zero mean and unit variance.
    # Prevents large squared features (t_f^2) from dominating small angle errors (q).
    input_mean = raw_train_X.mean(axis=0)
    input_scale = raw_train_X.std(axis=0)
    input_scale[np.isclose(input_scale, 0)] = 1.0

    # 6. Prepend Bias Column:
    # Adds a constant 1.0 column so linear regression fits an intercept (bias term).
    X_train = np.column_stack((np.ones(len(raw_train_X)), (raw_train_X - input_mean) / input_scale))
    X_test = np.column_stack((np.ones(len(raw_test_X)), (raw_test_X - input_mean) / input_scale))

    # 7. Exact Ordinary Least Squares (OLS) Solution:
    # Solves W = (X^T X)^(-1) X^T Y using the Moore-Penrose pseudoinverse (pinv).
    # Produces a 26 x 7 weight matrix mapping (1 bias + 25 features) -> 7 joint motor torques.
    weights = np.linalg.pinv(X_train) @ y_train

    print(f"Training: {len(train_files) - skipped_train} valid episodes ({skipped_train} skipped), {len(y_train)} samples")
    print(f"Testing : {len(test_files) - skipped_test} valid episodes ({skipped_test} skipped), {len(y_test)} samples\n")

    # 8. Evaluation & Metrics Computation:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics = {o: {} for o in OUTPUT_COLUMNS}
    for split, X, actual in [("train", X_train, y_train), ("test", X_test, y_test)]:
        sq_err = np.sum((actual - X @ weights) ** 2, axis=0)
        tot_var = np.sum((actual - actual.mean(axis=0)) ** 2, axis=0)
        rmse = np.sqrt(sq_err / len(actual))
        r2 = 1.0 - sq_err / tot_var
        for i, col in enumerate(OUTPUT_COLUMNS):
            metrics[col][f"{split}_rmse"] = rmse[i]
            metrics[col][f"{split}_r2"] = r2[i]

    print(f"{'Output':14s} | {'Train RMSE':>10s} | {'Train R2':>9s} | {'Test RMSE':>10s} | {'Test R2':>9s}")
    print("-" * 64)
    with (RESULTS_DIR / "metrics.csv").open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["output", "train_rmse", "train_r2", "test_rmse", "test_r2"])
        for col in OUTPUT_COLUMNS:
            m = metrics[col]
            print(f"{col:14s} | {m['train_rmse']:10.4f} | {m['train_r2']:9.4f} | {m['test_rmse']:10.4f} | {m['test_r2']:9.4f}")
            writer.writerow([col, m["train_rmse"], m["train_r2"], m["test_rmse"], m["test_r2"]])

    # 9. Save Plots and Workspace Diagnostics
    save_workspace_plot(workspace_points, workspace_bounds)
    save_plots(metrics, weights, input_mean, input_scale, train_files[0], test_files[0])
    print(f"\nSaved workspace analysis, metrics, and plots to {RESULTS_DIR}")

    # 10. Save Model Weights & Metadata to .npz:
    # Stored in `trained_model/arm_linear_regression.npz` so `robotnik_combined_model.py`
    # and evaluation scripts can load and run inference in microseconds.
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        MODEL_PATH,
        weights=weights,
        input_mean=input_mean,
        input_scale=input_scale,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        forward_min=workspace_bounds["forward_min"],
        forward_max=workspace_bounds["forward_max"],
        left_max=workspace_bounds["left_max"],
    )
    print(f"Saved arm model to {MODEL_PATH}")


if __name__ == "__main__":
    matplotlib.use("Agg")
    main()
