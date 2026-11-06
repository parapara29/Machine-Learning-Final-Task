from pathlib import Path
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from features import state_from_table, wheel_features, arm_features, wheel_feature_names, ARM_JOINTS
from train_rf import (
    CACHE, SUBSAMPLE, COLORS, INK, MUTED, split_episodes, load_split, train_rf, evaluate,
)

# Data analysis figures for the report, one per root cause:
#   1. A random forest cannot extrapolate (absolute coordinates, target range)
#   2. Hidden state: the expert holds a randomized start pose
#   3. Single instant vs temporal inputs: where the arm forest's errors are

RUN_DIR = Path(__file__).parent / "results" / "rf_run_4"  # saved forests to analyse
OUT_DIR = Path(__file__).parent / "results" / "analysis"  # report figures
TARGETS_CSV = Path(__file__).parent.parent / "data_randomized" / "targets.csv"  # random arm offset of each episode
COLORS = {**COLORS, "lags": "#1f9e89", "relative": "#1f9e89"}
GREY = "#a3a29f"
# Past joint states added as temporal inputs: 10 and 30 rows back = 0.1 s and 0.3 s at 100 Hz.
LAG_ROWS = (10, 30)
# The whole scene (base, goal, target) is moved by these distances in x; the expert command is unchanged.
SHIFTS = np.arange(0, 2.01, 0.25)
# Manual targets that failed in simulation, outside the training range.
OOD_TARGETS = np.array([[2.0, 1.0, 1.0], [2.0, 1.0, 0.5]])
JOINT_LABELS = [f"J{i}" for i in range(1, 8)]   # J1..J7 = motors 4..10 = torque_arm_0..6


def style(ax):
    # Shared axis style.
    ax.grid(True, alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def arm_targets(s):
    # The 7 joint torques.
    return np.column_stack([s[f"torque_arm_{i}"] for i in range(7)])


def load_split_with_lags(d, ids, subsample=SUBSAMPLE):
    # Like load_split, plus the arm joint angles and velocities LAG_ROWS rows earlier in the same
    # episode (the first row is repeated at the start). Lags are taken before subsampling.
    # Rows are indexed in the full table, so only the subsampled rows are copied (saves memory).
    rows = np.flatnonzero(np.isin(d["episode"], ids))
    episode = d["episode"][rows]
    first = np.r_[True, episode[1:] != episode[:-1]]
    start = np.maximum.accumulate(np.where(first, rows, 0))
    keep, start = rows[::subsample], start[::subsample]
    columns = list(d["columns"])
    state_cols = [columns.index(f"{k}_{i}") for k in ("q_mes", "qd_mes") for i in ARM_JOINTS]
    lags = np.column_stack([d["table"][np.maximum(keep - lag, start)][:, state_cols] for lag in LAG_ROWS])
    s = state_from_table(d["table"][keep], columns)
    s["nav_done"] = d["nav_done"][keep]
    return s, lags


def episode_targets(d, ids):
    # One target per episode.
    columns = list(d["columns"])
    ep = d["episode"]
    first = np.r_[True, ep[1:] != ep[:-1]] & np.isin(ep, ids)
    return d["table"][first][:, [columns.index(f"target_{c}") for c in "xyz"]]


def load_model(feature_set, part):
    # Saved forest of RUN_DIR.
    rf = joblib.load(RUN_DIR / feature_set / f"rf_{part}.joblib")
    rf.n_jobs = -1
    return rf


# //figure 1
def relative_wheel_features(s):
    # Only the goal-relative part of the extra wheel features (goal_dist ... cos_final_heading).
    return wheel_features(s, True)[:, -5:]


def wheel_xy(s):
    # Driving rows and their wheel commands.
    moving = ~s["nav_done"]
    return moving, np.column_stack([s[f"wheel_cmds_{i}"] for i in range(4)])[moving]


def shift_test(d, train_ids, test_ids):
    # Wheel R² when the whole scene is moved: the relative geometry and the expert command stay the same.
    s_train = load_split(d, train_ids)
    moving, y_train = wheel_xy(s_train)
    print("Training wheel forest on relative features only ...")
    feature_fns = {
        "raw": (load_model("raw", "wheel"), lambda s: wheel_features(s, False)),
        "extra": (load_model("extra", "wheel"), lambda s: wheel_features(s, True)),
        "relative": (train_rf(relative_wheel_features(s_train)[moving], y_train), relative_wheel_features),
    }
    s = load_split(d, test_ids)
    moving, y = wheel_xy(s)
    r2 = {}
    for name, (rf, features) in feature_fns.items():
        scores = []
        for shift in SHIFTS:
            shifted = dict(s)
            for key in ("base_x", "goal_x", "target_x"):
                shifted[key] = s[key] + shift
            scores.append(evaluate(rf, features(shifted)[moving], y).mean())
        r2[name] = np.array(scores)
    return r2


def figure_extrapolation(d, splits, path):
    # Figure 1: shift test, wheel importance, training target range.
    train_ids, _, test_ids = splits
    r2 = shift_test(d, train_ids, test_ids)
    rf = load_model("extra", "wheel")
    names = wheel_feature_names(True)
    targets = episode_targets(d, train_ids)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), gridspec_kw={"width_ratios": [1.1, 1, 1.1]})

    ax = axes[0]
    labels = {"raw": "raw (21)", "extra": "extra (29)", "relative": "goal-relative only (5)"}
    markers = {"raw": "o", "extra": "s", "relative": "^"}
    for name in ("raw", "extra", "relative"):
        ax.plot(SHIFTS, r2[name], marker=markers[name], linewidth=2, color=COLORS[name], label=labels[name])
    ax.set(xlabel="Shift of base, goal and target in x (m)", ylabel="Wheel test R² (mean of 4 wheels)",
           title="A. Same situation, moved across the room", ylim=(0, 1.05))
    ax.legend(title="Wheel input features", frameon=False, loc="lower left")

    ax = axes[1]
    order = np.argsort(rf.feature_importances_)[-10:]
    ax.barh(np.arange(len(order)), rf.feature_importances_[order], height=0.7, color=COLORS["extra"])
    ax.set_yticks(np.arange(len(order)), [names[i] for i in order])
    for y, value in enumerate(rf.feature_importances_[order]):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set(xlabel="Importance (share of total)", title="B. Wheel forest relies on goal-relative inputs")

    ax = axes[2]
    radius = np.hypot(targets[:, 0], targets[:, 1])
    points = ax.scatter(targets[:, 0], targets[:, 1], c=targets[:, 2], cmap="viridis", s=8, alpha=0.8)
    angle = np.linspace(-np.pi, np.pi, 200)
    for r in (radius.min(), radius.max()):
        ax.plot(r * np.cos(angle), r * np.sin(angle), color=MUTED, linestyle="--", linewidth=1)
    ax.scatter(OOD_TARGETS[:, 0], OOD_TARGETS[:, 1], marker="x", s=90, linewidths=2.5, color=INK,
               label="Failed manual targets")
    ax.scatter([0], [0], marker="^", s=70, color=INK)
    ax.set(xlabel="Target X (m)", ylabel="Target Y (m)", aspect="equal", xlim=(-2, 2.3), ylim=(-2.3, 2),
           title=f"C. Training targets (radius {radius.min():.1f}–{radius.max():.1f} m)")
    fig.colorbar(points, ax=ax, label="Target height Z (m)")
    ax.legend(frameon=False, loc="lower right", fontsize=8)

    for ax in axes:
        style(ax)
    fig.suptitle("Random forest cause 1: a forest cannot extrapolate beyond its training data")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    for name in ("raw", "extra", "relative"):
        print(f"[1] wheel R² vs shift {name:8s} {np.round(r2[name], 3)}")


# //figure 2
def hold_pose_data(d):
    # The nominal start pose is q_init minus the random offset in targets.csv (the same for every episode).
    columns = list(d["columns"])
    table = np.genfromtxt(TARGETS_CSV, delimiter=",", names=True)
    offsets = {int(row["episode"]): np.array([row[f"arm_offset_{i}"] for i in ARM_JOINTS]) for row in table}
    q_cols = [columns.index(f"q_mes_{i}") for i in ARM_JOINTS]
    init_cols = [columns.index(f"q_init_{i}") for i in ARM_JOINTS]
    ep = d["episode"]
    first = np.r_[True, ep[1:] != ep[:-1]]
    q_init_first = d["table"][first][:, init_cols]
    nominal = np.median(q_init_first - np.array([offsets[e] for e in ep[first]]), axis=0)

    nav = ~d["nav_done"]
    rows = d["table"][nav][::SUBSAMPLE]
    q, q_init = rows[:, q_cols], rows[:, init_cols]
    return nominal, np.abs(q - q_init).ravel(), np.abs(q - nominal).ravel(), offsets


def figure_hold_pose(d, splits, r2_no_hold, r2_extra, path):
    # Figure 2: the expert holds q_init, and what hold_err adds.
    _, _, test_ids = splits
    nominal, to_init, to_nominal, offsets = hold_pose_data(d)
    # Example: the test episode and joint with the largest start offset.
    episode = max(test_ids, key=lambda e: np.abs(offsets[e]).max())
    joint = int(np.argmax(np.abs(offsets[episode])))
    columns = list(d["columns"])
    rows = d["table"][d["episode"] == episode]
    t = rows[:, columns.index("time")]
    nav_done = d["nav_done"][d["episode"] == episode]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))

    ax = axes[0]
    ax.axvspan(t[nav_done].min(), t.max(), color=GREY, alpha=0.2, label="Reaching phase")
    ax.plot(t, rows[:, columns.index(f"q_mes_{joint + 4}")], color=INK, linewidth=2, label="Measured angle")
    ax.axhline(rows[0, columns.index(f"q_init_{joint + 4}")], color=COLORS["extra"], linestyle="--",
               linewidth=2, label="Randomized start pose (q_init)")
    ax.axhline(nominal[joint], color=COLORS["raw"], linestyle=":", linewidth=2, label="Nominal start pose")
    ax.set(xlabel="Time (s)", ylabel=f"{JOINT_LABELS[joint]} angle (rad)",
           title=f"A. Test episode {episode}: the expert holds q_init")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    bins = np.logspace(-4, 0, 50)
    ax.hist(to_nominal, bins=bins, color=COLORS["raw"], alpha=0.75, label="|q − nominal pose|")
    ax.hist(to_init, bins=bins, color=COLORS["extra"], alpha=0.75, label="|q − q_init|")
    ax.set_xscale("log")
    ax.set(xlabel="Joint angle distance (rad, log scale)", ylabel="Rows (driving phase, all joints)",
           title="B. During driving, q stays at q_init, not the nominal pose")
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    x = np.arange(7)
    width = 0.38
    ax.bar(x - width / 2, r2_no_hold, width, color=GREY, edgecolor="white", label="extra without hold_err")
    ax.bar(x + width / 2, r2_extra, width, color=COLORS["extra"], edgecolor="white", label="extra (with hold_err)")
    ax.set_xticks(x, JOINT_LABELS)
    ax.set(ylabel="Arm test R²", ylim=(0, 1), title="C. Adding hold_err = q_init − q")
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    for ax in axes:
        style(ax)
    fig.suptitle("Random forest cause 2: hidden state, the expert holds a randomized start pose")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[2] median |q - q_init| {np.median(to_init):.4f} rad, median |q - nominal| {np.median(to_nominal):.4f} rad")
    print(f"    arm R² without hold_err {np.round(r2_no_hold, 3)}\n    arm R² with hold_err    {np.round(r2_extra, 3)}")


# //figure 3
def figure_temporal(d, splits, r2, models, path, joint=5):
    # Figure 3: past joint states do not help; errors sit in torque transients.
    _, _, test_ids = splits
    episode = test_ids[0]
    s, lags = load_split_with_lags(d, [episode], subsample=1)
    x_extra = arm_features(s, True)
    expert = arm_targets(s)[:, joint]
    pred_extra = models["extra"].predict(x_extra)[:, joint]
    pred_lags = models["lags"].predict(np.column_stack([x_extra, lags]))[:, joint]
    t = s["time"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1, 1.2]})

    ax = axes[0]
    x = np.arange(7)
    width = 0.27
    for k, (key, label) in enumerate([("raw", "raw"), ("extra", "extra"), ("lags", "extra + past states")]):
        ax.bar(x + (k - 1) * width, r2[key], width, color=COLORS[key], edgecolor="white", label=label)
    ax.set_xticks(x, JOINT_LABELS)
    ax.set(ylabel="Arm test R²", ylim=(0, 1), title="A. Single instant vs temporal inputs")
    ax.legend(title="Input features", frameon=False, fontsize=8, loc="lower left")

    ax = axes[1]
    ax.axvspan(t[s["nav_done"]].min(), t.max(), color=GREY, alpha=0.2, label="Reaching phase")
    ax.plot(t, expert, color=INK, linewidth=2, label="Expert")
    ax.plot(t, pred_extra, color=COLORS["extra"], linewidth=1.5, label="RF extra")
    ax.plot(t, pred_lags, color=COLORS["lags"], linewidth=1.5, label="RF extra + past states")
    ax.set(xlabel="Time (s)", ylabel=f"{JOINT_LABELS[joint]} torque (N·m)",
           title=f"B. Open-loop prediction on test episode {episode}")
    ax.legend(frameon=False, fontsize=8)

    for ax in axes:
        style(ax)
    fig.suptitle("Random forest cause 3: errors sit in short torque transients; past joint states "
                 f"({', '.join(f'{lag / 100:.1f} s' for lag in LAG_ROWS)} earlier) do not help")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[3] arm R² raw   {np.round(r2['raw'], 3)}\n    arm R² extra {np.round(r2['extra'], 3)}\n"
          f"    arm R² lags  {np.round(r2['lags'], 3)}")


def arm_ablation(d, splits):
    # Test R² of the saved raw/extra arm forests, and two new ones: extra without hold_err, extra + lags.
    train_ids, _, test_ids = splits
    s_train, lags_train = load_split_with_lags(d, train_ids)
    s_test, lags_test = load_split_with_lags(d, test_ids)
    y_train, y_test = arm_targets(s_train), arm_targets(s_test)
    x_train, x_test = arm_features(s_train, True), arm_features(s_test, True)
    num_hold = len(ARM_JOINTS)   # hold_err is the last block of the extra arm features

    models = {"raw": load_model("raw", "arm"), "extra": load_model("extra", "arm")}
    print("Training arm forest without hold_err ...")
    models["no_hold"] = train_rf(x_train[:, :-num_hold], y_train)
    print("Training arm forest with past states ...")
    models["lags"] = train_rf(np.column_stack([x_train, lags_train]), y_train)

    r2 = {
        "raw": evaluate(models["raw"], arm_features(s_test, False), y_test),
        "extra": evaluate(models["extra"], x_test, y_test),
        "no_hold": evaluate(models["no_hold"], x_test[:, :-num_hold], y_test),
        "lags": evaluate(models["lags"], np.column_stack([x_test, lags_test]), y_test),
    }
    return r2, models


if __name__ == "__main__":
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    d = np.load(CACHE)
    d = {key: d[key] for key in d.files}
    splits = split_episodes()

    figure_extrapolation(d, splits, OUT_DIR / "cause1_extrapolation.png")
    r2, models = arm_ablation(d, splits)
    figure_hold_pose(d, splits, r2["no_hold"], r2["extra"], OUT_DIR / "cause2_hold_pose.png")
    figure_temporal(d, splits, r2, models, OUT_DIR / "cause3_temporal.png")
    print(f"\nSaved figures to {OUT_DIR}")
