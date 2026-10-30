import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from model_features import OUTPUT_COLUMNS, make_features

DATA_DIR = Path(__file__).parent / "data" / "data_randomized"
PLOT_DIR = Path(__file__).parent / "analysis_plots"
MAX_EPISODES = 250
ROW_STRIDE = 10
MAX_PLOT_POINTS = 4000


def wrap_angle(angle):
    return np.arctan2(np.sin(angle), np.cos(angle))


def plot_scatter_grid(x_dict, y, y_label, phase_mask, filename, suptitle):
    cols = 3
    rows = int(np.ceil(len(x_dict) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).flat

    rng = np.random.default_rng(42)
    idx = rng.choice(len(y), size=min(len(y), MAX_PLOT_POINTS), replace=False)

    for ax, (name, x) in zip(axes, x_dict.items()):
        xs, ys, nav = x[idx], y[idx], phase_mask[idx]
        ax.scatter(xs[nav], ys[nav], s=4, alpha=0.25, label="Nav phase (wheels moving)")
        ax.scatter(xs[~nav], ys[~nav], s=4, alpha=0.25, label="Reach phase (wheels stopped)")

        # Best-fit straight line across all points
        if np.std(x) > 1e-8:
            slope, intercept = np.polyfit(x, y, 1)
            r = np.corrcoef(x, y)[0, 1]
            x_line = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 100)
            ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.8, label=f"Fit (r={r:.2f})")
        ax.set_xlabel(name)
        ax.set_ylabel(y_label)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)

    for ax in list(axes)[len(x_dict):]:
        ax.set_visible(False)

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / filename, dpi=150)
    plt.close(fig)


def main():
    files = sorted(DATA_DIR.glob("episode_*.csv"))[:MAX_EPISODES]
    if not files:
        raise FileNotFoundError(f"No episode CSVs found in {DATA_DIR}")

    rows, bad_episodes = [], []
    for file in files:
        with file.open() as fp:
            episode = list(csv.DictReader(fp))
        end_err = float(episode[-1]["ee_error"])
        if end_err > 0.05:
            bad_episodes.append((file.name, float(episode[-1]["time"]), end_err))
            continue
        for r in episode[::ROW_STRIDE]:
            rows.append({k: float(v) for k, v in r.items()})

    print(f"Loaded {len(rows)} rows from {len(files) - len(bad_episodes)} valid episodes.")
    if bad_episodes:
        print(f"Skipped {len(bad_episodes)} failed timeout episodes (ee_error > 0.05m): {bad_episodes[:5]}")

    data = {k: np.array([r[k] for r in rows]) for k in rows[0]}
    feat_matrix = make_features(data)

    yaw = data["base_bearing"]
    gdx, gdy = data["goal_x"] - data["base_x"], data["goal_y"] - data["base_y"]
    heading_err = wrap_angle(np.arctan2(gdy, gdx) - yaw)
    bearing_err = wrap_angle(data["goal_bearing"] - yaw)

    Y = np.column_stack([data[c] for c in OUTPUT_COLUMNS])
    Y_wheels = Y[:, 7:]
    v_fwd = 0.5 * (Y_wheels[:, 0] + Y_wheels[:, 1])
    v_turn = 0.5 * (Y_wheels[:, 0] - Y_wheels[:, 1])
    nav_mask = np.max(np.abs(Y_wheels), axis=1) > 1e-6

    wheel_features = {
        "base_x (world)": data["base_x"],
        "target_x (world)": data["target_x"],
        "goal_forward": feat_matrix[:, 38],
        "goal_distance": feat_matrix[:, 40],
        "cos_heading_error": feat_matrix[:, 42],
        "goal_left": feat_matrix[:, 39],
        "target_left": feat_matrix[:, 36],
        "heading_error (wrapped)": heading_err,
        "sin_heading_error": feat_matrix[:, 41],
        "bearing_error (wrapped)": bearing_err,
        "sin_goal_bearing_error": feat_matrix[:, 43],
        "q_mes_5 (arm J2)": data["q_mes_5"],
    }

    print(f"\n{'Feature':26s} | {'Corr(Wheel Fwd)':>15s} | {'Corr(Wheel Turn)':>16s}")
    print("-" * 64)
    for name, col in wheel_features.items():
        cf = np.corrcoef(col, v_fwd)[0, 1]
        ct = np.corrcoef(col, v_turn)[0, 1]
        print(f"{name:26s} | {cf:15.4f} | {ct:16.4f}")

    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Forward wheel speed scatter plots
    plot_scatter_grid(
        {
            "base_x (world)": wheel_features["base_x (world)"],
            "goal_forward (m)": wheel_features["goal_forward"],
            "goal_distance (m)": wheel_features["goal_distance"],
            "cos_heading_error": wheel_features["cos_heading_error"],
            "heading_error (rad)": wheel_features["heading_error (rad)" if "heading_error (rad)" in wheel_features else "heading_error (wrapped)"],
            "q_mes_5 (arm J2 rad)": wheel_features["q_mes_5 (arm J2)"],
        },
        v_fwd,
        "Forward Wheel Cmd (rad/s)",
        nav_mask,
        "wheel_forward_relations.png",
        "Linearity Check: Features vs. Forward Wheel Speed (rw + lw) / 2",
    )

    # 2. Turning wheel speed scatter plots
    plot_scatter_grid(
        {
            "target_x (world)": wheel_features["target_x (world)"],
            "goal_left (m)": wheel_features["goal_left"],
            "target_left (m)": wheel_features["target_left"],
            "heading_error (wrapped)": wheel_features["heading_error (wrapped)"],
            "sin_heading_error": wheel_features["sin_heading_error"],
            "bearing_error (wrapped)": wheel_features["bearing_error (wrapped)"],
        },
        v_turn,
        "Turn Wheel Cmd (rad/s)",
        nav_mask,
        "wheel_turn_relations.png",
        "Linearity Check: Features vs. Turning Wheel Speed (rw - lw) / 2",
    )

    # 3. Arm torque vs joint angles, velocities, and EE errors
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    rng = np.random.default_rng(42)
    idx = rng.choice(len(v_fwd), size=min(len(v_fwd), MAX_PLOT_POINTS), replace=False)
    arm_pairs = [
        (data["q_mes_5"] - data["q_init_5"], Y[:, 1], "q_mes_5 - q_init_5 (rad)", "torque_arm_1 (J2 N m)"),
        (data["q_mes_5"], Y[:, 1], "q_mes_5 (J2 angle rad)", "torque_arm_1 (J2 N m)"),
        (feat_matrix[:, 47], Y[:, 1], "ee_error_z (m)", "torque_arm_1 (J2 N m)"),
        (data["q_mes_7"], Y[:, 3], "q_mes_7 (J4 angle rad)", "torque_arm_3 (J4 N m)"),
        (feat_matrix[:, 45], Y[:, 3], "ee_error_forward (m)", "torque_arm_3 (J4 N m)"),
        (feat_matrix[:, 46], Y[:, 0], "ee_error_left (m)", "torque_arm_0 (J1 N m)"),
        (data["qd_mes_4"], Y[:, 0], "qd_mes_4 (J1 vel rad/s)", "torque_arm_0 (J1 N m)"),
        (data["qd_mes_5"], Y[:, 1], "qd_mes_5 (J2 vel rad/s)", "torque_arm_1 (J2 N m)"),
        (data["q_init_10"] - data["q_mes_10"], Y[:, 6], "q_init_10 - q_mes_10 (rad)", "torque_arm_6 (J7 N m)"),
    ]
    for ax, (x, y_col, xl, yl) in zip(axes.flat, arm_pairs):
        xs, ys, nav = x[idx], y_col[idx], nav_mask[idx]
        ax.scatter(xs[nav], ys[nav], s=4, alpha=0.25, label="Nav phase")
        ax.scatter(xs[~nav], ys[~nav], s=4, alpha=0.25, label="Reach phase")
        if np.std(x) > 1e-8:
            slope, intercept = np.polyfit(x, y_col, 1)
            r = np.corrcoef(x, y_col)[0, 1]
            x_line = np.linspace(np.percentile(xs, 1), np.percentile(xs, 99), 100)
            ax.plot(x_line, slope * x_line + intercept, "r--", linewidth=1.8, label=f"Fit (r={r:.2f})")
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", fontsize=8)
    fig.suptitle("Linearity Check: Arm Torques vs. Joint State and EE Error", fontsize=14)
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "arm_relations.png", dpi=150)
    plt.close(fig)

    # 4. Correlation matrix heatmap across key features and outputs
    corr_x_names = list(wheel_features.keys()) + ["ee_error_fwd", "ee_error_left", "ee_error_z"]
    corr_x = np.column_stack([*wheel_features.values(), feat_matrix[:, 45], feat_matrix[:, 46], feat_matrix[:, 47]])
    corr_y_names = ["wheel_fwd", "wheel_turn"] + [f"tau_{i}" for i in range(7)]
    corr_y = np.column_stack([v_fwd, v_turn, Y[:, :7]])

    corr_mat = np.zeros(( len(corr_x_names), len(corr_y_names) ))
    for i in range(len(corr_x_names)):
        for j in range(len(corr_y_names)):
            corr_mat[i, j] = np.corrcoef(corr_x[:, i], corr_y[:, j])[0, 1]

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_mat, cmap="coolwarm", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr_y_names)), labels=corr_y_names, rotation=30, ha="right")
    ax.set_yticks(range(len(corr_x_names)), labels=corr_x_names)
    for i in range(len(corr_x_names)):
        for j in range(len(corr_y_names)):
            ax.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Pearson correlation (r)")
    ax.set_title("Feature vs. Output Correlation Heatmap")
    fig.tight_layout()
    fig.savefig(PLOT_DIR / "correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved plots to {PLOT_DIR}:")
    for p in sorted(PLOT_DIR.glob("*.png")):
        print(f"  - {p}")


if __name__ == "__main__":
    main()

