from pathlib import Path
import numpy as np
import time
import joblib
import matplotlib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from features import (
    state_from_table, wheel_features, arm_features, wheel_feature_names, arm_feature_names,
)
import json

CACHE = Path(__file__).parent / "data_randomized_cache.npz"
RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42
SPLIT_FILE = Path(__file__).parent / "episode_split.json"  # ricktho split: 1400 / 300 / 300
SUBSAMPLE = 10
# Every run trains both feature sets on the same split, so they can be compared directly.
FEATURE_SETS = ["raw", "extra"]
N_ESTIMATORS = 50
MIN_SAMPLES_LEAF = 10

ARM_OUTPUTS = [f"torque_arm_{i}" for i in range(7)]
WHEEL_OUTPUTS = [f"wheel_cmds_{i}" for i in range(4)]
COLORS = {"raw": "#2a78d6", "extra": "#eb6834"}
INK, MUTED = "#1f1f1e", "#52514e"

def split_episodes():
    split = json.loads(SPLIT_FILE.read_text())

    test_ids = np.array(split["test"])
    val_ids = np.array(split["validation"])
    train_ids = np.array(split["train"])
    return train_ids, val_ids, test_ids

def load_split(d, ids):
    mask = np.isin(d["episode"], ids)
    table = d["table"][mask][::SUBSAMPLE]
    nav_done = d["nav_done"][mask][::SUBSAMPLE]
    s = state_from_table(table, list(d["columns"]))
    s["nav_done"] = nav_done
    return s

def make_xy(s, extra):
    y_arm = np.column_stack([s[f"torque_arm_{i}"] for i in range(7)])
    y_wheel = np.column_stack([s[f"wheel_cmds_{i}"] for i in range(4)])

    X_arm = arm_features(s, extra)
    X_wheel = wheel_features(s, extra)

    moving = ~s["nav_done"]
    X_wheel = X_wheel[moving]
    y_wheel = y_wheel[moving]

    return X_arm, y_arm, X_wheel, y_wheel

def train_rf(X, y):
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        min_samples_leaf=MIN_SAMPLES_LEAF,
        n_jobs=-1,
        random_state=SEED,
    )
    rf.fit(X,y)
    return rf

def evaluate(rf, X, y):
    pred = rf.predict(X)
    return r2_score(y, pred, multioutput="raw_values")

def predict_ms(rf, x):
    # Time of one single-sample prediction, as in the simulation loop.
    rf.n_jobs = 1
    start = time.perf_counter()
    for _ in range(100):
        rf.predict(x)
    return (time.perf_counter() - start) / 100 * 1000

def next_run_dir():
    # Each run gets its own folder: rf_run_1, rf_run_2, ...
    n = 1
    while (RESULTS_DIR / f"rf_run_{n}").exists():
        n += 1
    run_dir = RESULTS_DIR / f"rf_run_{n}"
    run_dir.mkdir(parents=True)
    return run_dir

# //plotting
def plot_importance(rf, names, feature_set, title, path, top=15):
    importance = rf.feature_importances_
    order = np.argsort(importance)[-top:]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(np.arange(len(order)), importance[order], height=0.7, color=COLORS[feature_set])
    ax.set_yticks(np.arange(len(order)), [names[i] for i in order], color=INK)
    for y, value in enumerate(importance[order]):
        ax.text(value, y, f" {value:.2f}", va="center", fontsize=8, color=MUTED)
    ax.set(xlabel="Importance (share of total, sums to 1)", title=title)
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_r2_comparison(results, path):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True, gridspec_kw={"width_ratios": [7, 4]})
    width = 0.38
    for ax, part, outputs in [(axes[0], "arm", ARM_OUTPUTS), (axes[1], "wheel", WHEEL_OUTPUTS)]:
        x = np.arange(len(outputs))
        for k, feature_set in enumerate(FEATURE_SETS):
            ax.bar(x + (k - 0.5) * width, results[feature_set][f"test_r2_{part}"], width,
                   color=COLORS[feature_set], edgecolor="white", linewidth=1, label=feature_set)
        ax.set_xticks(x, [name.split("_")[-1] for name in outputs])
        ax.set(xlabel="Joint" if part == "arm" else "Wheel", title=f"{part.capitalize()} model")
        ax.set_ylim(0, 1)
        ax.grid(True, axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Test R²")
    fig.suptitle("Random forest: test R² per output, raw vs extra input features")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, title="Input features", loc="upper right", ncol=2, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=150)
    plt.close(fig)

def run_feature_set(d, splits, feature_set, run_dir):
    extra = feature_set == "extra"
    train_ids, val_ids, test_ids = splits
    X_arm, y_arm, X_wheel, y_wheel = make_xy(load_split(d, train_ids), extra)
    X_arm_val, y_arm_val, X_wheel_val, y_wheel_val = make_xy(load_split(d, val_ids), extra)
    X_arm_test, y_arm_test, X_wheel_test, y_wheel_test = make_xy(load_split(d, test_ids), extra)
    print(f"\n[{feature_set}] train {X_arm.shape} arm, {X_wheel.shape} wheel")

    start = time.perf_counter()
    rf_wheel = train_rf(X_wheel, y_wheel)
    wheel_time = time.perf_counter() - start
    start = time.perf_counter()
    rf_arm = train_rf(X_arm, y_arm)
    arm_time = time.perf_counter() - start

    result = {
        "val_r2_wheel": evaluate(rf_wheel, X_wheel_val, y_wheel_val),
        "val_r2_arm": evaluate(rf_arm, X_arm_val, y_arm_val),
        "test_r2_wheel": evaluate(rf_wheel, X_wheel_test, y_wheel_test),
        "test_r2_arm": evaluate(rf_arm, X_arm_test, y_arm_test),
        "train_s_wheel": wheel_time,
        "train_s_arm": arm_time,
        "predict_ms_wheel": predict_ms(rf_wheel, X_wheel_val[:1]),
        "predict_ms_arm": predict_ms(rf_arm, X_arm_val[:1]),
    }
    for key, value in result.items():
        print(f"{key:18s} {np.round(value, 3)}")

    out = run_dir / feature_set
    out.mkdir()
    joblib.dump(rf_wheel, out / "rf_wheel.joblib", compress=3)
    joblib.dump(rf_arm, out / "rf_arm.joblib", compress=3)
    np.savez(out / "metrics.npz", **result,
             wheel_features=wheel_feature_names(extra), arm_features=arm_feature_names(extra))

    plot_importance(rf_wheel, wheel_feature_names(extra), feature_set,
                    f"Wheel RF feature importance ({feature_set} inputs)", out / "importance_wheel.png")
    plot_importance(rf_arm, arm_feature_names(extra), feature_set,
                    f"Arm RF feature importance ({feature_set} inputs)", out / "importance_arm.png")
    return result


if __name__ == "__main__":
    matplotlib.use("Agg")
    d = np.load(CACHE)
    d = {key: d[key] for key in d.files}
    splits = split_episodes()
    run_dir = next_run_dir()

    np.savez(run_dir / "config.npz", seed=SEED, subsample=SUBSAMPLE, n_estimators=N_ESTIMATORS,
             min_samples_leaf=MIN_SAMPLES_LEAF, train_ids=splits[0], val_ids=splits[1], test_ids=splits[2])
    results = {feature_set: run_feature_set(d, splits, feature_set, run_dir) for feature_set in FEATURE_SETS}
    plot_r2_comparison(results, run_dir / "r2_comparison.png")
    print(f"\nSaved models, metrics and plots to {run_dir}")
