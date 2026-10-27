from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "linear_regression.npz"
RESULTS_DIR = Path(__file__).parent / "training_results"
TEST_FRACTION = 0.2
SEED = 42

INPUT_COLUMNS = (
    ["base_x", "base_y", "base_z", "base_bearing"]
    + [f"q_mes_{i}" for i in range(4, 11)]
    + [f"qd_mes_{i}" for i in range(4, 11)]
    + ["target_x", "target_y", "target_z"]
)
OUTPUT_COLUMNS = (
    [f"torque_arm_{i}" for i in range(7)]
    + [f"wheel_cmds_{i}" for i in range(4)]
)


def load_data(files):
    return pd.concat(
        [pd.read_csv(file, usecols=INPUT_COLUMNS + OUTPUT_COLUMNS) for file in files],
        ignore_index=True,
    )

# CHAT GPT CODED THIS PART:
def save_plots(metrics, weights, train_file, test_file):
    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    for row, (outputs, title, unit) in enumerate([
        (OUTPUT_COLUMNS[:7], "Arm torques", "N m"),
        (OUTPUT_COLUMNS[7:], "Wheel commands", "rad/s"),
    ]):
        metrics.loc[outputs, ["train_rmse", "test_rmse"]].plot.bar(ax=axes[row, 0], rot=30)
        axes[row, 0].set(title=title, ylabel=f"RMSE ({unit})", xlabel="")
        metrics.loc[outputs, ["train_r2", "test_r2"]].plot.bar(ax=axes[row, 1], rot=30)
        axes[row, 1].set(title=title, ylabel="R² score", xlabel="")
        axes[row, 1].axhline(1, color="black", linestyle="--", linewidth=1)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "metrics.png", dpi=150)
    plt.close(fig)

    for split, file in [("train", train_file), ("test", test_file)]:
        episode = pd.read_csv(file, usecols=INPUT_COLUMNS + OUTPUT_COLUMNS + ["time"])
        # Limit plotted points for readability; metrics still use every row.
        episode = episode.iloc[::max(1, len(episode) // 2000)]
        X = np.column_stack((np.ones(len(episode)), episode[INPUT_COLUMNS].to_numpy()))
        predictions = X @ weights
        fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex=True)
        for i, (ax, output) in enumerate(zip(axes.flat, OUTPUT_COLUMNS)):
            ax.plot(episode["time"], episode[output], label="Expert")
            ax.plot(episode["time"], predictions[:, i], label="Predicted", linestyle="--")
            ax.set(title=output, ylabel="N m" if i < 7 else "rad/s")
        axes[0, 0].legend()
        axes[-1, -1].set_visible(False)
        fig.suptitle(f"{split.capitalize()} episode: {file.name}")
        fig.supxlabel("Simulation time (s)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"{split}_predictions.png", dpi=150)
        plt.close(fig)
# CHAT GPT CODED THIS PART ENDS HERE
def main():
    files = sorted(DATA_DIR.glob("episode_*.csv"))

    # Keep whole episodes separate so neighbouring steps cannot leak into the test set.
    #CHAT GPT CODED THIS PART:
    np.random.default_rng(SEED).shuffle(files)
    num_test = int(np.ceil(len(files) * TEST_FRACTION))
    test_files = files[:num_test]
    train_files = files[num_test:]
    train = load_data(train_files)
    test = load_data(test_files)
    #END OF CHAT GPT CODED PART
    # A column of ones lets the model learn a constant offset (intercept).
    X_train = np.column_stack((np.ones(len(train)), train[INPUT_COLUMNS].to_numpy()))
    y_train = train[OUTPUT_COLUMNS].to_numpy()
    weights = np.linalg.pinv(X_train) @ y_train

    X_test = np.column_stack((np.ones(len(test)), test[INPUT_COLUMNS].to_numpy()))
    y_test = test[OUTPUT_COLUMNS].to_numpy()
    predictions = X_test @ weights
    error = y_test - predictions
    arm_rmse = np.sqrt(np.mean(error[:, :7]**2))
    wheel_rmse = np.sqrt(np.mean(error[:, 7:]**2))
    print(f"Training: {len(train_files)} episodes, {len(train)} rows")
    print(f"Testing: {len(test_files)} episodes, {len(test)} rows")
    print(f"Arm torque test RMSE: {arm_rmse:.4f} N m")
    print(f"Wheel command test RMSE: {wheel_rmse:.4f} rad/s")

    metrics = pd.DataFrame(index=OUTPUT_COLUMNS)
    for split, actual, predicted in [("train", y_train, X_train @ weights), ("test", y_test, predictions)]:
        squared_error = np.sum((actual - predicted)**2, axis=0)
        total_variation = np.sum((actual - actual.mean(axis=0))**2, axis=0)
        metrics[f"{split}_rmse"] = np.sqrt(squared_error / len(actual))
        # R² is undefined when the expert command is constant.
        metrics[f"{split}_r2"] = 1 - np.divide(
            squared_error, total_variation,
            out=np.full(len(OUTPUT_COLUMNS), np.nan), where=total_variation > 0,
        )
    print(metrics.round(4).to_string())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index_label="output")
    save_plots(metrics, weights, train_files[0], test_files[0])
    print(f"Saved metrics and graphs to {RESULTS_DIR}")

    np.savez(
        MODEL_PATH,
        weights=weights,
        input_columns=INPUT_COLUMNS,
        output_columns=OUTPUT_COLUMNS,
        train_episodes=[file.name for file in train_files],
        test_episodes=[file.name for file in test_files],
    )
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
