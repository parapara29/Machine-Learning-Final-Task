import os
os.environ["ROBOENV_HEADLESS"] = "1"

import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import rf_sim

SPLIT_FILE = Path(__file__).parent / "episode_split.json"
TARGETS_CSV = Path(__file__).parent.parent / "data_randomized" / "targets.csv"
EVAL_DIR = rf_sim.RUN_DIR / "eval"

EXPERIMENTS = {
    "E1": ("rf", "raw"),
    "E2": ("rf", "extra"),
    "E3": ("dagger", "raw"),
    "E4": ("dagger", "extra"),
}
SPLIT = "validation"      
NUM_EPISODES = 100        
REACH_TOLERANCE = 0.05
MAX_STEPS = 20000
WORKERS = max(1, os.cpu_count() // 2)

FIELDS = ["episode", "target_x", "target_y", "target_z", "success",
          "final_error", "best_error", "time", "nav_time", "stop"]
COLORS = {"raw": "#2a78d6", "extra": "#eb6834"}
INK, MUTED = "#1f1f1e", "#52514e"


def load_episodes():
    ids = json.loads(SPLIT_FILE.read_text())[SPLIT][:NUM_EPISODES]
    table = np.genfromtxt(TARGETS_CSV, delimiter=",", names=True)
    targets = {int(row["episode"]): [row["target_x"], row["target_y"], row["target_z"]] for row in table}
    return [(episode, targets[episode]) for episode in ids]


def run_episode(job):
    arm_model, feature_set, episode, target = job
    if arm_model == "dagger":
        import torch
        torch.set_num_threads(1)   
    try:
        result = rf_sim.main(np.array(target), arm_model, feature_set, verbose=False,
                             reach_tolerance=REACH_TOLERANCE, max_steps=MAX_STEPS, stop_on_stall=False)
    except Exception as error:
        result = {"success": 0, "final_error": np.nan, "best_error": np.nan,
                  "time": np.nan, "nav_time": np.nan, "stop": f"Error: {error!r}"}
    return {"episode": episode, "target_x": target[0], "target_y": target[1], "target_z": target[2], **result}


def read_rows(path):
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def run_experiment(name, episodes):
    arm_model, feature_set = EXPERIMENTS[name]
    path = EVAL_DIR / f"{name}.csv"
    done = {int(row["episode"]) for row in read_rows(path)}
    jobs = [(arm_model, feature_set, episode, target) for episode, target in episodes if episode not in done]
    print(f"\n[{name}] arm {arm_model}, wheels RF {feature_set}: "
          f"{len(jobs)} episodes to run, {len(done)} already done, {WORKERS} workers")
    if not jobs:
        return

    start = time.perf_counter()
    new_file = not path.exists()
    with open(path, "a", newline="") as f, Pool(WORKERS, maxtasksperchild=1) as pool:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for k, row in enumerate(pool.imap_unordered(run_episode, jobs), 1):
            writer.writerow(row)
            f.flush()
            status = "OK  " if row["success"] else "FAIL"
            print(f"  {k:3d}/{len(jobs)} episode {row['episode']:4d} {status} "
                  f"best {row['best_error']:.3f} m | {row['stop']}")
    print(f"[{name}] finished in {(time.perf_counter() - start) / 60:.1f} min")


def load_results(name):
    rows = read_rows(EVAL_DIR / f"{name}.csv")
    r = {key: np.array([float(row[key]) for row in rows]) for key in FIELDS[:-1]}
    r["success"] = r["success"].astype(bool)
    r["radius"] = np.hypot(r["target_x"], r["target_y"])
    return r


# //plotting
def plot_episodes(name, r, path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    arm_model, feature_set = EXPERIMENTS[name]
    ok = r["success"]

    ax = axes[0]
    ax.hist(r["final_error"], bins=30, color=COLORS[feature_set], edgecolor="white")
    ax.axvline(REACH_TOLERANCE, color=INK, linestyle="--", linewidth=1.5, label=f"Threshold ({REACH_TOLERANCE} m)")
    ax.set(xlabel="Final EE distance error (m)", ylabel="Number of episodes",
           title=f"Final error (success {ok.mean():.0%})")
    ax.legend(frameon=False)

    ax = axes[1]
    ax.scatter(r["target_x"][ok], r["target_y"][ok], s=18, marker="o", color=COLORS["raw"],
               label=f"Success (n={ok.sum()})")
    ax.scatter(r["target_x"][~ok], r["target_y"][~ok], s=30, marker="x", color=COLORS["extra"],
               label=f"Fail (n={(~ok).sum()})")
    ax.scatter([0], [0], s=70, marker="^", color=INK, label="Robot start")
    ax.set(xlabel="Target X (m)", ylabel="Target Y (m)", title="Top-down target map", aspect="equal")
    ax.legend(frameon=False, loc="upper right", fontsize=8)

    ax = axes[2]
    points = ax.scatter(r["target_z"], r["best_error"], c=r["radius"], cmap="viridis", s=18)
    ax.axhline(REACH_TOLERANCE, color=INK, linestyle="--", linewidth=1.5)
    ax.set(xlabel="Target height Z (m)", ylabel="Best EE distance error (m)", title="Best error vs target height")
    fig.colorbar(points, ax=ax, label="Target radius (m)")

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(f"{name}: arm {arm_model}, wheels RF {feature_set} ({len(ok)} {SPLIT} episodes)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_success_comparison(results, path):
    names = list(results)
    rates = [results[name]["success"].mean() * 100 for name in names]
    colors = [COLORS[EXPERIMENTS[name][1]] for name in names]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(np.arange(len(names)), rates, width=0.6, color=colors, edgecolor="white", linewidth=2)
    for x, (name, rate) in enumerate(zip(names, rates)):
        ax.text(x, rate + 1.5, f"{rate:.0f}%  ({results[name]['success'].sum()}/{len(results[name]['success'])})",
                ha="center", fontsize=9, color=INK)
    labels = [f"{name}\narm {EXPERIMENTS[name][0]}\nwheel {EXPERIMENTS[name][1]}" for name in names]
    ax.set_xticks(np.arange(len(names)), labels)
    ax.set(ylabel="Success rate (%)", ylim=(0, 105),
           title=f"Closed-loop success (< {REACH_TOLERANCE * 100:.0f} cm for {rf_sim.HOLD_STEPS} steps)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize():
    results = {name: load_results(name) for name in EXPERIMENTS if (EVAL_DIR / f"{name}.csv").exists()}
    print(f"\n{'':4s} {'arm':7s} {'wheel':6s} {'success':>14s} {'median best err':>16s} {'mean time ok':>13s}")
    for name, r in results.items():
        arm_model, feature_set = EXPERIMENTS[name]
        ok = r["success"]
        mean_time = f"{r['time'][ok].mean():.1f} s" if ok.any() else "-"
        print(f"{name:4s} {arm_model:7s} {feature_set:6s} {ok.sum():4d}/{len(ok):<4d}({ok.mean():4.0%}) "
              f"{np.nanmedian(r['best_error']):14.3f} m {mean_time:>13s}")
        plot_episodes(name, r, EVAL_DIR / f"{name}_episodes.png")
    if results:
        plot_success_comparison(results, EVAL_DIR / "success_comparison.png")
        print(f"\nSaved results and plots to {EVAL_DIR}")


if __name__ == "__main__":
    # python evaluate.py         -> all experiments
    # python evaluate.py E3 E4   -> only these
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    episodes = load_episodes()
    for name in sys.argv[1:] or list(EXPERIMENTS):
        run_experiment(name, episodes)
    summarize()
