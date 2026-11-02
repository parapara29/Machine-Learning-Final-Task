# Random forest controllers (robotnik_RF)

Behaviour cloning of the Robotnik base + Kinova arm expert with random forests, trained on the
`data_randomized` dataset (2000 episodes) and evaluated in closed loop in PyBullet.

The question is: **does a random forest work as a controller, and do extra input features help?**
The wheels are always a random forest. The arm is either a random forest or Ricktho's DAgger MLP
(`robotnik_arm_task/robotnik_dagger_v5`, branch `ricktho`), which gives a 2×2 experiment:

| | Arm | Wheels | Input features |
|---|---|---|---|
| **E1** | RF | RF | raw |
| **E2** | RF | RF | extra |
| **E3** | DAgger MLP | RF | raw |
| **E4** | DAgger MLP | RF | extra |

## Results

All numbers use Ricktho's episode split (1400 train / 300 validation / 300 test,
`episode_split.json`), so the RF and the MLP are trained and evaluated on the same episodes.

**Closed loop** (`results/rf_run_4/eval/`, first 100 validation episodes). Success = end-effector within
5 cm of the target for 30 steps in a row, within 20 s. This is the same rule as Ricktho's evaluation.

| | Success | Base reached its goal | Mean time to success |
|---|---|---|---|
| E1 | 0 % | 2 % | – |
| E2 | 0 % | 38 % | – |
| E3 | 3 % | 3 % | 13.6 s |
| **E4** | **97 %** | 98 % | 6.8 s |

**Offline test R²** (`results/rf_run_4/`, 300 test episodes):

| | Arm (7 joints) | Wheels (4) |
|---|---|---|
| raw | 0.63 – 0.93 | 0.89 – 0.90 |
| extra | 0.73 – 0.95 | 0.99 |

### Findings

1. **Extra wheel features are essential.** With raw inputs the base almost never reaches its goal
   (2–3 %). The raw inputs are absolute world coordinates, and a random forest cannot extrapolate from
   them. With goal-relative features (distance and heading to the goal) the base reaches it in 98 % of
   episodes, which is what takes E3 → E4 from 3 % to 97 %.
2. **The RF arm fails in closed loop despite a good offline R².** Small errors accumulate, the arm
   drifts into states that are not in the data, and the forest cannot extrapolate. It collapses within
   about 1.5 s (covariate shift). The DAgger MLP, trained on its own rollouts, does not have this problem.
3. **A failing arm also breaks the wheels.** E2 and E4 use the *same* wheel model, but the base reaches
   its goal in 38 % vs 98 % of episodes. The wheel inputs include the arm joint states and `base_z`, so a
   collapsing arm pushes them out of distribution. Wheel inputs restricted to navigation features would
   avoid this.
4. **Targets outside the training range fail.** Training targets have a radius of 0.9–1.8 m and a
   height of 0.3–0.9 m. Manual tests with targets outside this range (e.g. `[2, 1, 1]`) overshoot or drift.

### Caveats

- An episode stops as soon as the 5 cm rule is met, so these runs say nothing about 1 cm precision.
- 100 of the 300 validation episodes so far. Ricktho's full-MLP system reaches 280/300 (93 %) on all 300.
  Run with `NUM_EPISODES = None` for a direct comparison.
- The supervisor switches from driving to reaching when the base is within **0.15 m** and 0.05 rad of
  its goal (or after 25 s). The expert uses 0.08 m. Once it switches, the wheels are set to zero.

## Inputs

`features.py` builds the inputs from a dict of named columns. The same functions are used in
training (from the CSV table) and in simulation (from the live robot state).

- **raw (21)**: base x/y/z and bearing, 7 arm joint angles, 7 joint velocities, target x/y/z. The same
  inputs as the original MLP.
- **extra wheel (29)**: raw + goal x/y and bearing, distance to goal, sin/cos of the heading to the goal,
  sin/cos of the final heading error.
- **extra arm (49)**: raw + `nav_done`, end-effector x/y/z, target − end-effector, sin/cos of each joint
  angle, and `q_init − q` (the expert holds the randomized start pose during navigation).

The wheel model is trained only on rows before `nav_done` (the base is not moving afterwards).

## Files

| File | What it does |
|---|---|
| `prepare_data.py` | Reads the 2000 CSVs once into `data_randomized_cache.npz` (adds `episode` and `nav_done`) |
| `features.py` | Raw and extra input features and their names |
| `train_rf.py` | Trains wheel + arm forests for raw and extra; saves models, R², feature importance and plots to `results/rf_run_N/` |
| `rf_sim.py` | Runs one episode in PyBullet with the GUI (`ARM_MODEL` = `"rf"`, `"dagger"` or `"expert"`) |
| `evaluate.py` | Runs E1–E4 headless on the validation episodes in parallel; writes per-episode CSVs and plots to `results/rf_run_4/eval/` |
| `episode_split.json` | Ricktho's train / validation / test episode IDs |

## How to run

From `robotnik_RF/`, in the RoboEnv pixi environment. E3/E4 need torch, so use the `ml` environment
for the simulation and evaluation.

```bash
pixi run python prepare_data.py           # once; needs ../data_randomized/episode_XXXX.csv
pixi run python train_rf.py               # ~3 min, creates results/rf_run_N
pixi run -e ml python rf_sim.py           # one episode with the GUI (target set by TARGET_EE_POS)
pixi run -e ml python evaluate.py         # E1–E4, ~40 min for 100 episodes
pixi run -e ml python evaluate.py E3 E4   # only some experiments
```

`evaluate.py` appends a row per finished episode and skips episodes that are already in the CSV, so an
interrupted run continues where it stopped. Running it again with everything done just redraws the plots.

**Not in git** (too large): the dataset, the cache and the `.joblib` models. Get `data_randomized` from
the group, then run `prepare_data.py` and `train_rf.py`. `rf_sim.py` and `evaluate.py` read the models
from `RUN_DIR` (currently `results/rf_run_4`). Runs 1–3 used an older random split and are kept only for
their plots.
