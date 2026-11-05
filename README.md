# Random forest controllers (robotnik_RF)

Behaviour cloning of the Robotnik base + Kinova arm expert with random forests, trained on the
`data_randomized` dataset (2000 episodes) and evaluated in closed loop in PyBullet.

The question is: **does a random forest work as a controller, and do extra input features help?**
The wheels are always a random forest. The arm is either a random forest or Ricktho's DAgger MLP
(`robotnik_arm_task/robotnik_dagger_v5`, version V5.3-A, branch `ricktho`), which gives a 2×2 experiment:

| | Arm | Wheels | Input features |
|---|---|---|---|
| **E1** | RF | RF | raw |
| **E2** | RF | RF | extra |
| **E3** | DAgger MLP | RF | raw |
| **E4** | DAgger MLP | RF | extra |

## Results

The forests are trained on Ricktho's episode split (1400 train / 300 validation / 300 test,
`episode_split.json`), so the RF and the MLP are trained and evaluated on the same episodes.

**Closed loop** (`results/rf_run_4/eval_random_5cm/`): 1000 new random targets from the
demonstration target distribution (radius 0.9–1.8 m, any angle, height 0.3–0.9 m), seed 2026, so none of
them are in the training data. The robot starts at the origin with the default arm pose. Success = end-effector
within 5 cm of the target for 30 steps in a row, within 20 000 steps (20 s). This is the same rule as
Ricktho's evaluation.

| | Success | Base reached its goal | Median final error | Mean time to success |
|---|---|---|---|---|
| E1 | 0.0 % | 2.0 % | 1.61 m | – |
| E2 | 0.6 % | 40.4 % | 1.10 m | 10.2 s |
| E3 | 1.7 % | 2.7 % | 3.28 m | 7.9 s |
| **E4** | **94.3 %** (95 % Wilson [92.7, 95.6]) | 97.7 % | 0.047 m | 6.6 s |

Use the median, not the mean, for the error: a collapsing arm can launch the robot, which gives final
errors of hundreds of metres (the mean of E2 is 13 m).

**Tolerance study** (`results/rf_run_4/eval_2cm/`, `eval_1cm/`, first 100 validation targets): E4 succeeds
in 97 % at 5 cm, 66 % at 2 cm and 34 % at 1 cm.

**Offline test R²** (`results/rf_run_4/`, 300 test episodes):

| | Arm (7 joints) | Wheels (4) |
|---|---|---|
| raw | 0.63 – 0.93 | 0.89 – 0.90 |
| extra | 0.73 – 0.95 | 0.99 |

**Training and inference**: 50 trees, at least 10 samples per leaf, every 10th row. Training takes about 12 s
for the wheel forest and 45 s for the extra arm forest on a laptop CPU. One prediction takes about 3.5 ms per
forest; the wheel forest predicts every 10 steps (`PREDICT_EVERY`) and holds the command in between.

### Findings

1. **Extra wheel features are essential.** With raw inputs the base almost never reaches its goal
   (2–3 %). The raw inputs are absolute world coordinates, and a random forest cannot extrapolate from
   them. With goal-relative features (distance and heading to the goal) the base reaches it in 97.7 % of
   trials, which is what takes E3 → E4 from 1.7 % to 94.3 %.
2. **The RF arm fails in closed loop despite a good offline R².** Small errors accumulate, the arm
   drifts into states that are not in the data, and the forest cannot extrapolate. It collapses within
   about 1.5 s (covariate shift). E1 and E2 together succeed in 6 of 2000 trials. The DAgger MLP, trained
   on its own rollouts, does not have this problem.
3. **A failing arm also breaks the wheels.** E2 and E4 use the *same* wheel model, but the base reaches
   its goal in 40.4 % vs 97.7 % of trials. The wheel inputs include the arm joint states and `base_z`, so a
   collapsing arm pushes them out of distribution. Wheel inputs restricted to the 5 goal-relative features
   would avoid this.
4. **Targets outside the training range fail.** Training targets have a radius of 0.9–1.8 m and a
   height of 0.3–0.9 m. Manual tests with targets outside this range (e.g. `[2, 1, 1]`) overshoot or drift.

### Data analysis (`analyze_rf.py`, `results/analysis/`)

Three figures that explain why a high offline R² does not give closed-loop success:

- `cause1_extrapolation.png`: shifting the whole scene by up to 2 m in x (the expert command does not
  change) drops the wheel R² from 0.89 to 0.64 (raw) and from 0.99 to 0.66 (extra). A forest on only the
  5 goal-relative features stays at 0.987 for every shift.
- `cause2_hold_pose.png`: during navigation the expert holds the randomized start pose `q_init`, not the
  nominal pose (median deviation 0.007 rad vs 0.045 rad). Adding `q_init − q` raises the arm R² by up to 0.08.
- `cause3_temporal.png`: joint states from 0.1 s and 0.3 s earlier do not help (Joint 6: 0.728 vs 0.729);
  the errors sit in short torque transients at phase changes.

### Caveats

- An episode stops as soon as the 5 cm rule is met, so the median final error (0.047 m) sits just inside
  the 5 cm band and says nothing about 1 cm precision. See the tolerance study for that.
- The robot always starts at the origin. These results are not directly comparable with the random-spawn
  protocol (spawn anywhere in x, y ∈ [−2, 2] m) used for the linear model and DAgger V6.
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
| `analyze_rf.py` | Data analysis figures for the report (shift test, hold pose, temporal inputs) in `results/analysis/` |
| `rf_sim.py` | Runs one episode in PyBullet with the GUI (`ARM_MODEL` = `"rf"`, `"dagger"` or `"expert"`) |
| `evaluate.py` | Runs E1–E4 headless in parallel; writes per-episode CSVs and plots to `results/rf_run_4/eval_<split>_<tol>cm/` |
| `episode_split.json` | Ricktho's train / validation / test episode IDs |

## How to run

From `robotnik_RF/`, in the RoboEnv pixi environment. E3/E4 need torch, so use the `ml` environment
for the simulation and evaluation.

```bash
pixi run python prepare_data.py           # once; needs ../data_randomized/episode_XXXX.csv
pixi run python train_rf.py               # ~3 min, creates results/rf_run_N
pixi run python analyze_rf.py             # report figures in results/analysis/
pixi run -e ml python rf_sim.py           # one episode with the GUI (target set by TARGET_EE_POS)
pixi run -e ml python evaluate.py         # E1–E4, about 2 h per experiment for 1000 targets
pixi run -e ml python evaluate.py E3 E4   # only some experiments
```

The settings at the top of `evaluate.py` choose the protocol: `SPLIT = "random"` with `NUM_EPISODES = 1000`
for the random targets, or `SPLIT = "validation"` for the validation targets, and `REACH_TOLERANCE` for the
success threshold. Each combination gets its own output folder.

`evaluate.py` appends a row per finished episode and skips episodes that are already in the CSV, so an
interrupted run continues where it stopped. Running it again with everything done just redraws the plots.
Do not edit `rf_sim.py` or `features.py` while an evaluation is running: every episode starts a fresh
process that imports them again.

**Models**: the four forests of `results/rf_run_4` (raw/extra × wheel/arm, about 115 MB) are in git, so
`rf_sim.py` and `evaluate.py` run right after cloning. They read the models from `RUN_DIR` (currently
`results/rf_run_4`). The E3/E4 arm also needs Ricktho's `robotnik_dagger_v5/trained_model_dagger_v5_3_A`.

**Not in git** (too large): the dataset and the cache. The dataset `data_randomized` (2000 expert
episodes, seed 42: targets at radius 0.9–1.8 m, base spawn within ±0.25 m with any yaw, randomized arm start
pose) was collected with `robotnik_arm_task/collect_data.py` on the `ricktho` and `shivansh/linear_regression`
branches. To retrain, get `data_randomized` from the group, then run `prepare_data.py` and `train_rf.py`.

**Other files used from this repo**: `robotnik_arm_task/robotnik_arm_control.py` and
`robotnik_arm_task/configs/robotnik_arm.json` (simulation setup), and Ricktho's
`robotnik_arm_task/robotnik_dagger_v5/` (`arm_v4.py`, `dagger_common.py`, `trained_model_dagger_v5_3_A/`)
for the DAgger arm. The robot `models/` folder comes from RoboEnv.
