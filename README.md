# Random forest controllers (robotnik_RF)

Behaviour cloning of the Robotnik base + Kinova arm expert with random forests, trained on the
`data_randomized` dataset (2000 episodes) and evaluated in closed loop in PyBullet. All code is in
`robotnik_RF/`, and the paths below are relative to that folder.

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
for the wheel forest and 45 s for the extra arm forest on a laptop CPU. One wheel forest prediction takes about
5 ms and runs every 10 steps (`PREDICT_EVERY`), and one step of the V5.3-A arm takes about 0.5 ms, so E4 costs
about 1 ms per control step (measured on two episodes, laptop plugged in).


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

### Random-spawn test (`test_random_spawn.py`, `results/rf_run_4/eval_random_spawn/`)

E4 on the group's random-spawn test: the 1000 cases of Shivansh's `test_combined_random.py`
(branch `shivansh/linear_regression`, seed 42) in its **0.9–3.5 m** version, the same cases on which DAgger V6
reaches 90.0 % and the linear model 65.0 %. Spawn anywhere in x, y ∈ [−2, 2] m with any yaw, target 0.9–3.5 m
from the spawn, height 0.3–0.9 m. An episode stops at 1 cm error, after a 1.5 s stall or after 40 s, and
succeeds if the final error is ≤ 5 cm. Only the controller is swapped (RF wheels + DAgger V5.3-A arm, with our
own base goal and supervisor); the cases, the simulation setup and the stop rule are unchanged.

| | E4 |
|---|---|
| Success (≤ 5 cm) | **6.9 %** (69/1000, 95 % Wilson [5.5, 8.6]) |
| Mean / median final error | 0.80 / 0.81 m |
| Within 1 cm | 2.9 % |
| Mean episode time | 22.8 s |
| Base parked (not the 25 s timeout) | 33.2 % |

| | Spawn ≤ 0.35 m from origin (n = 30) | Spawn further away (n = 970) |
|---|---|---|
| Base parked | 100 % | 31.1 % |
| Success | 27 % | 6.3 % |

By spawn-to-target distance the success rate is flat: 4.0 % (0.9–1.5 m), 10.2 % (1.5–2.0 m), 6.2 % (2.0–2.5 m),
7.2 % (2.5–3.5 m).

**Why the random forest fails here.** The test asks for states the forests have never seen, and a forest
cannot extrapolate:

- *The training data does not cover these spawns.* All 2000 demonstrations start within ±0.25 m of the origin
  (any yaw) with targets 0.6–2.1 m away. The test spawns up to 2 m from the origin, with targets up to 3.5 m away.
- *The inputs contain world coordinates.* The extra wheel features keep `base_x`, `base_y`, `goal_x`,
  `goal_y`, the target and the arm state next to the goal-relative features, and the forest still splits on
  them. Outside the training range a tree returns the mean of the nearest leaf, so the wheel command is wrong
  and the base circles until the 25 s timeout (67 % of trials). This is the closed-loop version of the shift test
  in `cause1_extrapolation.png` (wheel R² 0.99 → 0.66 for a 2 m shift).
- *The arm is world-frame too.* V5.3-A uses world-frame inputs, so even when the base parks the arm succeeds
  in only 21 % (Ricktho's V5 likewise dropped to 6 % from 2 m spawns).

The evidence: success depends on **where the robot starts**, not on how far the target is. Near the origin,
inside the training region, the base parks in every trial; elsewhere it parks in only 31 %. The linear model
and DAgger V6 show the opposite pattern, because their inputs are robot-relative (body-frame / egocentric) and
look the same anywhere in the room. DAgger V6 was also trained on rollouts with spawns in ±2 m.

The fix would be a wheel forest on only the 5 goal-relative features (R² stays 0.987 under any shift) together
with an egocentric arm such as DAgger V6. This was not run.

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
- E1–E4 always start at the origin, inside the training region. For the random-spawn protocol of the
  linear model and DAgger V6, see the random-spawn test above.
- The 2.0–4.0 m version of the random-spawn test (Table II of the report) was not run for the forests.
- The supervisor switches from driving to reaching when the base is within **0.15 m** and 0.05 rad of
  its goal (or after 25 s). The expert uses 0.08 m. Once it switches, the wheels are set to zero.

### Robustness (`test_adversarial.py`, `results/rf_run_4/eval_adversarial/`, `eval_noise*/`)

**Disturbances.** E4 with the disturbances of Shivansh's `test_adversarial_robustness.py`, from the origin on the
same 25 targets in the training range, because E4 already fails from far spawns without any disturbance. An episode
stops at 1 cm or after a 2 s stall, and succeeds if the final error is ≤ 5 cm. This stop rule is stricter than the
one of `evaluate.py`, so the undisturbed 76 % is not comparable with the 94.3 % above: 22 of the 25 undisturbed
episodes came within 5 cm at some point.

| Test | Success (≤ 5 cm) | Median final error |
|---|---|---|
| No disturbance | 19/25 (76 %) | 22 mm |
| Shove: 30 N on the gripper for 0.25 s | 11/25 (44 %) | 63 mm |
| Target jump: 0.6–0.9 m at t = 2 s, new target in the training range | 10/25 (40 %) | 55 mm |
| Base displacement: 0.4 m after parking, only the base moves | 4/25 (16 %) | 742 mm |
| Payload: 0.5 kg on the gripper | 0/25 (0 %) | 1012 mm |

**Sensor noise.** `evaluate.py E4` on 100 random targets with `noise_flag` set to 1 in `robotnik_arm.json`. The
`_cov` values are used as standard deviations (`np.random.normal(0, value)`), and they also affect the measured
end-effector error.

| σ | Success | Base reached its goal |
|---|---|---|
| 0.0001 | 76/100 | 79/100 |
| 0.001 | 37/100 | 39/100 |

The arm succeeds in 113 of the 118 episodes in which the base arrives, so the wheel forest is the weak part. Offline,
1 mm of noise on `base_z`, which varies by only about 0.5 mm while driving, lowers the wheel R² from 0.99 to 0.90. A
noise-free run on the same 100 targets was not done.

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
| `test_random_spawn.py` | E4 on the group's random-spawn test (Shivansh's cases and stop rule); resumable, writes `results/rf_run_4/eval_random_spawn/` and prints the metrics of the report, including inference time |
| `test_adversarial.py` | E4 under four disturbances (shove, target jump, payload, base displacement) and without any; resumable, writes `results/rf_run_4/eval_adversarial/`. Needs `noise_flag` and `collision_enable` set to 0 |
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
pixi run -e ml python test_random_spawn.py      # E4 on the random-spawn test, ~5 h with 8 workers
pixi run -e ml python test_random_spawn.py 10   # only the first 10 cases (same cases as in the full run)
pixi run -e ml python test_adversarial.py       # E4 under the four disturbances, 125 episodes
pixi run -e ml python test_adversarial.py 2     # only the first 2 targets of every test
```

Inference time printed by `test_random_spawn.py` is measured inside the parallel run, so it is higher than
for a single process. Measure it with the laptop plugged in: on battery Windows throttles the CPU.

The settings at the top of `evaluate.py` choose the protocol: `SPLIT = "random"` with `NUM_EPISODES = 1000`
for the random targets, or `SPLIT = "validation"` for the validation targets, and `REACH_TOLERANCE` for the
success threshold. Each combination gets its own output folder.

`evaluate.py` appends a row per finished episode and skips episodes that are already in the CSV, so an
interrupted run continues where it stopped. Running it again with everything done just redraws the plots.
Do not edit `rf_sim.py` or `features.py` while an evaluation is running: every episode starts a fresh
process that imports them again.

**Models**: the four forests of `results/rf_run_4` (raw/extra × wheel/arm, about 115 MB) are in git, so
`rf_sim.py` and `evaluate.py` run after cloning once RoboEnv's `models/` folder is copied into
`robotnik_arm_task/`. They read the models from `RUN_DIR` (currently
`results/rf_run_4`). The E3/E4 arm also needs Ricktho's `robotnik_dagger_v5/trained_model_dagger_v5_3_A`.

**Not in git** (too large): the dataset and the cache. The dataset `data_randomized` (2000 expert
episodes, seed 42: targets at radius 0.9–1.8 m, base spawn within ±0.25 m with any yaw, randomized arm start
pose) was collected with `robotnik_arm_task/collect_data.py` on the `ricktho` and `shivansh/linear_regression`
branches. To retrain, get `data_randomized` from the group, then run `prepare_data.py` and `train_rf.py`.

**Other files used from this repo**: `robotnik_arm_task/robotnik_arm_control.py` and
`robotnik_arm_task/configs/robotnik_arm.json` (simulation setup), and Ricktho's
`robotnik_arm_task/robotnik_dagger_v5/` (`arm_v4.py`, `dagger_common.py`, `trained_model_dagger_v5_3_A/`)
for the DAgger arm. The robot `models/` folder comes from RoboEnv.
