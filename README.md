# COMP0245 Final Task, Group 10: Behaviour Cloning for a Mobile Manipulator
The link to the github repo for this group is at [github.com/parapara29/Machine-Learning-Final-Task](github.com/parapara29/Machine-Learning-Final-Task), each branch of the github repo is made into a folder in the final zip file.
A simulated Robotnik mobile manipulator (4-wheel skid-steer base + 7-DoF Kinova arm, PyBullet + Pinocchio) must drive to a 3D target and touch it with its gripper. A model-based expert controller solves the task. The goal is a learned model that produces the 4 wheel commands and 7 arm torques from the robot state and the target.

Each group member developed one approach on their own branch. This `main` branch holds only the shared starting point: the expert simulation and the data collector. Check out a branch to get its model, results and instructions.

## Branches

| Branch | Approach | Author |
|---|---|---|
| [`shivansh/linear_regression`](../../tree/shivansh/linear_regression) | Decoupled linear regression | Shivansh Chaudhary |
| [`ricktho`](../../tree/ricktho) | MLP policy trained with DAgger | Ricktho Sarkar |
| [`hafizh/random_forest`](../../tree/hafizh/random_forest) | Random forest controller | Hafizh Ilham |

### `shivansh/linear_regression`: decoupled linear regression

- **Model:** two least-squares linear models, one for the base (15 hand-engineered body-frame features: saturated goal distance, alignment gate, steering and docking terms) and one for the arm (25 features: joint angles and velocities, body-frame target offsets and their quadratic terms, low-height hinges). The arm model is trained with DART noise injection so it learns a stabilising spring–damper.
- **Analysis:** explains why a single naive linear model fails: regime mixing, world-coordinate sign flips, the "anti-spring" instability and the elbow-twist bifurcation.
- **Result:** 96.8 % within 5 cm and 83.6 % within 1 cm on 1000 random trials (spawn anywhere in x, y ∈ [−2, 2] m, target 2.0–4.0 m away), 0.025 ms per step.
- **Start with:** `robotnik_arm_task/robotnik_combined_model.py` (one episode), `robotnik_arm_task/test_combined_random.py` (the 1000-trial benchmark).

### `ricktho`: MLP policy with DAgger

- **Model:** three MLPs on egocentric (robot-frame) features: one for the wheels, and HOLD and REACH networks for the arm, selected by a latched navigation flag. 342,674 parameters, 0.21 ms per step.
- **Training:** Dataset Aggregation (DAgger) in three rounds, with the learner driving and the expert labelling every state it visits. This lets the policy recover from its own mistakes: behaviour cloning alone succeeds in 0 of 200 test episodes.
- **Result:** 99.1 % within 5 cm on the same 1000 random trials as the linear model (target 2.0–4.0 m away), and 90.0 % when targets as close as 0.9 m are included.
- **Includes:** the full pipeline (collection, training, evaluation), a demo video, the bundled `RoboEnv/` environment (`pixi shell -e ml`), and the head-to-head comparison with the linear model.
- **Start with:** `robotnik_arm_task/robotnik_dagger_v5/README.md`.

### `hafizh/random_forest`: random forest controller

- **Model:** a random forest for the wheels (raw or goal-relative "extra" inputs), paired with either a random forest arm or the DAgger MLP arm (V5.3-A from `ricktho`), giving four experiments E1–E4.
- **Result:** the random forest base with the DAgger arm (E4) succeeds in 94.3 % of 1000 random targets from the default spawn at the origin. A fully random forest controller succeeds in fewer than 1 % despite an offline R² of up to 0.95, which shows that offline accuracy does not imply closed-loop success.
- **Start with:** the branch README and `robotnik_RF/`.

### Head-to-head comparison

The linear model and the DAgger policy were run on the same 1000 cases of `test_combined_random.py` (spawn x, y ∈ [−2, 2] m with any heading, target 2.0–4.0 m from the spawn, height 0.3–0.9 m, success = final error ≤ 5 cm), on the same machine:

| | Linear regression | DAgger MLP |
|---|---:|---:|
| Success (≤ 5 cm) | 97.0 % | **99.1 %** |
| Within 1 cm | **84.2 %** | 23.3 % |
| Median / max final error | **10 mm** / 997 mm | 16 mm / **85 mm** |
| Inference per step | **0.03 ms** | 0.21 ms |

- The linear model's failures are all base-navigation timeouts at the closest targets (2.0–2.45 m).
- DAgger's failures are arm stalls just outside the 5 cm tolerance.
- The linear model is more precise when it succeeds and much faster to train and run.

The random forest results use a different test (fixed spawn at the origin), so they are not directly comparable. Details are in the `ricktho` branch (`robotnik_arm_task/robotnik_dagger_v5/README.md`, section "Comparison with the linear-regression model").

## What is on `main`

```text
robotnik_arm_task/
├── robotnik_arm_task.py      expert simulation: one episode with the model-based controller
├── robotnik_arm_control.py   expert controllers (base state machine, arm IK + PD)
├── collect_data.py           records expert demonstrations as CSV
└── configs/                  robot configuration files
```

**Run the expert:**
1. Use the RoboEnv environment from the labs (pixi, uv or conda). The `simulation_and_control` package must be importable.
2. Copy or link RoboEnv's `models/` and `configs/` folders into `robotnik_arm_task/`.
3. Run `python robotnik_arm_task/robotnik_arm_task.py`. A PyBullet window shows the robot driving to the red target and reaching it.

The key variables are at the top of `robotnik_arm_task.py`: `TARGET_EE_POS` (the target), `base_pos` and `base_bearing` (base pose), `q_mes` and `qd_mes` (arm joint angles and velocities), and the expert's commands `wheel_cmds` and `torque_arm`.

**Record demonstrations:** `python robotnik_arm_task/collect_data.py` writes `targets.csv` and one CSV per episode to `robotnik_arm_task/data/`. Edit `NUM_TARGETS`, `SEED` and `OUTPUT_DIR` at the top of the file. Every branch trains on a randomised version of this dataset (2000 episodes with random targets, base start poses and arm start angles); see each branch for its collector.

## Get a branch

```bash
git clone git@github.com:parapara29/Machine-Learning-Final-Task.git
cd Machine-Learning-Final-Task
git checkout ricktho                       # or shivansh/linear_regression, hafizh/random_forest
```
