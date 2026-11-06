# COMP0245 Final Task: Robotnik Mobile Manipulator (branch `ricktho`: MLP policy with DAgger)

A simulated Robotnik mobile manipulator (4-wheel skid-steer base + 7-DoF Kinova arm, PyBullet + Pinocchio) must drive to a 3D target and reach it with its gripper. A model-based expert controller solves the task. The goal is a learned model that produces the wheel commands and arm torques from measurements of the robot state.

This branch contains the **MLP policy trained with Dataset Aggregation (DAgger)**. The other group approaches are on their own branches: decoupled linear regression on `shivansh/linear_regression`, random forest on `hafizh/random_forest`.

<p align="center">
  <img src="robotnik_arm_task/robotnik_dagger_v5/media/v6_random_spawn_demo.gif" alt="Learned policy driving from a random spawn to a random target and reaching it" width="360">
</p>

**Result:** from a random spawn (x, y ∈ [−2, 2] m, any heading), with no expert involvement, the learned policy reaches **99.1 %** of 1000 random targets 2.0–4.0 m away within 5 cm (median error 16 mm), and 90.0 % when targets as close as 0.9 m are included. On the same 2.0–4.0 m cases, run with the linear model's own test script `test_combined_random.py`, the linear-regression model reaches 97.0 %. The method, the results and how to reproduce them are in [`robotnik_arm_task/robotnik_dagger_v5/README.md`](robotnik_arm_task/robotnik_dagger_v5/README.md).

## Repository layout

```text
Machine-Learning-Final-Task/
├── RoboEnv/                       simulation environment from the labs (Pixi manifest, simulation_and_control,
│                                  robot models/ and configs/)
├── robotnik_arm_task/
│   ├── robotnik_arm_task.py       expert simulation: one episode with the model-based controller (GUI or headless)
│   ├── robotnik_arm_control.py    expert controllers used by robotnik_arm_task.py
│   ├── collect_data.py            records the expert demonstrations -> data_randomized/
│   ├── data_randomized/           2000 demonstration episodes (not in git, created by collect_data.py)
│   ├── models, configs            links to RoboEnv/models and RoboEnv/configs
│   └── robotnik_dagger_v5/        the DAgger policy: code, trained model, evaluation, figures, demo video
│                                  (see its README; also links to RoboEnv/models and RoboEnv/configs)
└── README.md                      this file
```

## Setup

The simulation environment is included in `RoboEnv/`. Install [Pixi](https://pixi.sh), then create and activate the `ml` environment (Python 3.12 with PyTorch, PyBullet, Pinocchio and the bundled `simulation_and_control` package):

```bash
cd RoboEnv
pixi install -e ml        # first time only, downloads a few GB
pixi shell -e ml          # activates the environment in this terminal
cd ../robotnik_arm_task
```

Check the environment with `python -c "import torch, pybullet, simulation_and_control"`. The robot models and configs need no copying: `robotnik_arm_task/` and `robotnik_arm_task/robotnik_dagger_v5/` contain links to `RoboEnv/models` and `RoboEnv/configs` (on Windows, enable symlinks in git or copy the two folders there).

## Workflow

All commands run from `robotnik_arm_task/` in the terminal where `pixi shell -e ml` is active.

**1. Watch the expert.** One episode of the model-based controller, which drives to the target (red dot) and reaches it:
```bash
python robotnik_arm_task.py
```
Set `TARGET_EE_POS` at the top of the file to change the target.

**2. Record demonstrations.** 2000 expert episodes, each with:
- a random target (radius 0.9–1.8 m around the origin, height 0.3–0.9 m);
- a random base start (±0.25 m, any yaw);
- random arm start angles (±0.1 rad per joint).

The state and the expert's commands are logged every 10 ms.
```bash
python collect_data.py                      # -> data_randomized/
python collect_data.py --num-episodes 5 --output-dir /tmp/check   # quick test
```
The same seed reproduces the same tasks and start states (`targets.csv`). Tiny numerical differences in the physics can still make individual torque spikes differ between runs.

**3. Train and evaluate the DAgger policy.** Three DAgger rounds, training, and evaluation on 1000 random episodes, with progress bars and resumable stages:
```bash
cd robotnik_dagger_v5
bash run_v6_pipeline.sh
```

**4. Run the trained policy** in the GUI on a random target and spawn:
```bash
cd robotnik_dagger_v5
python run_robotnik_v6.py --random-target --random-spawn --seed $RANDOM
```

## Simulation variables

These are the variables in `robotnik_arm_task.py`:

| Variable | Meaning |
|---|---|
| `TARGET_EE_POS` | reach target (x, y, z) in the world frame, shown as a red dot |
| `base_pos`, `base_bearing` | base position (x, y, z) and heading (rad) |
| `q_mes`, `qd_mes` | arm joint angles (rad) and velocities (rad/s) |
| `wheel_cmds` | angular velocity command per wheel (rad/s) |
| `torque_arm` | torque command per arm joint (N·m), clipped to ±30 N·m by the controller |
| `torque_fingers` | gripper finger joints, held fixed |

Task of the assessment: a machine-learning model that produces (`wheel_cmds`, `torque_arm`) from measurements (`base_pos`, `base_bearing`, `q_mes`, `qd_mes`) and the target.
