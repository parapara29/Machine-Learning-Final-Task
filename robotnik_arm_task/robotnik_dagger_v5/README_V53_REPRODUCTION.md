# Robotnik Mobile Manipulator — DAgger V5.3-A

**Current best evaluated checkpoint:** `trained_model_dagger_v5_3_A`  
**Validation:** 280/300 (93.33%) successes, compared with V5's 263/300 (87.67%).  
**Status:** research simulation; final untouched test evaluation is still required. This README documents the current workflow, including known unresolved failures.

## 0. Quick repository map

If you are reading this repository for the first time, these files are the main path through the work:

- `robotnik_dagger_collect.py`: simulation loop that runs the mobile base + arm policy and logs trajectories.
- `dagger_common.py`: shared feature contract and runtime policy loader used by training and deployment.
- `train_dagger.py`: main imitation-learning training script that builds the wheel and torque models.
- `train_arm_v5.py`: phase-specific arm training helper used for hold/reach behavior separation.
- `robotnik_arm_control.py`: low-level controller that handles base navigation and the 7-DoF arm reach law.
- `run_robotnik_v53.py`: convenience runner for a single custom target or randomized rollout.
- `evaluate_robotnik_v53.py`: metrics and trajectory evaluator used to benchmark model success.
- `README_V53_REPRODUCTION.md`: this guide to reproduction and diagnostics.

The overall pattern is: collect trajectories -> extract shared features -> train policies -> deploy -> evaluate success.

## 1. What this project does

A simulated Robotnik mobile manipulator drives a four-wheel base toward a target-dependent standoff pose, aligns with the target, and uses a seven-DoF torque-controlled arm to reach a 3D target. PyBullet simulates the robot and Pinocchio supplies forward kinematics. The learned policy imitates a stateful expert controller, then improves through Dataset Aggregation (DAgger).

The deployed policy has **three neural networks**:

| Network | Input | Hidden layers | Output |
|---|---:|---|---:|
| Wheel MLP (behaviour-cloned once, `phase_observable_raw_v4`; never DAgger-retrained) | 38 current features | Read exact architecture from `wheel_mlp_metadata.json` | 4 wheel angular velocities |
| Arm HOLD MLP | 66 features | 256 → 256 → 128, ReLU, dropout 0.1 | 7 arm torques |
| Arm REACH MLP | 66 features | 256 → 256 → 128, ReLU, dropout 0.1 | 7 arm torques |

The 66 arm features are **38 current raw observations + 14 arm position/velocity values at lag 10 steps + 14 at lag 30 steps**. No feature scaling is used. The `navigation_complete` observation selects HOLD versus REACH. V5.3-A arm checkpoints intentionally retain the filenames `arm_hold_v4.pth` and `arm_reach_v4.pth` for loader compatibility; they are *not* V4-trained weights.

## 2. Folder layout

The commands below assume you run them from the `robotnik_dagger_v5/` directory, as in the existing project:

```text
Machine-Learning-Final-Task/
├── data_randomized/
│   ├── targets.csv
│   └── [original demonstration episode CSVs]
└── robotnik_dagger_v5/
    ├── robotnik_dagger_collect.py
    ├── dagger_common.py
    ├── arm_v4.py
    ├── robotnik_arm_control.py
    ├── robotnik_arm.json
    ├── train_dagger.py
    ├── train_arm_v5.py
    ├── make_combined_dirs.sh             # rebuilds arm_dagger_v5_3_combined / _v5_4_combined
    ├── arm_dagger_v5/ arm_dagger_v5_3/   # DAgger rounds 1 and 2 (in git)
    ├── arm_dagger_v5_3_randomized/       # DAgger round 3, 26 GB, NOT in git (local only)
    ├── trained_model_dagger_v5_4/        # V5.4 experiment (round 3), worse than V5.3-A
    └── trained_model_dagger_v5_3_A/      # BEST VALIDATION MODEL (wheel MLP + episode split live here)
        ├── wheel_mlp.pth
        ├── wheel_mlp_metadata.json
        ├── arm_hold_v4.pth
        ├── arm_hold_v4_metadata.json
        ├── arm_reach_v4.pth
        ├── arm_reach_v4_metadata.json
        └── episode_split.json
```

**The `.pth` checkpoints are essential.** The source archive alone does not contain the trained V5.3-A weights. Keep the trained model directory produced on your machine; the checkpoint cannot be reconstructed from this README. The Robotnik URDF and other assets referenced by `robotnik_arm.json` must also exist at the configured paths.

## 3. Environment setup

The project uses the existing **Pixi** environment named `ml` (with simulation packages, PyTorch and pandas). From the project directory:

```bash
pixi run -e ml python -c "import torch, pandas, numpy, pybullet, pinocchio; print('Core imports OK')"
pixi run -e ml python robotnik_dagger_collect.py --help
```

If `pixi` is not found, install Pixi using its official instructions and use the repository's Pixi manifest/lockfile. If the `ml` environment does not exist, restore the original project `pixi.toml` and `pixi.lock` or create the appropriate environment; **do not assume the source ZIP includes them**. The model and PyBullet configuration also require the project's `simulation_and_control` package to be importable.

**Important version check:** the archived V5 collector and the later collector used for V5.3-A differ. The commands here require the later collector that accepts `--mode deploy`, `--wheel-source`, `--arm-source`, `--save-trajectory`, and `--gui`. Confirm using `--help`. For GUI use on Linux, run inside a desktop session with a working display server.

## 4. Launch the best model in the PyBullet GUI

### 4.1 Fixed target deployment (single validation episode)

**Recommended first command** (one known validation episode; the checkpoint must exist):

```bash
pixi run -e ml python robotnik_dagger_collect.py \
  --mode deploy \
  --model-dir trained_model_dagger_v5_3_A \
  --targets-csv ../data_randomized/targets.csv \
  --episode-ids 173 \
  --output-dir demo_v53_gui \
  --save-trajectory \
  --gui
```

This is a **diagnostic episode**, not a guaranteed success under V5.3-A: episode 173 is one of the known navigation regressions. To watch an episode that succeeded under both models, use an ID from the paired successful set (e.g. choose one confirmed from `v5_vs_v53_episode_comparison.csv`). Do not infer success from the GUI alone; inspect the printed `stop_reason` and `deploy_summary.csv`.

To test a few known V5.3-A *fixed* validation episodes, use:

```bash
pixi run -e ml python robotnik_dagger_collect.py \
  --mode deploy \
  --model-dir trained_model_dagger_v5_3_A \
  --targets-csv ../data_randomized/targets.csv \
  --episode-ids 59 170 188 \
  --output-dir demo_v53_fixed \
  --save-trajectory --gui
```

### 4.2 Randomized target + randomized spawn for stress testing

The V5.3 custom deployment runner now supports **random target sampling** and **random initial bot placement** in the same simulation loop. This is useful for checking robustness beyond the fixed CSV episodes.

Fixed target:

```bash
pixi run -e ml python run_robotnik_v53.py \
  --target 0.5 1.2 0.6 \
  --model-dir trained_model_dagger_v5_3_A \
  --output-dir custom_target_demo \
  --episode-id 1 --gui
```

Random target, random spawn:

```bash
pixi run -e ml python run_robotnik_v53.py \
  --random-target \
  --target-x-range 0.4 1.2 \
  --target-y-range -1.0 1.0 \
  --target-z-range 0.2 1.2 \
  --random-spawn \
  --spawn-x-range -1.5 1.5 \
  --spawn-y-range -1.5 1.5 \
  --spawn-yaw-range -3.14159 3.14159 \
  --model-dir trained_model_dagger_v5_3_A \
  --output-dir custom_target_demo_random \
  --episode-id 101 --seed 7 --gui
```

You can also pin one exact starting pose and still randomize the target:

```bash
pixi run -e ml python run_robotnik_v53.py \
  --random-target \
  --spawn-position 0.0 0.0 0.0 \
  --model-dir trained_model_dagger_v5_3_A \
  --output-dir custom_target_demo_spawn_fixed \
  --episode-id 202 --seed 11 --gui
```

If you are using a headless Linux machine, **omit `--gui`**. The simulation still runs and writes trajectory files. The `--gui` switch is supported by the supplied collector versions, but graphical display availability depends on your system.

## 5. Run a batch evaluation

Run without GUI for speed:

```bash
pixi run -e ml python robotnik_dagger_collect.py \
  --mode deploy \
  --model-dir trained_model_dagger_v5_3_A \
  --targets-csv ../data_randomized/targets.csv \
  --episode-ids 173 265 342 1932 \
  --output-dir v53_navigation_diagnostics \
  --save-trajectory
```

Output includes `deploy_summary.csv` and, with `--save-trajectory`, `deploy_000173.csv` and similarly named episode trajectories. If a velocity jump is detected, an `anomaly_*.json` file is also written. **Some collector versions append to an existing summary**; use a fresh `--output-dir` for each experiment, or check for duplicate episode rows before computing success rates.

To evaluate the entire validation split reproducibly, read the episode IDs from `trained_model_dagger_v5_3_A/episode_split.json` and supply **only** its `validation` IDs. For example:

```bash
pixi run -e ml python - <<'PY'
import json, subprocess
from pathlib import Path
split = json.loads(Path('trained_model_dagger_v5_3_A/episode_split.json').read_text())
ids = list(map(str, split['validation']))
cmd = ['python', 'robotnik_dagger_collect.py', '--mode', 'deploy',
       '--model-dir', 'trained_model_dagger_v5_3_A',
       '--targets-csv', '../data_randomized/targets.csv',
       '--output-dir', 'eval_v53_validation_fresh', '--episode-ids', *ids]
subprocess.run(cmd, check=True)
PY
```

The outer `pixi run -e ml` provides the environment for the subprocess. Run the same ID list, simulation configuration, safety thresholds, and random seed for comparisons. Preserve the original split and never train on validation/test IDs.

Quick summary check:

```bash
pixi run -e ml python - <<'PY'
import pandas as pd
p='eval_v53_validation_fresh/deploy_summary.csv'
df=pd.read_csv(p).drop_duplicates('episode',keep='last')
print('Episodes:',len(df),'Successes:',df['success'].sum())
print('Success rate:',df['success'].mean())
print(df.loc[~df['success'], 'stop_reason'].value_counts())
print('Anomaly episodes:',df['velocity_jump_anomaly'].notna().sum())
PY
```

## 6. How the training was done

**Original demonstrations:** 2,000 episodes split 1,400 training / 300 validation / 300 test, with split IDs saved in `trained_model_dagger_v5_3_A/episode_split.json`.

**V5 mixed-arm DAgger example** (training IDs only):

```bash
pixi run -e ml python - <<'PY'
import json, subprocess
from pathlib import Path
split = json.loads(Path('trained_model_dagger_v5_3_A/episode_split.json').read_text())
ids = split['train']
cmd = [
    'python', 'robotnik_dagger_collect.py',
    '--mode', 'collect',
    '--wheel-source', 'expert',
    '--arm-source', 'mix',
    '--model-dir', 'trained_model_dagger_v5_3_A',
    '--targets-csv', '../data_randomized/targets.csv',
    '--episode-ids', *map(str, ids),
    '--beta', '0.5',
    '--mix-interval-steps', '50',
    '--output-dir', 'arm_dagger_v5',
    '--seed', '42',
]
subprocess.run(cmd, check=True)
PY
```

> **`--model-dir` in a collection command is the learner being corrected**, i.e. the checkpoint produced by the *previous* round. The `trained_model_dagger_v5_3_A` value shown in these commands cannot be the learner that actually generated the V5 and V5.3 rollouts (that checkpoint did not exist yet); substitute the checkpoint of the preceding round.

For V5.3-A, the subsequent collection changed to **expert wheels, learner arm, β=0.0** (β has no effect when `--arm-source learner`), on the **fixed targets of `targets.csv` with the default spawn**. These are the 110 rollouts in `arm_dagger_v5_3/`:

```bash
pixi run -e ml python - <<'PY'
import json, subprocess
from pathlib import Path
split = json.loads(Path('trained_model_dagger_v5_3_A/episode_split.json').read_text())
ids = split['train']
cmd = [
    'python', 'robotnik_dagger_collect.py',
    '--mode', 'collect',
    '--wheel-source', 'expert',
    '--arm-source', 'learner',
    '--model-dir', 'trained_model_dagger_v5',   # previous-round learner
    '--targets-csv', '../data_randomized/targets.csv',
    '--output-dir', 'arm_dagger_v5_3',
    '--episode-ids', *map(str, ids),
    '--beta', '0.0',
    '--seed', '42',
]
subprocess.run(cmd, check=True)
PY
```

V5.3-A was then trained on the **union of both rounds**: `arm_dagger_v5_3_combined/` holds symlinks to the 51 rollouts of `arm_dagger_v5/` and the 110 of `arm_dagger_v5_3/` (160 belong to training IDs; one held-out rollout is excluded). Build it with `bash make_combined_dirs.sh`. The wheel weights are copied unchanged from `trained_model_dagger_v5_3_A` (the 38-feature raw-input wheel model, `phase_observable_raw_v4`).

```bash
pixi run -e ml python train_arm_v5.py \
  --data-dir ../data_randomized \
  --wheel-model-dir trained_model_dagger_v5_3_A \
  --dagger-dir arm_dagger_v5_3_combined \
  --output-dir trained_model_dagger_v5_3_A_repro \
  --epochs 100 --patience 15 --seed 42 \
  --recovery-radius 0.35 --recovery-weight 4 \
  --smoothness 0.01
```

This reproduces the **training configuration**, not necessarily the exact checkpoint, unless the same original dataset, episode selection, rollout CSVs, software version, random seeds and hardware/runtime behaviour are used. Preserve those original files for exact reproducibility.

### V5.4: third DAgger round (experiment, not adopted)

Round 3 rolled out V5.3-A (expert wheels, learner arm, β=0) on 1400 training-ID episodes with a **random target and random spawn**. With no explicit ranges, this collector version draws targets from the per-axis ranges of `targets.csv` (x, y ∈ [−1.8, 1.8] m, z ∈ [0.3, 0.9] m) and the spawn from the demonstration offsets (x, y ∈ [−0.25, 0.25] m, any yaw):

```bash
pixi run -e ml python - <<'PY'
import json, subprocess
from pathlib import Path
ids = json.loads(Path('trained_model_dagger_v5_3_A/episode_split.json').read_text())['train']
subprocess.run(['python', 'robotnik_dagger_collect.py', '--mode', 'collect',
    '--wheel-source', 'expert', '--arm-source', 'learner', '--beta', '0.0',
    '--model-dir', 'trained_model_dagger_v5_3_A', '--random-target', '--random-spawn',
    '--output-dir', 'arm_dagger_v5_3_randomized', '--episode-ids', *map(str, ids), '--seed', '42'], check=True)
PY
bash make_combined_dirs.sh            # arm_dagger_v5_4_combined = rounds 1-3 (1561 rollouts)
pixi run -e ml python train_arm_v5.py --data-dir ../data_randomized \
  --wheel-model-dir trained_model_dagger_v5_3_A --dagger-dir arm_dagger_v5_4_combined \
  --output-dir trained_model_dagger_v5_4 --epochs 100 --patience 15 --seed 42 \
  --recovery-radius 0.35 --recovery-weight 4 --smoothness 0.01
bash run_v54_evaluation.sh            # V5.4 on the validation targets + both models on 100 random episodes
pixi run -e ml python plot_v54_results.py   # figures/v54_*.pdf + figures/v54_results.json
```

The 26 GB of round-3 rollouts are **not in git**; regenerate them with the first command. `train_arm_v5.py` loads only the needed DAgger columns as float32, which is what makes 1561 rollouts fit into 12 GB of RAM.

| | V5.3-A | V5.4 |
|---|---:|---:|
| 300 validation targets | **280 (93.3 %)** | 265 (88.3 %) |
| 100 random target + spawn (same episodes) | **76** | 75 |
| Best validation MSE HOLD / REACH | **0.032 / 0.300** | 0.080 / 0.469 |

Paired on the validation targets, V5.4 fixes 13 episodes and breaks 28 (exact McNemar p = 0.028). The likely cause is that the 1400 random-target rollouts outnumber the 161 deployment-target rollouts ~9:1 in the aggregated set. **V5.3-A remains the reported checkpoint.** The analysis is in `dagger_section(1).tex`, Sec. "Third DAgger round (V5.4)".

**Known data caveats.**
- The original demonstrations are logged every 10 ms (801 rows / 8 s), while DAgger rollouts and deployment run at 1 ms per row. The arm's lag-10/lag-30 *row* history therefore spans 100/300 ms in the demonstrations and 10/30 ms at deployment.
- The collector does not re-apply the per-episode `base_offset_*` / `arm_offset_*` from `targets.csv`, so "validation episode" evaluations reuse only the targets.

To run the best model on a random target from a random start pose, use:
```bash
pixi run -e ml python run_robotnik_v53.py \
  --random-target \
  --random-spawn \
  --model-dir trained_model_dagger_v5_3_A \
  --output-dir demo_random \
  --episode-id 1 --seed 42 --gui
```
## 7. Known results and current issues

| Paired validation metric (300 episodes) | V5 | V5.3-A |
|---|---:|---:|
| Successes | 263 | **280** |
| Velocity-guard terminations | 22 | **0** |
| Step-cap terminations | 15 | 20 |
| Mean final EE error | 0.1188 m | **0.0953 m** |
| Episodes with logged velocity-jump anomaly | 224 | 231 |

There were **25 fixed episodes, 8 regressions, 12 failures shared by both, and 255 shared successes**. Four regressions (`173, 265, 342, 1932`) approach the navigation goal within approximately 0.057–0.067 m but never set navigation completion and eventually move away. Deployment requires **both** position and orientation tolerances simultaneously; the failed subcondition remains to be verified by logging heading error and actual tolerances. Joint-6 velocity jumps remain frequent even though the 8 rad/s guard no longer terminates episodes.

**Do not equate zero velocity-guard terminations with elimination of velocity spikes.** The anomaly column contains a dictionary-like record, not a simple Boolean; count non-null entries instead.

## 8. Troubleshooting

- **Missing `arm_hold_v4.pth` / metadata:** the V5.3-A model folder is incomplete. Restore the actual trained checkpoints; source code alone is insufficient.
- **`--wheel-source` or `--mode diagnose` unrecognised:** you have an older collector; restore the later V5.3-A collector and inspect `--help`.
- **Missing URDF / config assets:** inspect paths inside `robotnik_arm.json`; run from the expected working directory.
- **No PyBullet window:** ensure graphical Linux desktop access, correct `DISPLAY`/Wayland/X11 support, and `--gui`; headless runs should omit `--gui`.
- **Unexpected results:** verify episode IDs, `--model-dir`, output directory freshness, safety guard settings, and model metadata; check the split for train/validation/test leakage.
- **`navigation_complete` never activates:** log base-goal distance, wrapped orientation error, `pos_tol`, `ang_tol`, and both Boolean conditions. Do not loosen tolerances just to inflate success.
- **Velocity anomaly interpretation:** `previous_executed_torque` is the command applied in the preceding simulation step; `current_predicted_torque` has not yet been executed when the jump is detected.

## 9. Suggested next experiments

1. Instrument position and orientation completion conditions on regressions 173/265/342/1932.
2. Compare expert wheels + V5.3-A arm against learner wheels + V5.3-A arm on those episodes.
3. Inspect error trajectories and phase activation for the nine smaller-error step-cap failures.
4. Examine joint-6 torque and velocity around anomaly timestamps.
5. After model selection, evaluate the untouched 300-episode test split once, with the same stopping and safety criteria.

## Provenance

This README is based on the supplied `robotnik_dagger_v5.zip` source files and subsequent user-provided V5.3-A collector excerpts, commands and evaluation logs. **The final V5.3-A checkpoint and the full latest collector were not provided for independent execution here.** Commands have been matched to the available CLI evidence, but should be checked against your local `--help` and project layout.
