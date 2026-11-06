# Robotnik Mobile Manipulation — Decoupled Linear Regression

This repository contains the **Linear Regression** solution for the UCL COMP0245 Mobile Manipulation assignment. 

The goal is to control a mobile manipulator (a 4-wheeled skid-steer base + 7-DOF Kinova robotic arm) in PyBullet to drive across the room and touch a 3D target with the arm's gripper.

Instead of relying on a black-box neural network, this project uses **two decoupled, physically engineered Linear Regression models** running at **0.025 ms per step** (theoretical 40,000 Hz) with **83.60% strict success at $\le 1\text{ cm}$** and **96.80% task success at $\le 5\text{ cm}$** across 1,000 randomized test trials.

---

## 1. Environment Setup (`uv` and `pixi`)

We recommend using Python 3.10. A standard `requirements.txt` is provided in the repository root.

### Option A: Using `uv` (Recommended)

`uv` is extremely fast and manages virtual environments directly:

```bash
# 1. Create a virtual environment named roboenv
uv venv roboenv --python 3.10

# 2. Activate the virtual environment
source roboenv/bin/activate

# 3. Install core dependencies
uv pip install -r requirements.txt
```
Ensure the configuration and robot description folders are accessible:
- `robotnik_arm_task/configs/robotnik_arm.json`
- `models/` (Kinova Jaco2 and Robotnik Summit-XL URDFs and meshes)

---

## 3. Project Structure: Which Files Are Actually Used?

To maintain academic integrity and complete transparency, here is the breakdown of external course code, modified components, and code developed from scratch:

### A. External Course Infrastructure (Provided Starter Code)
These files were provided as part of the UCL COMP0245 Mobile Manipulation assignment harness:
* **`robotnik_arm_task/robotnik_arm_task.py`:** Expert simulation harness that sets up the PyBullet physics engine, loads URDFs, manages 1,000 Hz time-stepping, places 3D targets, and sets camera views.
* **`robotnik_arm_task/robotnik_arm_control.py`:** Expert demonstration controller. Implements base navigation via unicycle steering, arm differential inverse kinematics with Damped Least Squares (DLS) using Pinocchio, joint PD control, gravity compensation, and finger gripper holding.
* **`robotnik_arm_task/configs/` & `robotnik_arm_task/models/`:** Configuration JSONs and 3D URDF/mesh assets for the Robotnik Summit-XL chassis and Kinova arm.
* **`simulation_and_control` library:** Course helper library providing PyBullet wrapper utilities (`pb`), `MotorCommands` data structures, and the `PinWrapper` dynamics interface.

### B. Modified Code
* **Harness Hooks:** Minor compatibility adjustments were made to `robotnik_arm_task.py` and `robotnik_arm_control.py` to support headless multi-worker simulation via the `ROBOENV_HEADLESS` environment variable, ensuring background benchmarking processes terminate cleanly without opening 12 OpenGL GUI windows. The core simulation physics and expert control laws were kept strictly unmodified.

### C. Developed From Scratch (Our Linear Regression Pipeline)
All machine learning, feature engineering, training, and evaluation scripts were written entirely from scratch for this project:
* **`collect_data.py`:** Multiprocessing demonstration collector that parallelizes expert runs across CPU cores and logs 2,000 randomized episodes at 100 Hz into CSV files with boundary safety checks.
* **`train_base_model.py`:** Phase 1 data filtering, 15 body-frame feature extraction, Aicardi Lyapunov unicycle steering gate, 4 cm coasting deadband, Moore-Penrose pseudoinverse OLS solver, and weight exporter.
* **`train_arm_model.py`:** Phase 2 data filtering, mirror posture bifurcation pruning, 25 body-frame feature extraction (omitting $\mathbf{x}_{\text{ee}}$ to stop the anti-spring explosion, adding $\dot{q}$ damping, quadratic workspace interaction terms, and low-$Z$ elbow hinge), DART state perturbation noise injection, and OLS solver.
* **`robotnik_combined_model.py`:** Runtime mobile manipulation controller executing the two decoupled linear models, handling the empirical sweet-spot regime transition ($0.60 \le t_f \le 0.85$\,m, $|t_l| \le 0.20$\,m), and enforcing motor velocity/torque limits.
* **`test_combined_random.py` & `test_combined_min_goal_dist.py`:** Multiprocessing benchmark harness evaluating 1,000 randomized episodes, computing success rates at $1\text{ cm}$ and $5\text{ cm}$, and plotting diagnostic figures.
* **`analyze_data.py`:** Dataset audit and visualization script generating Pearson correlation matrices, feature distributions, and failure diagnosis plots.

---

## 3. Why Simple "Input In, Output Out" Failed (The 4 Traps)

When we first attempted to train a single linear regression model predicting all 11 motor commands from all 48 raw state features, the robot slammed into its own chassis and flew into the air. Here is why:

1. **Regime Mixing (Two completely different jobs):** In 77.4% of the expert data, the wheels drive while the arm is locked at home. In only 22.6% does the arm reach while wheels are stopped. Fitting one global line across both pulls down the arm torque slope by $3\times$ and learns false couplings (the arm flails while the base is still meters away).
2. **Room Coordinates Flip Sign:** If the robot faces East, moving forward is $\Delta x = +1$\,m. If it turns $180^\circ$ to face West, moving forward is $\Delta x = -1$\,m. World coordinates have zero linear correlation with wheel speeds ($r = -0.01$). Rotating into the robot's local body frame fixes this completely ($r = +0.79$).
3. **The "Anti-Spring" Explosion:** The raw state included both joint angles $q$ and gripper position $\mathbf{x}_{\text{ee}}$. Because $\mathbf{x}_{\text{ee}} = f(q)$, multicollinearity caused linear regression to learn a **positive stiffness** ($+11.1$\,N$\cdot$m/rad) on Shoulder Joint 2. A positive stiffness is an anti-spring: whenever gravity pulls the arm down, the motor pushes down *harder*, causing an exponential runaway loop that slams into the metal base.
4. **Elbow-Twist Bifurcation:** For targets below $Z = 0.46$\,m, Shoulder Joint 2 hits its bottom mechanical limit ($0.88$\,rad) and freezes. The expert reaches lower by twisting the elbow roll joint ($q_6$) either left ($+0.55$\,rad) or right ($-0.60$\,rad). Linear regression averages both mirror postures to $0.0$\,rad, so the arm reaches straight down without bending and gets stuck.

---

## 4. How to Run the Code & Generated Outputs

Here is the complete step-by-step pipeline from data collection to testing, including every output file generated.

```
final/
├── requirements.txt                       <- Environment dependencies
├── README.md                              <- Project documentation
├── trained_model/
│   ├── base_linear_regression.npz         <- Trained 15-feature base weights (16 x 4)
│   └── arm_linear_regression.npz          <- Trained 25-feature arm weights (26 x 7)
└── robotnik_arm_task/
    ├── collect_data.py                    <- Data collection
    ├── train_base_model.py                <- Base training
    ├── train_arm_model.py                 <- Arm training
    ├── robotnik_combined_model.py         <- Single-run controller
    ├── test_combined_random.py            <- 1,000-trial parallel benchmark
    ├── analyze_data.py                    <- Correlation analysis
    ├── base_training_results/             <- Plots & metrics for base model
    ├── arm_training_results/              <- Plots & metrics for arm model
    └── combined_test_results/             <- Final benchmark summary plots
```

---

### Step 1: Collect Expert Demonstration Data (Optional)
> *Pre-collected data (2,000 episodes) is already included in `robotnik_arm_task/data_randomized/`.*

If you want to collect fresh training data:
```bash
uv run robotnik_arm_task/collect_data.py
```
* **What it does:** Launches parallel expert simulations across CPU cores, varying robot starting poses and target locations.
* **Outputs Generated:**
  - `robotnik_arm_task/data_randomized/episode_0000.csv` to `episode_1999.csv`: 2,000 CSV files containing 100 Hz logs of base states, joint angles, wheel velocities, arm torques, and gripper error.

---

### Step 2: Train the Mobile Base Model
```bash
uv run robotnik_arm_task/train_base_model.py
```
* **Execution Time:** ~1.6 seconds.
* **What it does:** Filters for Phase 1 driving rows, computes 15 body-frame features (0.35m saturation, Aicardi Lyapunov alignment gate, 4 cm coasting deadband, near/far blending), and solves OLS using the Moore-Penrose pseudoinverse.
* **Outputs Generated:**
  - `trained_model/base_linear_regression.npz`: Saved $16 \times 4$ weight matrix and feature normalization statistics.
  - `robotnik_arm_task/base_training_results/metrics.csv`: Train and test RMSE and $R^2$ scores for all 4 wheels.
  - `robotnik_arm_task/base_training_results/metrics.png`: Bar charts showing prediction accuracy per wheel.
  - `robotnik_arm_task/base_training_results/train_predictions.png`: Time-series plots of predicted vs expert wheel speeds on training episodes.
  - `robotnik_arm_task/base_training_results/test_predictions.png`: Time-series plots on unseen test episodes.
  - `robotnik_arm_task/base_training_results/wheel_forward_relations.png` & `wheel_turn_relations.png`: Feature correlation scatter plots.

---

### Step 3: Train the Arm Model
```bash
uv run robotnik_arm_task/train_arm_model.py
```
* **Execution Time:** ~28 seconds (mostly disk I/O loading 2,000 CSVs; the OLS fit itself takes $< 0.2$ seconds).
* **What it does:** Filters for Phase 2 arm rows, drops failed timeouts, prunes the negative elbow branch, constructs 25 body-frame features, injects DART Gaussian noise to teach $-100$\,N$\cdot$m/rad restoring stiffness, computes the empirical sweet-spot reach envelope, and solves OLS.
* **Outputs Generated:**
  - `trained_model/arm_linear_regression.npz`: Saved $26 \times 7$ weight matrix, input normalization statistics, and empirical sweet-spot boundaries ($t_f \in [0.60, 0.85]$\,m, $|t_l| \le 0.20$\,m).
  - `robotnik_arm_task/arm_training_results/metrics.csv`: Train and test RMSE and $R^2$ scores across all 7 joints.
  - `robotnik_arm_task/arm_training_results/metrics.png`: Bar charts of torque prediction performance.
  - `robotnik_arm_task/arm_training_results/arm_workspace.png`: Top-down and side views of reached target positions relative to the base, showing the sweet-spot envelope.
  - `robotnik_arm_task/arm_training_results/train_predictions.png` & `test_predictions.png`: Predicted vs expert torque profiles over time.

---

### Step 4: Run a Single Test Episode (Watch the Robot in PyBullet)
```bash
# With PyBullet 3D GUI window:
uv run robotnik_arm_task/robotnik_combined_model.py

# Or in Headless mode (fast, no window):
ROBOENV_HEADLESS=1 python robotnik_arm_task/robotnik_combined_model.py
```
* **What it does:** Initializes the robot at $(0, 0)$, drives toward the 3D target using the base model, executes a smooth handover once inside the arm sweet spot, and deploys the arm model to touch the target.
* **Changing the Target:** Open `robotnik_combined_model.py` and modify line 46:
  ```python
  TARGET_EE_POS = np.array([2.4, -2.4, 0.8])  # [X, Y, Z] in meters
  ```
* **Outputs Generated:**
  - Real-time terminal output reporting navigation distance, docking transition, and final End-Effector Error (m).
  - If the target is reached within 5 cm, reports `SUCCESS: Target reached!`.

---

### Step 5: Run the 1,000-Trial Parallel Benchmark
```bash
uv run robotnik_arm_task/test_combined_random.py
```
* **Execution Time:** ~3–4 minutes across 12 CPU cores.
* **What it does:** Tests 1,000 fully randomized episodes:
  - Random base starting positions shifted by up to $\pm 0.25$\,m.
  - Random initial compass headings across the full $360^\circ$ ($-\pi$ to $+\pi$).
  - Random 3D targets distributed between $2.0$\,m and $4.0$\,m away, at heights from $0.3$\,m to $0.9$\,m.
  - Random initial arm joint perturbations up to $\pm 0.1$\,rad.
* **Outputs Generated:**
  - Real-time multiprocessing progress bar and summary table printed to the console.
  - `robotnik_arm_task/combined_test_results/test_summary.png`:
    - **Top Plot:** Histogram of final End-Effector errors showing median error ($3.8$\,mm) and the $1\text{ cm}$ / $5\text{ cm}$ success thresholds.
    - **Bottom Plot:** Top-down spatial reach scatter plot showing green dots (successes) and red crosses (fails) across the entire room.

---

### Step 6: Run the Adversarial Robustness Benchmark (100 Stress Tests)
```bash
uv run robotnik_arm_task/test_adversarial_robustness.py
```
* **What it does:** Tests 100 randomized episodes (25 per category) across 4 adversarial and disturbance conditions:
  1. **Arm Shove:** Sudden $30\,\text{N}$ external force pulse on the gripper mid-reach to test restoring stiffness.
  2. **Target Jump Midway:** Dynamic $0.6\text{--}1.0\,\text{m}$ target relocation en route to test dynamic re-planning.
  3. **Unmodeled Arm Payload:** $+0.5\,\text{kg}$ downward gravitational load attached to the end-effector.
  4. **Base Displacement:** $0.4\,\text{m}$ lateral base kick after parking to test closed-loop re-docking recovery.
* **Outputs Generated:**
  - Live console progress and category breakdown table (success rates at $\le 5\,\text{cm}$ and $\le 1\,\text{cm}$, median errors).
  - `robotnik_arm_task/adversarial_results/adversarial_results.csv`: Complete log of every test case and outcome.
  - `robotnik_arm_task/adversarial_results/adversarial_summary.png`: Publication-quality comparison bar chart and error distribution scatter plots.

---

## 5. Latest Benchmark Results

From the comprehensive 1,000-trial random evaluation across $[2.0, 4.0]$\,m:

| Metric | Decoupled Linear Regression | Expert Demonstration Controller |
| :--- | :--- | :--- |
| **Strict Success ($\le 1.0\text{ cm}$)** | **83.60%** (836 / 1,000) | 62.95% |
| **Task Success ($\le 5.0\text{ cm}$)** | **96.80%** (968 / 1,000) | 97.25% |
| **Median Error** | **3.8 mm** | 7.1 mm |
| **Step Latency** | **0.025 ms** ($25\,\mu\text{s}$) | ~1.0 ms |
| **Training Time** | **< 2 seconds** (OLS math) | N/A |
| **Total Parameters** | **246 weights** | N/A |

### Known Physical Limitation: Close Targets with $180^\circ$ Heading Error
When targets are closer than $1.8$\,m and the robot starts facing directly away from the target ($180^\circ$ initial heading):
1. After subtracting the arm's $0.70$\,m standoff, the base parking goal is only $15\text{--}30$\,cm from the robot.
2. Turning a 4-wheel skid-steer chassis in place on high friction induces $10\text{--}15$\,cm of lateral scrub drift.
3. This drift pushes the robot slightly past the goal before it finishes aligning its heading, flipping the heading error sign and trapping the base in an orbit limit cycle until timeout.
4. **Engineering Recommendation:** For production mobile manipulators, targets within $35$\,cm of the base should lock forward drive until heading error is $< 5^\circ$, or immediately hand over to the arm if the target already falls inside the reachable workspace envelope.
