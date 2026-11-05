# Robotnik Mobile Manipulation — Decoupled Linear Regression

This repository contains the **Linear Regression** solution for the UCL COMP0245 Mobile Manipulation assignment. 

The goal is to control a mobile manipulator (a 4-wheeled skid-steer base + 7-DOF Kinova robotic arm) in PyBullet to drive across the room and touch a 3D target with the arm's gripper.

Instead of using a black-box neural network, we solved this with **two lightweight Linear Regression models** running at **0.025 ms per step** (theoretical 40,000 Hz) with **83.6% strict accuracy at $\le 1\text{ cm}$** and **96.8% success at $\le 5\text{ cm}$** across 1,000 randomized test trials.

---

## 1. Why Simple "Input In, Output Out" Failed (The 4 Traps)

When we first tried training a single linear regression model to predict all 11 motor commands from all 48 raw state features, the robot slammed into its own base and flew into the air. Here is why:

1. **Regime Mixing (Two different jobs):** In 77.4% of the expert data, the wheels drive while the arm stays locked at home. In only 22.6% does the arm actually reach. Fitting one line across both pulls down the arm slope by $3\times$ and learns fake couplings (the arm swings while the base is still meters away).
2. **Room Coordinates Flip Sign:** If the robot faces East, moving forward is $\Delta x = +1$\,m. If it turns $180^\circ$ to face West, moving forward is $\Delta x = -1$\,m. World coordinates have zero linear correlation with wheel speeds ($r = -0.01$). Rotating into the robot's local body frame fixes this completely ($r = +0.79$).
3. **The "Anti-Spring" Explosion:** The raw data included both joint angles $q$ and gripper position $\mathbf{x}_{\text{ee}}$. Because $\mathbf{x}_{\text{ee}} = f(q)$, multicollinearity caused linear regression to learn a **positive stiffness** ($+11.1$\,N$\cdot$m/rad) on Shoulder Joint 2. A positive stiffness is an anti-spring: whenever gravity sags the arm down, the motor pushes down *harder*, causing an exponential runaway loop that slams into the metal base.
4. **Elbow-Twist Bifurcation:** For targets below $Z = 0.46$\,m, Shoulder Joint 2 hits its bottom mechanical limit ($0.87$\,rad) and freezes. The expert reaches lower by twisting the elbow roll joint ($q_6$) either left ($+0.55$\,rad) or right ($-0.60$\,rad). Linear regression averages both mirror postures to $0.0$\,rad, so the arm reaches straight down without bending and gets stuck.

---

## 2. Environment Setup

Make sure your `roboenv` virtual environment (or whichever environment you use with PyBullet, Pinocchio, and `simulation_and_control`) is active.


Ensure the configuration and robot description folders are accessible:
- `robotnik_arm_task/configs/robotnik_arm.json`
- `models/` (Kinova Jaco2 and Robotnik Summit-XL URDFs and meshes)

---

## 3. Project Structure: Which Files Are Actually Used?

You only need a few key scripts. Here is the active pipeline:

```
final/
├── README.md                              <- You are here!
├── trained_model/
│   ├── base_linear_regression.npz         <- Trained 15-feature base weights (16 x 4)
│   └── arm_linear_regression.npz          <- Trained 25-feature arm weights (26 x 7)
└── robotnik_arm_task/
    ├── analyze_data.py                    <- Data analysis & Pearson correlation plots
    ├── collect_data.py                    <- Collects expert demonstration CSVs
    ├── train_base_model.py                <- Trains Phase 1 (Base Navigation) model
    ├── train_arm_model.py                 <- Trains Phase 2 (Arm Reaching) model
    ├── robotnik_combined_model.py         <- Runs 1 episode (GUI or headless)
    ├── test_combined_random.py            <- 1,000-trial parallel benchmark ([2.0, 4.0]m)
    └── combined_test_results/
        └── test_summary.png               <- Latest evaluation histogram & scatter plot
```

*(Note: `train_linear_regression.py` and `robotnik_arm_model.py` are legacy monolithic files kept for baseline comparison and failure analysis).*

---

## 4. How to Train the Models

Both models train in under **2 seconds** using closed-form Ordinary Least Squares (`np.linalg.pinv`):

### Step 1: Train the Mobile Base Model
```bash
uv run robotnik_arm_task/train_base_model.py
```
* **Inputs (15 features):** Saturated body-frame goal offsets ($g_f, g_l \in [-0.35, 0.35]$\,m), saturated distance, alignment gate (`drive_term` inspired by Aicardi et al. Lyapunov unicycle steering that locks forward drive until heading error $< 31.8^\circ$), regime blending weights (`near_goal`), heading steering terms, and docking bearing terms.
* **Outputs (4 wheel velocities):** Commands the 4 skid-steer wheels to turn in place, cruise, and brake at the parking spot.
* **Saved to:** `trained_model/base_linear_regression.npz`

### Step 2: Train the Arm Torque Model
```bash
uv run robotnik_arm_task/train_arm_model.py
```
* **Inputs (25 features):** 7 joint angles $q_{4..10}$ (omitting $\mathbf{x}_{\text{ee}}$ to stop the anti-spring), 7 joint velocities $\dot{q}_{4..10}$ (for $-K_d \dot{q}$ damping), 3 body-frame target offsets ($t_f, t_l, t_h$), 6 quadratic interaction terms ($t_f^2, t_l^2, t_h^2, t_f t_h, t_f t_l, t_l t_h$), and 2 low-$Z$ hinge features ($\text{low\_z} = \max(0, 0.46 - t_h)$ and $\text{low\_z}^2$).
* **DART State Perturbation:** Adds Gaussian noise to states during training so the model learns restoring spring forces ($-100$\,N$\cdot$m/rad) across all 7 joints instead of drifting off-trajectory.
* **Filters Bifurcation:** Only trains on the positive elbow twist branch ($q_6 \ge +0.2$\,rad) for low targets.
* **Saved to:** `trained_model/arm_linear_regression.npz`

---

## 5. How to Run and Test

### A. Run a Single Episode in PyBullet
To watch the robot drive to a target and reach it with its arm:
```bash
uv run robotnik_arm_task/robotnik_combined_model.py
```
* **Headless Mode:** If you don't need the GUI, run `ROBOENV_HEADLESS=1 uv run robotnik_arm_task/robotnik_combined_model.py`.
* **Change the Target:** Open `robotnik_combined_model.py` and edit line 27:
  ```python
  TARGET_EE_POS = np.array([2.4, -2.4, 0.8])  # [X, Y, Z] in meters
  ```

### B. Run the 1,000-Episode Parallel Benchmark
To run the full random benchmark across all CPU cores (12 workers):
```bash
python robotnik_arm_task/test_combined_random.py
```
* Tests 1,000 episodes with random base start positions ($\pm 2.0$\,m), random initial compass headings ($-\pi$ to $+\pi$), and random 3D targets across $[2.0, 4.0]$\,m.
* Automatically plots and saves the error histogram and spatial reach map to `robotnik_arm_task/combined_test_results/test_summary.png`.

---

## 6. Latest Benchmark Results

From the latest 1,000-trial test across $[2.0, 4.0]$\,m:

| Metric | Decoupled Linear Regression | Expert Demonstration Controller |
| :--- | :--- | :--- |
| **Strict Success ($\le 1.0\text{ cm}$)** | **83.60%** (836 / 1,000) | 62.95% |
| **Task Success ($\le 5.0\text{ cm}$)** | **96.80%** (968 / 1,000) | 97.25% |
| **Median Error** | **3.8 mm** | 7.1 mm |
| **Inference Time / Step** | **0.025 ms** ($25\,\mu\text{s}$) | ~1.0 ms |
| **Training Time** | **< 2 seconds** | N/A |
| **Total Parameters** | **246 weights** | N/A |

### Known Limitation: Targets Closer than 1.8m
If a target is closer than $1.8$\,m and the robot starts facing the opposite direction ($180^\circ$ yaw):
* After subtracting the arm's $0.70$\,m standoff, the base parking goal is only $15\text{--}30$\,cm from the chassis.
* Spinning a 4-wheel skid-steer base on high friction causes $10\text{--}15$\,cm of lateral slip, which pushes the robot past the goal before it finishes turning.
* The robot then enters an orbit limit cycle around the goal until timeout.
* **Fix:** In an actual mobile robot pipeline, if the base goal is within $35$\,cm, you simply lock forward drive until heading alignment is under $5^\circ$, or hand off directly to the arm if the target is already within the reachable envelope.
