1) Make sure the folders "/models", "/configs" from RoboEnv are accessible (easiest is to copy them to this folder)
2) Use your RoboEnv environent (in whichever way you have been running the labs: pixi, uv, conda, etc)
3) Run robotnik_arm_task.py

Expected behaviour: pybullet 3D environment with 7dof arm mounted on a 4-wheeled platform. The robot moves to reach a target (red dot) with its end-effector. The simulation terminates when the target is reached, and plots of the robot behaviour are shown.

Key variables in robotnik_arm_task.py:

 - TARGET_EE_POS: reach target point in cartesian coordinates (x,y,z), represented by a red dot.

 - base_pos:     wheeled base position (x,y,z)
 - base_bearing: wheeled base bearing (angle in radians)
 - q_mes:        arm joint angles (radians)
 - qd_mes:       arm joint angle velocities (rad/s)

 - wheel_cmds:     angular velocity command for each wheel (rad/s)
 - torque_arm:     command torques for each arm joint
 - torque_fingers: torque for each finger joint of the end-effector grasper. In this work, they should just remain fixed.
 
 Assessment Goal:
  - ML model that produces (wheel_cmds, torque_arm) from measurements (base_pos,base_bearing,q_mes,qd_mes)

To record training data as CSV run:

```bash
uv run robotnik_arm_task/collect_data.py
```

Instead of uv run use whatever environment you have set up in your system.

Edit `NUM_TARGETS`, `SEED`, and `OUTPUT_DIR` at the top of `collect_data.py`. The default output is `robotnik_arm_task/data`. It contains `targets.csv` and one CSV per episode. Each row records the target, base position and bearing, seven arm joint angles and velocities, four expert wheel commands, and seven expert arm torques. `time` and `ee_error` are metadata;

Currently, the simulation is started with the same initial state, so to make the data more better, we can add changes in the initial state as well.

After data collection finishes, train a linear regression model with:

```bash
uv run robotnik_arm_task/train_linear_regression.py
```

The model takes 21 values: base position and bearing, seven arm joint angles and velocities, and target position. It predicts seven arm torques followed by four wheel velocity commands. `time` and `ee_error` are excluded from training.

Like the week 1 lab, training uses the pseudoinverse: `weights = np.linalg.pinv(X_train) @ y_train`. A column of ones is prepended to the inputs to learn an intercept, so `weights` has shape `(22, 11)`. Prediction is `X_test @ weights`, with the same column of ones prepended to test inputs.

The script uses 80% of episodes for training and 20% for testing, and prints separate root mean squared errors (RMSE) for arm torques and wheel commands. It saves `linear_regression.npz`.
It also saves results in `robotnik_arm_task/training_results/`:

- `metrics.csv`: training and test RMSE and R² for each of the 11 commands.
- `metrics.png`: training/test comparisons of RMSE and R², with arm and wheel units kept separate.
- `train_predictions.png` and `test_predictions.png`: expert and predicted commands over time for one episode from each split. Plotting uses about 2,000 points per episode; metrics use every row.


To run the simulation using the trained model:

```bash
uv run robotnik_arm_task/robotnik_arm_model.py
```

This is a separate version of `robotnik_arm_task.py` with the same simulation setup. It loads the saved model from `trained_model/linear_regression.npz` once and predicts arm torques and wheel velocities from the current state and target at every step. Arm torques retain the original ±30 N m limit, and fingers stay fixed. Set `TARGET_EE_POS` in the new script to change the target. The existing stopping criteria use measured base position/bearing (or the navigation timeout) to determine base arrival; no expert controller generates commands. The plots show end-effector tracking/error and measured joint angles; the model does not produce desired joint angles.

The model runner limits wheel predictions to ±3.18 rad/s, matching the expert's maximum wheel speed. 
