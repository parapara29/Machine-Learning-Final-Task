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
