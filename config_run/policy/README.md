# Unitree G1 walking policy

`G1MotionPolicy` wraps the recurrent `motion.pt` policy from
[`unitreerobotics/unitree_rl_gym`](https://github.com/unitreerobotics/unitree_rl_gym).
It builds the original 47-value observation and returns 12 target leg angles in
radians.

```python
from policy import G1MotionPolicy

walking_policy = G1MotionPolicy()  # loads motion.pt from this directory

leg_targets = walking_policy.predict_joint_positions(
    joint_positions=q_legs,             # (12,) or (num_envs, 12), radians
    joint_velocities=dq_legs,           # rad/s, same order
    angular_velocity=pelvis_gyro,       # body-frame rad/s, (..., 3)
    base_quaternion_wxyz=pelvis_quat,   # WXYZ, (..., 4)
    command=[0.5, 0.0, 0.0],           # vx, vy [m/s], yaw rate [rad/s]
)
```

The order of both input leg state and output targets is available as
`G1MotionPolicy.JOINT_NAMES`. For a single robot, mappings can be passed in and
returned directly:

```python
leg_target_dict = walking_policy.predict_joint_dict(
    joint_positions=q_by_joint_name,
    joint_velocities=dq_by_joint_name,
    angular_velocity=pelvis_gyro,
    base_quaternion_wxyz=pelvis_quat,
    command=[0.5, 0.0, 0.0],
)
action_dict[robot_name]["dof_pos_target"].update(leg_target_dict)
```

For a vectorized environment, instantiate the wrapper once and pass arrays with
shape `(num_envs, ...)`. Call `walking_policy.reset(env_ids)` whenever those
environments reset; this clears their LSTM state, previous action, and gait
phase. Calling `reset()` clears all environments.

Important: Chairman configurations that use `g1_slider` or
`g1_slider_simple` expose slider/arm joints and not these 12 leg joints. Before
merging this output into `SB3_chairman_env`, select a robot configuration and
URDF that contain all names in `G1MotionPolicy.JOINT_NAMES` (for example the
full `g1_with_hands` configuration).

The integrated `SB3_chairman_env` exposes no direct leg actions. Its action
vector is:

```text
[31 waist/arm/wrist/finger position targets, vx, vy, yaw_rate]
```

The exact runtime order is available as `env.action_names`. The last three
values use m/s, m/s, and rad/s and are limited to `[0.8, 0.5, 1.57]` in
absolute value. `SB3_chairman_env` passes them into this walking policy, merges
the resulting 12 leg targets with the 31 upper-body targets, and sends all 43
joint positions to the simulated `g1_with_hands` robot.

The wrapper follows the upstream constants: 50 Hz policy rate, 0.8 s gait
period, joint velocity scale 0.05, gyro scale 0.25, command scale
`[2, 2, 0.25]`, and action-to-position scale 0.25. Inputs are validated and
targets are clipped to the G1 joint limits by default. This clipping is only a
last software guard; real-robot deployment still requires the Unitree safety
procedure, PD/torque limits, an emergency stop, and initial hoisting.
