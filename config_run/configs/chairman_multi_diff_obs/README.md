# ChairMan multi observation-space experiments

These configs are small overlays on the corresponding files in
`configs/chairman_multi`. `main_multi_diff_obs.py` resolves `base_config` first
and then applies the values in the selected overlay.

Supported `sb3_env_file` values:

- `SB3_chairman_multi_env.py`: current full 328-dimensional observation.
- `SB3_chairman_multi_relative_env.py`: 56 values (43 joints, pelvis velocity
  and chair pose relative to the pelvis).
- `SB3_chairman_multi_absolute_env.py`: 63 values (43 joints, pelvis
  pose/velocity and chair pose in the world frame).

Keep `seed` equal across experiments. Use a separate output directory for each
variant so that checkpoints and TensorBoard logs cannot overwrite each other.
