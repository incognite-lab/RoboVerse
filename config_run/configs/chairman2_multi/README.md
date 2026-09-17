# Chairman2: five stage policies

Run from the repository root in the training environment:

```bash
python config_run/main_multi.py chairman2_multi/train_ppo
python config_run/main_multi.py chairman2_multi/load_and_train_ppo
python config_run/main_multi.py chairman2_multi/eval_ppo_video
```

For resume/evaluation set `load_model_path` to a new bundle directory containing
`multi_policy_manifest.json` and `stage_0` through `stage_4`. Five-stage bundles
carry `task_version: chairman2_five_stage_v1`; incompatible old bundles are
rejected. Old files are retained. Models and snapshots use separate directories:

- `config_run/output/ppo_models_chairman2_five_stage_v1`
- `config_run/snapshots_chairman2_five_stage_v1`

The task uses `ChairMan_2.Chairman2Cfg`, `_ChairMan2Checker`,
`stages_chairman2`, and `SB3_chairman2_env`. `g1_without_hands` selects
`g1_mygym_without_hand.urdf`. Policies output ten arm angles and three walking
commands; the motion controller drives the legs. The observation includes stage
anchors, fixed pull direction, remaining pull distance, and elapsed stage time.

| Stage | Goal | Main checker conditions | Hold | Timeout |
|---|---|---|---|---|
| 0 | Walk behind chair with specified walking arm pose | within 5 cm of target 0.745 m from chair base, every arm joint within 0.15 rad, torso within 5 degrees, robot/chair stopped | 0.25 s | 15 s |
| 1 | Extend arms to specified pose | every joint within 0.15 rad, robot drift <=5 cm, chair drift <=3 cm, stopped and facing chair, arm speed <=0.20 rad/s | 0.25 s | 6 s |
| 2 | Place both palms on backrest | both actual contacts in backrest target regions, palms down and elbows straight within 10 degrees, robot/chair stationary, limited force/slip | 0.25 s | 8 s |
| 3 | Pull chair backward 1 m | distance/lateral error <=5 cm, chair yaw <=5 degrees, both contacts, palms down, straight elbows, robot/chair stopped | 0.40 s | 20 s |
| 4 | Lift both hands | both palm surfaces >=10 cm above targets, within 10 cm horizontally, no hand-chair contact, anchors and stillness maintained | 0.25 s | 6 s |

Stillness uses XY speed <=0.08 m/s and Z angular speed <=0.10 rad/s.
Contact must carry >=0.5 N; success requires <=60 N per hand and relative
hand speed <=0.08 m/s. Contact gaps in stage 3 are tolerated for 0.10 s;
positive pull shaping remains disabled during a gap. Larger robot/chair drift,
fall, invalid state, or timeout fails the stage. Small errors receive penalties.
The checker measures actual motion, not just requested walking commands.

`metasim/utils/chairman2_geometry.py` contains shared goals, thresholds and
geometry. Contact uses solver contact positions near each backrest target;
without contact positions it requires actual contact and palm surface proximity.
Distance alone never creates a contact. Palm surface points are local
`[0.05, -0.02, 0]` (left) and `[0.05, 0.02, 0]` (right), with outward normals
-Y and +Y. Elbow straightness is the angle between shoulder-to-elbow and
elbow-to-wrist vectors, rather than an assumed zero joint angle.

Each stage has a dedicated logged reward. Its dense term is
`4 * (previous_cost - cost) - (0.02 + 0.15 * cost) * dt / 0.02`.
Costs combine average and worst-hand/joint error, heading, stillness and
stage-specific constraints. Waiting has a negative reward. Completion gives
+10 once per transition; final completion gives +20, failure -10. Positive
stage-3 shaping additionally requires both contacts and correct palm/elbow
geometry. Separate small penalties regularize rapid arm commands, joint
velocity, walking-command changes and torso tilt. Thresholds/weights are initial
engineering settings; contact force and surface offsets should be checked in
simulation before long training.

Policies switch without resetting the simulator on intermediate success.
Actor inheritance, freezing, curriculum caps and `train_only: true` /
`train_stage: 0..4` remain available. Stages above zero require a new-schema
snapshot. The first successful entry into each new stage is retained even
when snapshot sampling is enabled. `main.py` remains a single-policy entry;
Chairmanmulti remains a separate six-policy task.
