# Inheriting actors between ChairMan stages

`inherit_stage_actor: true` (default) initializes a new stage's actor from
the preceding stage immediately before its first action. For normal
transitions this occurs after the preceding stage succeeds. It does not wait
for a rolling success threshold or for the preceding policy to freeze.

The copy includes `mlp_extractor.policy_net`, `action_net`, and `log_std`.
The destination's independent critic is retained and its optimizer state is
cleared. Source and destination do not share parameter storage; later PPO
updates remain stage-local. Matching MLP actor architectures with standard
flattened observations are required; incompatible actors raise an error.

ChairMan observations include a one-hot stage indicator. The destination
column in the actor's first linear layer is initialized from the source
column, preserving the action mean for an otherwise identical
physical observation despite the changed stage indicator. After copying,
the standard deviation is floored at `inherit_actor_std_min: 0.10`; finger
actions in stages 2 and 4 use `inherit_finger_std_min: 0.20`. This keeps
exploration available when the predecessor has become nearly deterministic.
These values are in policy action units. This applies to
both NumPy and torch observation layouts. Absolute joint-target actions and
the walking controller are unchanged.

The transfer happens once. Trained policies, frozen policies, and policies
with saved initialization metadata are preserved on resume. Each stage's
`actor_initialization` manifest entry records the source and its sample/update
counts at transfer time. Old manifests remain supported: existing training
counters in the manifest or model protect a loaded policy from replacement.

## Whole-task training

```yaml
train_only: false
inherit_stage_actor: true
```

Random snapshot resets are capped at the highest contiguous initialized
stage. A fresh whole-task run starts with cap 0, even if later-stage snapshots
exist on disk. A natural successful transition initializes the next actor
and unlocks that stage for future resets. Resume reconstructs this cap from
the loaded actors' initialization/training metadata.

An explicitly requested single-stage start can bypass this cap. If its
predecessor has no training data, a warning is logged and the destination
starts independently. This fallback is recorded and is not overwritten
after training begins.

## Single-stage training / transfer from a saved bundle

```yaml
train_or_eval: load_and_train
train_only: true
train_stage: 2
inherit_stage_actor: true
load_model_path: ./path/to/multi_policy_bundle
```

The bundle must contain a trained stage-1 source and an unused stage-2 target.
An already trained stage-2 checkpoint resumes its own actor instead of being
replaced. `train_only` changes only the selected policy. A fresh `train` run
starting directly at stage 2 has no trained stage-1 actor to inherit: a
simulation snapshot contains physical state, not a trained controller.

Set `inherit_stage_actor: false` to use independent actor initialization.
The copied source may still favor behavior from its old task (for example,
open fingers); inheritance is an initialization, not evidence that the next
task is solved. Completion rewards, stage terminations, and per-stage PPO
buffers retain their existing behavior.

Tests: `python -m unittest config_run.test_multi_ppo_actor_inheritance
config_run.test_multi_ppo_trainer config_run.test_chairman_multi_reward_pipeline
config_run.test_SB3_chairman_multi_env`.
